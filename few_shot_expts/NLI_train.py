import argparse
import math
import random
import os
import sys
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
import torchvision.models as models
from tqdm import tqdm
# import viz
from copy import deepcopy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from glob import glob
from PIL import Image

from NLI_model import Generator
from model import Discriminator
from NLI_model import NLI

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
                (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_loss(cnn, normalization_mean, normalization_std, target_img, style_layers, device):

    style_losses = []
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(target_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses


def train(args, nli, nli_optim, g_ema, d, d_optim, img_trg, w_trg, w_aux, imsave_path, model_path, expt_number, device):

    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    vgg19 = models.vgg19(pretrained=True).features.to(device).eval()

    style_weight = 50
    d_factor = 1

    requires_grad(g_ema, False)
    requires_grad(d, True)

    for idx in pbar:
        nli.train()
        requires_grad(nli, True)
        i = idx + args.start_iter
        g_factor = 5 * (1 - i / args.iter)

        if i > args.iter:
            print("Done!")
            break

        loss_dict = {}

        t_img_idx = np.random.default_rng().choice(len(img_trg), size=args.adv_bs, replace=len(img_trg) < args.adv_bs)
        trg_sample = img_trg[t_img_idx]
        t_w_idx1 = np.random.default_rng().choice(len(w_trg), size=args.adv_bs, replace=len(w_trg) < args.adv_bs)
        t_w_idx2 = np.random.default_rng().choice(len(w_trg), size=args.adv_bs, replace=len(w_trg) < args.adv_bs)
        w_mixed = nli(torch.cat([w_trg[t_w_idx1], w_aux[t_w_idx2]], 2))
        ipol_sample, _ = g_ema([w_mixed], input_is_latent=True, return_feats=False)
        real_pred = d(trg_sample)
        fake_pred = d(ipol_sample)

        d_loss = d_factor * d_logistic_loss(real_pred, fake_pred)

        d.zero_grad()
        d_loss.backward()
        d_optim.step()

        w_mixed = nli(torch.cat([w_trg[t_w_idx1], w_aux[t_w_idx2]], 2))
        ipol_sample, _ = g_ema([w_mixed], input_is_latent=True, return_feats=False)
        fake_pred = d(ipol_sample)

        g_loss = g_factor * g_nonsaturating_loss(fake_pred)

        nli.zero_grad()
        g_loss.backward()
        nli_optim.step()

        t_w_idx = np.random.default_rng().choice(len(w_trg), size=args.sty_bs, replace=len(w_trg) < args.sty_bs)
        for t_id in t_w_idx:
            t_img_id = np.random.randint(0, args.n_shot)
            target_sample = img_trg[t_img_id].unsqueeze(0)

            w_mixed = nli(torch.cat([w_trg[t_id].unsqueeze(0), w_aux[t_img_id].unsqueeze(0)], 2))
            interpolated_sample, _ = g_ema([w_mixed], input_is_latent=True, return_feats=False)

            model, style_losses = get_style_model_and_loss(vgg19, cnn_normalization_mean, cnn_normalization_std,
                                                           target_sample, style_layers_default, device)
            requires_grad(model, False)
            style_score = 0
            model(interpolated_sample)
            for sl in style_losses:
                style_score += sl.loss
            style_score *= style_weight

            nli.zero_grad()
            style_score.backward()
            nli_optim.step()

        loss_dict["style_loss"] = style_score.item()
        loss_dict["d"] = d_loss.item()
        loss_dict["g"] = g_loss.item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"style_loss: {loss_dict['style_loss']:.4f}; "
                    f"d_loss: {loss_dict['d']:.4f}; "
                    f"g_loss: {loss_dict['g']:.4f}"
                )
            )

        if i % args.img_freq == 0 and i > 0:
            with torch.set_grad_enabled(False):
                g_ema.eval()
                nli.eval()
                for trg_id, w_target in enumerate(w_trg):
                    w_mixed = nli(torch.cat([w_target.unsqueeze(0), w_aux[trg_id].unsqueeze(0)], 2))
                    interpolated_sample, _ = g_ema([w_mixed], input_is_latent=True, return_feats=False)
                    utils.save_image(
                        interpolated_sample.detach().clamp_(min=-1, max=1),
                        f"%s/generated_{expt_number}_{trg_id}.png" % (imsave_path),
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                w_mixed = nli(torch.cat([w_trg, w_aux], 2))
            return w_mixed

        # if (i % args.save_freq == 0) and (i > 0):
        #     torch.save(
        #         {
        #             "nli": nli.state_dict(),
        #         },
        #         f"%s/NLI_{expt_number}.pt" % (model_path),
        #     )


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=0)
    parser.add_argument("--img_freq", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--adv_bs", type=int, default=8)
    parser.add_argument("--sty_bs", type=int, default=10)
    parser.add_argument("--n_shot", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--ckpt", type=str, default='../checkpoint/550000_backup.pt')
    parser.add_argument("--s_ds", type=str, default='ffhq')
    parser.add_argument("--t_ds", type=str, default='babies')
    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument(
        "--npy_file", type=str, help="path to target domain image files"
    )

    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    n_gpu = 1
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    s_ds = args.s_ds

    t_ds = args.t_ds

    imsave_path = os.path.join('msmt_exp_samples', args.exp)
    model_path = os.path.join('msmt_exp_checkpoints', args.exp)
    w_path = os.path.join('msmt_exp_w', args.exp)

    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(w_path):
        os.makedirs(w_path)

    tensor_imgs = []
    img_arr = np.load(args.npy_file)
    transform = transforms.Compose(
        [
            # transforms.Resize(resize),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    for img in img_arr:
        img = transform(Image.fromarray(img).convert("RGB"))
        tensor_imgs.append(img)

    img_target = torch.stack(tensor_imgs[:args.n_shot], 0).to(device)

    w_target = torch.randn([args.n_shot, 14, 512]).to(device)
    
    for expt_number in range(args.start, args.stop):
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        d = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(device)
        nli = NLI().to(device)

        g_ema.eval()
        # accumulate(g_ema, generator, 0)

        nli_optim = optim.Adam(
            nli.parameters(),
            lr=args.lr,
            betas=(0, 0.99),
        )
        d_optim = optim.Adam(
            d.parameters(),
            lr=args.lr,
            betas=(0, 0.99),
        )

        if args.ckpt is not None:
            print("load model:", args.ckpt)
            ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
            g_ema.load_state_dict(ckpt_source["g_ema"], strict=False)
            d.load_state_dict(ckpt_source["d"], strict=False)

        if args.distributed:
            g_ema = nn.parallel.DataParallel(g_ema)
            nli = nn.parallel.DataParallel(nli)

        w_noise = torch.randn_like(w_target)

        w_target = train(args, nli, nli_optim, g_ema, d, d_optim, img_target, w_target, w_noise, imsave_path,
                         model_path, expt_number, device)
        
        torch.save({'w': w_target}, f'{w_path}/w-{expt_number}.pt')

        del nli, nli_optim, g_ema, d, d_optim, w_noise
