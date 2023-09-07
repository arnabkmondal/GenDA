from glob import glob
from tqdm import tqdm
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

args = parser.parse_args()
args.latent = 512
args.n_mlp = 8

w_path_list = glob("./msmt_exp_w/babies/w-*.pt")

ws = []
for w in tqdm(w_path_list):
    ws.append(torch.load(w)['w'].to('cuda'))

g_ema = Generator(
    args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
).to('cuda')

ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
g_ema.load_state_dict(ckpt_source["g_ema"], strict=False)

imsave_path = f'./gen_imgs'
os.makedirs(imsave_path, exist_ok=True)

for i, w in tqdm(enumerate(ws)):
    fake_img, _ = g_ema([w], input_is_latent=True, return_feats=False)

    for j, img in enumerate(fake_img):
        utils.save_image(
            img.detach().clamp_(min=-1, max=1),
            f"{imsave_path}/generated_{i}_{j}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
