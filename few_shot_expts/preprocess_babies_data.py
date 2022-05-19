from PIL import Image
from glob import glob
from natsort import natsorted
import numpy as np


img_list = natsorted(glob("<path_to_10shot_babies_images>\\*.png"))

imgs = []
for im_path in img_list:
    im = Image.open(im_path).convert('RGB').resize((256, 256))
    imgs.append(np.expand_dims(np.array(im), 0))

imgs = np.concatenate(imgs)
np.save(f'./babies_training.npy', imgs)
print(imgs.shape, imgs.dtype)