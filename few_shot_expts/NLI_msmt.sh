#!/bin/bash
period=3
for i in $(seq 0 $period 500):
do
j=$(expr $i + $period)
python NLI_train.py --s_ds ffhq --t_ds babies --exp babies --npy_file ./babies_training.npy --n_shot 10 --iter 51 --img_freq 50 --adv_bs 8 --sty_bs 10 --start $i --stop $j
done
python -m pytorch_fid ../../stylegan2-pytorch/real_babies/ msmt_exp_samples/babies/
