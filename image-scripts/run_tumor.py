#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shiming chen
"""
import os
os.system('''OMP_NUM_THREADS=8  python /home/LAB/chenlb24/compare_model/FREE/train_free.py --gammaD 10 --gammaG 10 \
--gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
--nepoch 501 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot /home/LAB/chenlb24/ZhengDaFuyi --dataset ZDFY \
 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
--nclass_all 3 --nclass_seen 2 --batch_size 32 --nz 312 --latent_size 312 --attSize 768 --resSize 2048  \
--syn_num 700 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.8''')