import torch
import numpy as np
import os

data_path = 'C:/Users/Barbod/Desktop/uni/research/tum/Pointnet_Pointnet2_pytorch/data' \
            '/shapenetcore_partanno_segmentation_benchmark_v0_normal'
checkpoint_path = './trained_models/'

torch.manual_seed(0)
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

npoint = 2048
normal = True
batch_size = 16
num_part = 50
num_classes = 16
LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = 20
learning_rate = 0.001
decay_rate = 1e-4
final_epoch = 251
lr_decay = 0.5
step_size = 20

