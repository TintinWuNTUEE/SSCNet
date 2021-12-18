import argparse
import torch
import yaml
import sys
import os
import numpy as np
from dataloader.dataset import collate_fn_BEV,SemKITTI,voxel_dataset
from common.configs import merge_configs
device=('cuda')
############################## grid size setting ##############################
max_bound = np.asarray([51.2,25.6,4.4])
min_bound = np.asarray([0,-25.6,-2])
crop_range = max_bound-min_bound
cur_grid_size = np.asarray([256,256,32])
intervals = crop_range/(cur_grid_size-1)
############################## grid size setting ##############################
def train(model,args,scheduler):
    data_path = args['dataset']['path']
    grid_size = args['dataset']['grid_size']
    batch_size = args['model']['train_batch_size']
    val_batch_size = args['model']['val_batch_size']