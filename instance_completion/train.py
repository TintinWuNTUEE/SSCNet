import argparse
import torch
import yaml
import sys
import os
import numpy as np
from common.io_tools import dict_to
from dataloader.dataset import get_dataset
from common.configs import merge_configs
from common.utils import get_instance
from common.utils import get_unique_label
from common.logger import get_logger
device=torch.device('cuda')
############################## grid size setting ##############################
max_bound = np.asarray([51.2,25.6,4.4])
min_bound = np.asarray([0,-25.6,-2])
crop_range = max_bound-min_bound
cur_grid_size = np.asarray([256,256,32])
intervals = crop_range/(cur_grid_size-1)
############################## grid size setting ##############################
def get_mem_allocated(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-d', '--data_dir', help='path to dataset root folder', default='../semanticKITTI/dataset')
    parser.add_argument('-p', '--model_save_path', default='./weights')
    parser.add_argument('-c', '--configs', help='path to config file', default='configs/Panoptic-PolarNet.yaml')
    parser.add_argument('--pretrained_model', default='empty')
    args = parser.parse_args()
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args,new_args)
  
    return args
def train(dataset,args,start_epoch=0):
    dset = dataset['train']
    nbr_epochs = args['model']['max_epoch']
    dtype = torch.float32
    for epoch in range(start_epoch,nbr_epochs+1):
        for _,(val_in,val_center,val_offset,_,_,_,_,filenames,gt_sem,gt_center,gt_offset)  in enumerate(dset):
            val_in,val_center,val_offset= val_in.to(device),val_center.to(device),val_offset.to(device)
            # print(val_in.dtype)
            # print(gt_sem.shape)
            gt_sem,gt_center,gt_offset =gt_sem.to(device),gt_center.to(device),gt_offset.to(device)
            instance_input,_ = get_instance(args,val_in,val_center,val_offset,dset)
            instance_label,_ = get_instance(args,gt_sem,gt_center,gt_offset,dset)
            
    return
if __name__ == '__main__':
    args = parse_args()
    dataset=get_dataset(args)
    
    train(dataset,args)