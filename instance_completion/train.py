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
from common.utils import sample
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
def train(model,loss_fn,scheduler,optimizer,dataset,args,start_epoch=0):
    dset = dataset['train']
    nbr_epochs = args['model']['max_epoch']
    sample_type = "points"
    dtype = torch.float32
    # model.train()
    for epoch in range(start_epoch,nbr_epochs+1):
        for _,(pos_in,center_in,offset_in,_,_,_,_,filenames,gt_sem,gt_center,gt_offset)  in enumerate(dset):
            pos_in,center_in,offset_in= pos_in.to(device),center_in.to(device),offset_in.to(device)
            # print(val_in.dtype)
            # print(gt_sem.shape)
            gt_sem,gt_center,gt_offset =gt_sem.to(device),gt_center.to(device),gt_offset.to(device)
            instance_input,input_class_list = get_instance(args,pos_in,center_in,offset_in,dset)
            instance_label,label_class_list = get_instance(args,gt_sem,gt_center,gt_offset,dset)
            input_class_nbr = len(input_class_list)
            label_class_nbr = len(label_class_list)
            class_nbr = np.min((input_class_nbr,label_class_nbr))
            
            instance_input = sample(instance_input,input_class_list,type=sample_type)
            
            pred_list = []
            for i in range(class_nbr):
                pred = model(instance_input[i])
                pred_list.append(pred)
            pred_list = torch.cat(pred_list,dim=0)
            loss = loss_fn(pred_list,instance_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
if __name__ == '__main__':
    args = parse_args()
    dataset=get_dataset(args)
    model = None
    loss_fn = None
    scheduler = None
    optimizer = None
    train(model,loss_fn,scheduler,optimizer,dataset,args)