import argparse
import torch
import yaml
import sys
import os
import numpy as np
from common.io_tools import dict_to
from dataloader.dataset import get_preprocess_dataset
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
def splitPath(path):
    folderpath = os.path.split(path)[0]
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]
    return folderpath, filename
def main(args,dataset):
    dset = dataset['train']
    for _,(pos_in,center_in,offset_in,_,_,_,_,filenames,gt_sem,gt_center,gt_offset)  in enumerate(dset):
        pos_in,center_in,offset_in= pos_in.to(device),center_in.to(device),offset_in.to(device)
        gt_sem,gt_center,gt_offset =gt_sem.to(device),gt_center.to(device),gt_offset.to(device)
        instance_input,input_class_list = get_instance(args,pos_in,center_in,offset_in,dset)
        instance_label,label_class_list = get_instance(args,gt_sem,gt_center,gt_offset,dset)
        input_class_nbr = len(input_class_list)
        label_class_nbr = len(label_class_list)
        class_nbr = np.min((input_class_nbr,label_class_nbr))
        for i in range(class_nbr):
            data_tuple = (instance_input[i],input_class_list[i],instance_label[i],label_class_list[i])
            folder_path,filename = splitPath(filenames[0])
            folder_path=folder_path.replace('velodyne','instance')
            filename=filename+'_'+str(i)+'.pt'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path,filename)
            torch.save(data_tuple,save_path)
    dset = dataset['val']
    for _,(pos_in,center_in,offset_in,_,_,_,_,filenames,gt_sem,gt_center,gt_offset)  in enumerate(dset):
        pos_in,center_in,offset_in= pos_in.to(device),center_in.to(device),offset_in.to(device)
        gt_sem,gt_center,gt_offset =gt_sem.to(device),gt_center.to(device),gt_offset.to(device)
        instance_input,input_class_list = get_instance(args,pos_in,center_in,offset_in,dset)
        instance_label,label_class_list = get_instance(args,gt_sem,gt_center,gt_offset,dset)
        input_class_nbr = len(input_class_list)
        label_class_nbr = len(label_class_list)
        class_nbr = np.min((input_class_nbr,label_class_nbr))
        
        
        for i in range(class_nbr):
            data_tuple = (instance_input[i],input_class_list[i],instance_label[i],label_class_list[i])
            folder_path,filename = splitPath(filenames[0])
            folder_path=folder_path.replace('velodyne','instance')
            filename=filename+'_'+str(i)+'.pt'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path,filename)
            torch.save(data_tuple,save_path)
if __name__ == '__main__':
    args = parse_args()
    dataset=get_preprocess_dataset(args)
    main(args,dataset)
