#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import numba as nb
import yaml
import pickle
import errno
from torch.utils.data import DataLoader,Dataset
from glob import glob
import random

# from .io_data import io_data 
from .process_panoptic import PanopticLabelGenerator
from .instance_augmentation import instance_augmentation
def get_dataset(_cfg):
    grid_size = _cfg['dataset']['grid_size']
    data_path = _cfg['dataset']['path']
    train_batch_size = _cfg['model']['train_batch_size']
    val_batch_size = _cfg['model']['val_batch_size']
    num_workers = 4
    dataset={}
    train_instance_dataset = Instance_Dataset(_cfg['dataset'],phase='train')
    val_instance_dataset = Instance_Dataset(_cfg['dataset'],phase='val')
    dataset['train'] = DataLoader(train_instance_dataset,batch_size=train_batch_size,num_workers=num_workers,shuffle=True,pin_memory=True)
    dataset['val'] = DataLoader(val_instance_dataset,batch_size=val_batch_size, num_workers=num_workers, shuffle=False,pin_memory=True)
    return dataset
def get_preprocess_dataset(_cfg):
    grid_size = _cfg['dataset']['grid_size']
    data_path = _cfg['dataset']['path']
    train_batch_size = 1
    val_batch_size = 1
    
    dataset={}
    train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True, instance_pkl_path=_cfg['dataset']['instance_pkl_path'])
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, instance_pkl_path=_cfg['dataset']['instance_pkl_path'])
    train_dataset=voxel_dataset(train_pt_dataset, _cfg['dataset'], grid_size = grid_size, ignore_label = 0,use_aug = True,max_volume_space = [51.2,25.6,4.4], min_volume_space = [0,-25.6,-2])
    val_dataset=voxel_dataset(val_pt_dataset, _cfg['dataset'], grid_size = grid_size, ignore_label = 0,max_volume_space = [51.2,25.6,4.4], min_volume_space = [0,-25.6,-2],phase='val')
    dataset['train']= DataLoader(dataset = train_dataset,
                                                    batch_size = train_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)
    dataset['val'] = DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)
    return dataset

class Instance_Dataset(Dataset):
    def __init__(self,args,type='points',phase='train'):
        self.phase = phase
        self.filepaths=[]
        self.type = type
        self.root_dir = args['path']
        self.split = {'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], 'val': [8], 'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}
        self.get_filepaths()
        self.nbr_files = len(self.filepaths)
        
    def get_filepaths(self):
        # print(glob(os.path.join(self.root_dir, 'sequences', '*')))
        sequences = list(sorted(glob(os.path.join(self.root_dir, 'sequences', '*')))[i] for i in self.split[self.phase])
        for sequence in sequences:
            assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
            self.filepaths+= sorted(glob(os.path.join(sequence, 'instance', '*.pt')))

    def get_data(self, index):
        DATA = torch.load(self.filepaths[index])
        instance_input,input_class,instance_label,label_class = DATA
        return instance_input,input_class,instance_label,label_class

    def sample(self,instances,sample_num=4096):
        '''
        Sample the instance either with voxel padding or points
        '''
        # print(instances.nonzero())
        
        
        if self.type =="points":
            instance_num = len(instances)           
            instance_grid = np.zeros((sample_num,3))
            instance_grid += np.array([256,256,32])
            if instance_num > sample_num:
                instance_num = sample_num
            for i in range (instance_num):
                instance_grid[i] = instances[i]
            return instance_grid
        elif self.type =="voxel":
            # pad your instance here
            return instances
        
        return instances
    def __getitem__(self, index):
        #get data
        instance_input,input_class,instance_label,label_class = self.get_data(index)
        # sample
        instance_input = self.sample(instance_input)
        instance_label = self.sample(instance_label,sample_num=16384)
        
        return instance_input,input_class,instance_label,label_class
    def __len__(self):
        """
        Return the length of the dataset
        """
        return self.nbr_files
class SemKITTI(Dataset):
    def __init__(self, data_path, imageset = 'train', return_ref = False, instance_pkl_path ='data'):
        self.return_ref = return_ref
        with open("configs/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        thing_class = semkittiyaml['thing_class']
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')
        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
        self.im_idx.sort()
        # get class distribution weight 
        epsilon_w = 0.001
        origin_class = semkittiyaml['content'].keys()
        weights = np.zeros((len(semkittiyaml['learning_map_inv'])-1,),dtype = np.float32)
        for class_num in origin_class:
            if semkittiyaml['learning_map'][class_num] != 0:
                weights[semkittiyaml['learning_map'][class_num]-1] += semkittiyaml['content'][class_num]
        self.CLS_LOSS_WEIGHT = 1/(weights + epsilon_w)
        self.instance_pkl_path = instance_pkl_path
        
        
         
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)
    
    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            sem_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
            inst_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=np.uint32),axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne','labels')[:-3]+'label', dtype=np.uint32).reshape((-1,1))
            sem_data = annotated_data & 0xFFFF #delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data
        data_tuple = (raw_data[:,:3], sem_data.astype(np.uint8),inst_data)
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        #file name added
        data_tuple += (self.im_idx[index],)

        return data_tuple

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        filenames = sorted(filenames)
        filenames = filenames[::5]
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

class voxel_dataset(Dataset):
    def __init__(self, in_dataset, args, grid_size, ignore_label = 0, return_test = False, fixed_volume_space= True, use_aug = False, max_volume_space = [50,50,1.5], min_volume_space = [-50,-50,-3],phase='train'):
        'Initialization'
        with open("configs/semantic-kitti.yaml",'r') as stream:
            self.dataset_config = yaml.safe_load(stream)
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = args['rotate_aug'] if use_aug else False
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = args['flip_aug'] if use_aug else False
        self.instance_aug = args['inst_aug'] if use_aug else False
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        
        self.root_dir =args['path']
        self.panoptic_proc = PanopticLabelGenerator(self.grid_size,sigma=args['gt_generator']['sigma'])
        if self.instance_aug:
            self.inst_aug = instance_augmentation(self.point_cloud_dataset.instance_pkl_path+'/instance_path.pkl',self.point_cloud_dataset.thing_list,self.point_cloud_dataset.CLS_LOSS_WEIGHT,\
                                                random_flip=args['inst_aug_type']['inst_global_aug'],random_add=args['inst_aug_type']['inst_os'],\
                                                random_rotate=args['inst_aug_type']['inst_global_aug'],local_transformation=args['inst_aug_type']['inst_loc_aug'])
        
        self.split = {'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], 'val': [8], 'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}
        self.phase = phase
        self.get_preprocess_filepaths()
        thing_class = self.dataset_config['thing_class']
        self.thing_list = [class_nbr for class_nbr, is_thing in thing_class.items() if is_thing]
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        # if len(data) == 3:
        if len(data) == 4:
            xyz,labels,insts, filename = data
        # elif len(data) == 4:
        elif len(data) == 5:
            xyz,labels,insts,feat, filename = data
            if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        else: raise Exception('Return invalid data tuple')
        if len(labels.shape) == 1: labels = labels[..., np.newaxis]
        if len(insts.shape) == 1: insts = insts[..., np.newaxis]
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        # random instance augmentation
        if self.instance_aug:
            xyz,labels,insts,feat = self.inst_aug.instance_aug(xyz,labels.squeeze(),insts.squeeze(),feat)

        max_bound = np.percentile(xyz,100,axis = 0)
        min_bound = np.percentile(xyz,0,axis = 0)
        
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        
        intervals = crop_range/(cur_grid_size-1)
        if (intervals==0).any(): print("Zero interval!")
        
        grid_ind = (np.floor((np.clip(xyz,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = (np.indices(self.grid_size) + 0.5)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)

        # get thing points mask
        mask = np.zeros_like(labels,dtype=bool)
        for label in self.point_cloud_dataset.thing_list:
            mask[labels == label] = True
        
        inst_label = insts[mask].squeeze()
        unique_label = np.unique(inst_label)
        unique_label_dict = {label:idx+1 for idx , label in enumerate(unique_label)}
        if inst_label.size > 1:            
            inst_label = np.vectorize(unique_label_dict.__getitem__)(inst_label)
            
            # process panoptic
            processed_inst = np.ones(self.grid_size[:2],dtype = np.uint8)*self.ignore_label
            inst_voxel_pair = np.concatenate([grid_ind[mask[:,0],:2],inst_label[..., np.newaxis]],axis = 1)
            inst_voxel_pair = inst_voxel_pair[np.lexsort((grid_ind[mask[:,0],0],grid_ind[mask[:,0],1])),:]
            processed_inst = nb_process_inst(np.copy(processed_inst),inst_voxel_pair)
        else:
            processed_inst = None       
        center,center_points,offset = self.panoptic_proc(insts[mask],xyz[mask[:,0]],processed_inst,voxel_position[:2,:,:,0],unique_label_dict,min_bound,intervals)
        
        if inst_label.size > 1: 
            processed_inst = processed_inst[:,:,np.newaxis].astype(np.int32).repeat(32,axis=2)
            processed_inst = processed_inst << 16
        else:
            processed_inst = 0
        processed_label = processed_label.astype(np.int32)
        processed_label += processed_inst
        
        preprocess_label,preprocess_center,preprocess_offset = self.get_preprocess_data(index)
        # print(preprocess_center.shape)
        data_tuple = (processed_label,center,offset)
         # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1)
        
        # if len(data) == 3:
        if len(data) == 4:
            return_fea = return_xyz
        # elif len(data) == 4:
        elif len(data) == 5:
            return_fea = np.concatenate((return_xyz,feat),axis = 1)
                    
        if self.return_test:
            data_tuple += (grid_ind,labels,insts,return_fea,index,filename,preprocess_label,preprocess_center,preprocess_offset)
        else:
            data_tuple += (grid_ind,labels,insts,return_fea,filename,preprocess_label,preprocess_center,preprocess_offset)

        return data_tuple
    
    def get_preprocess_filepaths(self):
        # print(glob(os.path.join(self.root_dir, 'sequences', '*')))
        sequences = list(sorted(glob(os.path.join(self.root_dir, 'sequences', '*')))[i] for i in self.split[self.phase])
        self.filepaths = []
        for sequence in sequences:
            assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
            self.filepaths+= sorted(glob(os.path.join(sequence, 'preprocess', '*.pt')))
        return
    def get_preprocess_data(self, index):
        PREPROCESS = torch.load(self.filepaths[index])
        sem_label, center_label, offset_label = PREPROCESS
        return sem_label, center_label, offset_label
# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

@nb.jit('u1[:,:](u1[:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_inst(processed_inst,sorted_inst_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_inst_voxel_pair[0,2]] = 1
    cur_sear_ind = sorted_inst_voxel_pair[0,:2]
    for i in range(1,sorted_inst_voxel_pair.shape[0]):
        cur_ind = sorted_inst_voxel_pair[i,:2]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_inst[cur_sear_ind[0],cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_inst_voxel_pair[i,2]] += 1
    processed_inst[cur_sear_ind[0],cur_sear_ind[1]] = np.argmax(counter)
    return processed_inst

def collate_fn_BEV(data):
    label2stack=np.stack([d[0] for d in data])
    center2stack=np.stack([d[1] for d in data])
    offset2stack=np.stack([d[2] for d in data])
    grid_ind_stack = [d[3] for d in data]
    point_label = [d[4] for d in data]
    point_inst = [d[5] for d in data]
    xyz = [d[6] for d in data]
    filename = [d[7] for d in data]
    gt_label2stack=np.stack([d[8] for d in data])
    gt_center2stack=np.stack([d[9] for d in data])
    gt_offset2stack=np.stack([d[10] for d in data])
    return torch.from_numpy(label2stack),torch.from_numpy(center2stack),torch.from_numpy(offset2stack),grid_ind_stack,point_label,point_inst,xyz,filename,torch.from_numpy(gt_label2stack),torch.from_numpy(gt_center2stack),torch.from_numpy(gt_offset2stack)

def collate_fn_BEV_test(data):    
    label2stack=np.stack([d[0] for d in data])
    center2stack=np.stack([d[1] for d in data])
    offset2stack=np.stack([d[2] for d in data])
    grid_ind_stack = [d[3] for d in data]
    point_label = [d[4] for d in data]
    point_inst = [d[5] for d in data]
    xyz = [d[6] for d in data]
    index = [d[7] for d in data]
    return torch.from_numpy(label2stack),torch.from_numpy(center2stack),torch.from_numpy(offset2stack),grid_ind_stack,point_label,point_inst,xyz,index

# load Semantic KITTI class info
# with open("semantic-kitti.yaml", 'r') as stream:
#     semkittiyaml = yaml.safe_load(stream)
# SemKITTI_label_name = dict()
# for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
#     SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]


if __name__ == '__main__':
    device = torch.device('cuda')
    with open('Panoptic-PolarNet.yaml','r') as stream:
        config = yaml.safe_load(stream)
    dset = get_dataset(config)
    for _,(instance_input,input_class_list,instance_label,label_class_list) in enumerate(dset['train']):
        continue