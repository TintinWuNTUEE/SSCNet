import argparse
import torch
import yaml
import sys
import os
import numpy as np
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from utils.configs import merge_configs

device=('cuda')

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def get_remap_lut():
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''
    dataset_config = yaml.safe_load(open(os.path.join('./semantic-kitti.yaml'), 'r'))
    # make lookup table for mapping
    maxkey = max(dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(dataset_config['learning_map'].keys())] = list(dataset_config['learning_map'].values())

    # in completion we have to distinguish empty and invalid voxels.
    # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    return remap_lut

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick
def splitPath(path):
    folderpath = os.path.split(path)[0]
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]
    return folderpath, filename
def main(args):
    data_path = args['dataset']['path']
    grid_size = args['dataset']['grid_size']
    batch_size = args['model']['train_batch_size']
    val_batch_size = args['model']['val_batch_size']

    #prepare dataset
    train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    print("train_pt size = "+str(len(train_pt_dataset)))
    print("val_pt size = "+str(len(val_pt_dataset)))
    train_dataset=voxel_dataset(train_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0,use_aug = True,max_volume_space = [51.2,25.6,4.4], min_volume_space = [0,-25.6,-2])
    val_dataset=voxel_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0,max_volume_space = [51.2,25.6,4.4], min_volume_space = [0,-25.6,-2])
    print("train size = "+str(len(train_dataset)))
    print("val size = "+str(len(val_dataset)))
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    for _,(val_vox_fea,val_vox_label,val_gt_center,val_gt_offset,val_grid,val_pt_labels,val_pt_ints,val_pt_fea,filenames) in enumerate(val_dataset_loader):
        val_vox_label = SemKITTI2train(val_vox_label)
        val_label_tensor=val_vox_label.to(device)
        val_gt_center_tensor = val_gt_center.to(device)
        val_gt_offset_tensor = val_gt_offset.to(device)
        
        for i in range(len(filenames)):
            label_to_be_save= (val_label_tensor[i].cpu().numpy(),val_gt_center_tensor[i].cpu().numpy(), val_gt_offset_tensor[i].cpu().numpy())
            folder_path,filename = splitPath(filenames[i])
            folder_path=folder_path.replace('velodyne','preprocess')
            filename+='.pt'
            if not os.path.exists(folder_path):
                print(folder_path)
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path,filename)
            torch.save(label_to_be_save,save_path)
    for _,(_,train_label_tensor,train_gt_center,train_gt_offset,_,_,_,_, filenames)  in enumerate(train_dataset_loader):
        train_label_tensor = SemKITTI2train(train_label_tensor)
        train_label_tensor = train_label_tensor.to(device)
        train_gt_center_tensor = train_gt_center.to(device)
        train_gt_offset_tensor = train_gt_offset.to(device)
        for i in range(len(filenames)):
            label_to_be_save=(train_label_tensor[i].cpu().numpy(),train_gt_center_tensor[i].cpu().numpy(),train_gt_offset_tensor[i].cpu().numpy())
            folder_path,filename = splitPath(filenames[i])
            folder_path=folder_path.replace('velodyne','preprocess')
            filename+='.pt'
            if not os.path.exists(folder_path):
                print(folder_path)
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path,filename)
            torch.save(label_to_be_save,save_path)
            # use labels = np.load("file name", allow_pickle=True) to get data back, the labels is (3,), 
            # and label[0]'s size = (256,256,32)->sem label, label[1]'s size = (1,256,256)->center label, label[2]'s size = (2,256,256)->offset label
    
    return

        
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='../semanticKITTI/dataset')
    parser.add_argument('-p', '--model_save_path', default='./Panoptic_SemKITTI.pt')
    parser.add_argument('-c', '--configs', default='configs/SemanticKITTI_model/Panoptic-PolarNet.yaml')
    parser.add_argument('--pretrained_model', default='empty')

    args = parser.parse_args()
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args,new_args)

    print(' '.join(sys.argv))
    print(args)
    main(args)


