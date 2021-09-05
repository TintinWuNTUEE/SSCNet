import argparse
import torch
import yaml
import sys
import numpy as np
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from utils.configs import merge_configs


def main(args):
    data_path = args['dataset']['path']
    grid_size = args['dataset']['grid_size']
    batch_size = args['model']['train_batch_size']


    #prepare dataset
    train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    print("train_pt size = "+str(len(train_pt_dataset)))
    print("val_pt size = "+str(len(val_pt_dataset)))
    if args['model']['polar']:
        train_dataset=spherical_dataset(train_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0, use_aug = True)
        val_dataset=spherical_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0)
    else:
        train_dataset=voxel_dataset(train_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0,use_aug = True)
        val_dataset=voxel_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0)
    print("train size = "+str(len(train_dataset)))
    print("val size = "+str(len(val_dataset)))
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    print(len(train_dataset_loader))

    for i_iter,data in enumerate(train_dataset_loader):
        (train_vox_fea,train_label_tensor,train_gt_center,train_gt_offset,train_grid,_,_,train_pt_fea, filenames) = data

        print("train_vox_fea : ", train_vox_fea.shape)
        print(train_vox_fea)
        
        print("train_label_tensor : ", train_label_tensor.shape)
        print(train_label_tensor)
        
        print("train_gt_center : ", train_gt_center.shape)
        print(train_gt_center)

        print("train_gt_offset : ", train_gt_offset.shape)
        print(train_gt_offset)

        print("train_grid : ", len(train_grid[0]), len(train_grid[1]))
        print(np.array(train_grid[0]).shape)
        print(train_grid)

        print("train_pt_fea : ", len(train_pt_fea[0]), len(train_pt_fea[1]))
        print(np.array(train_pt_fea[0]).shape)
        print(train_pt_fea)        

        print(filenames)

        for i in range(len(filenames)):
            label_to_be_save = np.empty(3,object)
            label_to_be_save[:] = [train_label_tensor[i], train_gt_center[i], train_gt_offset[i]]

            np.save(filenames[i],label_to_be_save)

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


