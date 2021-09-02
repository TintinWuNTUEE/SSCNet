import argparse
import torch
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset


def main(args):
    data_path = args['dataset']['path']
    grid_size = args['dataset']['grid_size']
    batch_size = args['model']['batch_size']


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

    for i_iter,(train_vox_fea,train_label_tensor,train_gt_center,train_gt_offset,train_grid,_,_,train_pt_fea) in enumerate(train_dataset_loader):
        print("train_grid : ", train_grid)
        
        return


