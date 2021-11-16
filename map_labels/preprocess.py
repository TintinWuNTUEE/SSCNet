import argparse
import torch
import yaml
import sys
import os
import numpy as np
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from configs import merge_configs
from sklearn.neighbors import KNeighborsClassifier
from torch_cluster import knn 
import matplotlib.pyplot as plt
device=('cuda')
dataset_config = yaml.safe_load(open(os.path.join('./semantic-kitti.yaml'), 'r'))
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

def fix_label_tensor(data):
    out = data.cpu().numpy()
    out = out&0xffff
    out = np.array(out)+1
    return out
def mask(voxel_label,label_tensor):
    mask1 = torch.zeros_like(voxel_label,dtype=bool)
    mask2 = torch.zeros_like(voxel_label,dtype=bool)
    thing_list = [i for i in dataset_config['thing_class'].keys() if dataset_config['thing_class'][i]==True]
    data = fix_label_tensor(label_tensor)
    for label in thing_list:
        mask1[voxel_label == label] = True
        mask2[data==label] = True
    voxel_label[~mask1] = 0
    return voxel_label,mask1,mask2

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
    indice = np.moveaxis(np.array(np.meshgrid(np.arange(0,256,1),np.arange(0,256,1))),(0,1,2),(2,1,0))
    indice = torch.from_numpy(np.repeat(indice[:,:,np.newaxis,:], 32, axis=2)).to(device)
    remap_lut = get_remap_lut()
    knn = KNeighborsClassifier(n_jobs=4)
    for _,(_,train_label_tensor,train_gt_center,train_gt_offset,_,_,_,_, filenames)  in enumerate(train_dataset_loader):
        train_label_tensor = SemKITTI2train(train_label_tensor)
        train_label_tensor = train_label_tensor.to(device)
        train_gt_center_tensor = train_gt_center.to(device)
        train_gt_offset_tensor = train_gt_offset.to(device)
        for i in range(len(filenames)):
            voxel_label_path = filenames[i].replace('velodyne','voxels').replace('.bin', '.label')
            if(os.path.isfile(voxel_label_path)==False):
                continue
            voxel_label = np.fromfile(voxel_label_path, dtype=np.uint16).reshape(256,256,32)
            voxel_label =torch.from_numpy(remap_lut[voxel_label.astype(np.uint16)]).to(device)
            train_label = train_label_tensor[i]
            voxel_label,mask1,mask2 = mask(voxel_label,train_label)
            
            partial_label = (torch.cat((train_label[:,:,:,np.newaxis],indice),axis=3)[mask2]).reshape(-1,3)
            complete_label = (torch.cat((voxel_label[:,:,:,np.newaxis],indice),axis=3)[mask1]).reshape(-1,3)
            knn.fit(partial_label.cpu().numpy()[:,1:], partial_label.cpu().numpy()[:,0])
            predict = knn.predict(complete_label.cpu().numpy()[:,1:])
            
            voxel_label,mask1 = voxel_label.cpu().numpy(),mask1.cpu().numpy()
            voxel_label[mask1]=predict
            
            ########################## plotting################################
            # bev = (voxel_label>0).sum(axis=2)
            # t4 = plt.figure(5)
            # plt.imshow(bev,cmap=plt.cm.gray,origin='lower')
            # for i in range(0,predict.shape[0],20):
            #     plt.text( complete_label[i,2],complete_label[i,1], str(predict[i]),color="red",fontsize=8)

            # print("done")
            # plt.show()
            ########################## plotting################################
            label_to_be_save= (voxel_label,train_gt_center_tensor[i].cpu().numpy(), train_gt_offset_tensor[i].cpu().numpy())
            folder_path,filename = splitPath(filenames[i])
            folder_path=folder_path.replace('velodyne','preprocess')
            filename+='.pt'
            if not os.path.exists(folder_path):
                print(folder_path)
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path,filename)
            torch.save(label_to_be_save,save_path)
    for _,(_,val_vox_label,val_gt_center,val_gt_offset,val_grid,_,_,_,filenames) in enumerate(val_dataset_loader):
        val_vox_label = SemKITTI2train(val_vox_label)
        val_label_tensor=val_vox_label.to(device)
        val_gt_center_tensor = val_gt_center.to(device)
        val_gt_offset_tensor = val_gt_offset.to(device)
        
        for i in range(len(filenames)):
            voxel_label_path = filenames[i].replace('velodyne','voxels').replace('.bin', '.label')
            if(os.path.isfile(voxel_label_path)==False):
                continue
            voxel_label = np.fromfile(voxel_label_path, dtype=np.uint16).reshape(256,256,32)
            voxel_label =torch.from_numpy(remap_lut[voxel_label.astype(np.uint16)]).to(device)
            
            val_label = val_label_tensor[i]
            voxel_label,mask1,mask2 = mask(voxel_label,val_label)
            
            partial_label = (torch.cat((val_label[:,:,:,np.newaxis],indice),axis=3)[mask2]).reshape(-1,3)
            complete_label = (torch.cat((voxel_label[:,:,:,np.newaxis],indice),axis=3)[mask1]).reshape(-1,3)
            knn.fit(partial_label.cpu().numpy()[:,1:], partial_label.cpu().numpy()[:,0])
            predict = knn.predict(complete_label.cpu().numpy()[:,1:])
            
            voxel_label,mask1 = voxel_label.cpu().numpy(),mask1.cpu().numpy()
            voxel_label[mask1]=predict
            ########################## plotting################################
            # bev = (voxel_label>0).sum(axis=2)
            # t4 = plt.figure(5)
            # plt.imshow(bev,cmap=plt.cm.gray,origin='lower')
            # for i in range(0,predict.shape[0],20):
            #     plt.text( complete_label[i,2],complete_label[i,1], str(predict[i]),color="red",fontsize=8)

            # print("done")
            # plt.show()
            ########################## plotting################################
            label_to_be_save= (voxel_label,val_gt_center_tensor[i].cpu().numpy(), val_gt_offset_tensor[i].cpu().numpy())
            folder_path,filename = splitPath(filenames[i])
            folder_path=folder_path.replace('velodyne','preprocess')
            filename+='.pt'
            if not os.path.exists(folder_path):
                print(folder_path)
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path,filename)
            torch.save(label_to_be_save,save_path)
    
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


