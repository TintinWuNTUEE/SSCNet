import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

def fix_label_tensor(data):
    '''fixing uint 8 trick'''
    out = data
    out = out&0xffff
    out = np.array(out)+1
    return out

def get_remap_lut(dataset_config):
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

def create_mask(voxel_label,label_tensor,offset,thing_list):
    '''applying mask'''
    mask1 = np.zeros_like(voxel_label,dtype=bool)
    mask2 = np.zeros_like(label_tensor,dtype=bool)
    mask3 = (np.linalg.norm(offset, axis=0))>0
    data = fix_label_tensor(label_tensor)
    for label in thing_list:
        mask1[voxel_label == label] = True
        mask2[data==label] = True
    return mask1,mask2,mask3


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p','--file_path',default='./semanticKITTI/dataset/sequences')
    parser.add_argument('-s','--sequence',default='04')
    parser.add_argument('-n','--num',type=int,default=0)
    parser.add_argument('-t','--type',nargs='+',help='preprocess voxel_label instance',default=['preprocess','voxel_label', 'instance'])
    
    args = parser.parse_args()
    
    preprocess_file = sorted(glob(os.path.join(args.file_path, args.sequence, 'preprocess', "*.pt")))
    label_file = sorted(glob(os.path.join(args.file_path, args.sequence, "voxels", "*.label")))
    point_file = sorted(glob(os.path.join(args.file_path, args.sequence, "velodyne","*.bin")))[::5]
    point_label_file = sorted(glob(os.path.join(args.file_path, args.sequence, "labels","*.label")))[::5]
    instance_file = sorted(glob(os.path.join(args.file_path, args.sequence, 'instance', "*.pt")))
    num_with_instance = (np.array([int(i[-11:-5]) for i in instance_file])/5).astype(np.int16)
    print(num_with_instance)
    print(args)
    # print(preprocess_file)
    
    dataset_config = yaml.safe_load(open(os.path.join('./data/semantic-kitti.yaml'), 'r'))
    thing_list = [i for i in dataset_config['thing_class'].keys() if dataset_config['thing_class'][i]==True]
    
    if 'preprocess' in args.type:
        print('into preprocess')
        preprocess_data = list(torch.load(preprocess_file[args.num]))
        if len(preprocess_data) == 3:
            partial_label = preprocess_data[0]
            center = preprocess_data[1]
            offset = preprocess_data[2]
            inst_data = (partial_label&0xffff0000)>>16
            data = fix_label_tensor(partial_label)
            mask1 = np.zeros_like(partial_label,dtype=bool)
            for label in thing_list:
                mask1[data == label] = True
            partial_label[~mask1] = 0
            partial_voxel_bev = ((partial_label>0).sum(axis=2))>0
            
            plot0 = plt.figure('partial_voxel_bev')
            plt.imshow(partial_voxel_bev,cmap=plt.cm.gray,origin='lower')
            partial_voxel_bev_nonzero = np.nonzero(partial_voxel_bev)
            # print(partial_voxel_bev_nonzero)
            for row, col in zip(partial_voxel_bev_nonzero[0],partial_voxel_bev_nonzero[1]):
                for i in range(32):
                    if partial_label[row, col, i] != 0:
                        # plt.text(col, row, str(data[row, col, i]), color='green',fontsize=12)
                        plt.text(col, row, str(inst_data[row, col, i]), color='red',fontsize=12)
                        break
            plot1 = plt.figure('center heat map')
            plt.imshow(center.squeeze(), cmap='hot', origin='lower')
            
            plot2, ax = plt.subplots()
            a = ax.quiver(offset[1,:,:],offset[0,:,:],angles='xy',scale_units='xy',scale=1)
        elif len(preprocess_data) == 2:
            center = preprocess_data[0]
            offset = preprocess_data[1]
            plot1 = plt.figure('center heat map')
            plt.imshow(center.squeeze(), cmap='hot', origin='lower')
            
            plot2, ax = plt.subplots()
            a = ax.quiver(offset[1,:,:],offset[0,:,:],angles='xy',scale_units='xy',scale=1)
        else:
            raise ValueError("preprocess data should be length 2 or length 3")

        
    if 'voxel_label' in args.type:
        print('into voxel_label')
        voxel_label = np.fromfile(label_file[args.num], dtype=np.uint16).reshape(256,256,32)
        remap_lut = get_remap_lut(dataset_config)
        transformed_voxel_label = remap_lut[voxel_label.astype(np.uint16)]
        mask2 = np.zeros_like(voxel_label,dtype=bool)
        
        for label in thing_list:
            mask2[transformed_voxel_label == label] = True
        voxel_label[~mask2] = 0
        complete_voxel_bev = ((voxel_label>0).sum(axis=2))
        
        plot3 = plt.figure('complete_voxel_bev')
        plt.imshow(complete_voxel_bev,cmap=plt.cm.gray,origin='lower')
        complete_voxel_bev_nonzero = np.nonzero(complete_voxel_bev)
        # for row, col in zip(complete_voxel_bev_nonzero[0],complete_voxel_bev_nonzero[1]):
        #     for i in range(32):
        #         if voxel_label[row,col,i] != 0:
        #             plt.text(col,row,str(voxel_label[row,col,i]),color='green',fontsize=12)
        #             break
        
    if 'instance' in args.type:
        pad_size = [60,60,20]
        center = [30,30,10]
        if args.num in num_with_instance:
            instance_num = np.where(num_with_instance==args.num)[0]
            print(instance_num)
            partial_instances = []
            complete_instances = []
            x_max, y_max, z_max = [], [], []
            for i in instance_num:
                partial_instance, partial_instance_label, complete_instance, complete_instance_label = torch.load(instance_file[i])
                # print(partial_instance.min(axis=0))"
                # print(partial_instance)
                partial_sem = partial_instance_label&0xffff
                partial_inst = (partial_instance_label&0xffff0000)>>16
                complete_sem = complete_instance_label&0xffff
                complete_inst = (complete_instance_label&0xffff0000)>>16

                partial_instance -= ((partial_instance.max(axis=0)+partial_instance.min(axis=0))/2).astype(np.int16)
                partial_instance += center
                complete_instance -= ((complete_instance.max(axis=0)+complete_instance.min(axis=0))/2).astype(np.int16)
                complete_instance += center
                pad1 = np.zeros(pad_size)
                pad2 = np.zeros(pad_size)
                pad1[partial_instance[:,0],partial_instance[:,1],partial_instance[:,2]] = 1
                pad2[complete_instance[:,0],complete_instance[:,1],complete_instance[:,2]] = 1
                partial_instances.append(partial_instance)
                complete_instances.append(complete_instance)     
                fig = plt.figure()
                voxel1 = fig.add_subplot(121,projection='3d')
                voxel1.voxels(pad1)
                voxel1.title.set_text(" partial_sem : "+str(partial_sem)+" inst : "+str(partial_inst))
                voxel2 = fig.add_subplot(122,projection='3d')
                voxel2.voxels(pad2)           
                voxel2.title.set_text(" complete_sem : "+str(complete_sem)+" inst : "+str(complete_inst))
           

        else:
            print("this scene doesn't have static instance")
        pass
    print('done')
    plt.show()
    
    return
    
    
    
    


if __name__ == '__main__':
    main()