import argparse
import torch
import yaml
import sys
import os
import numpy as np
from dataloader.dataset import collate_fn_BEV,SemKITTI,voxel_dataset
from configs import merge_configs
from sklearn.neighbors import KNeighborsClassifier
device=('cuda')
dataset_config = yaml.safe_load(open(os.path.join('./semantic-kitti.yaml'), 'r'))
############################## grid size setting ##############################
max_bound = np.asarray([51.2,25.6,4.4])
min_bound = np.asarray([0,-25.6,-2])
crop_range = max_bound-min_bound
cur_grid_size = np.asarray([256,256,32])
intervals = crop_range/(cur_grid_size-1)
############################## grid size setting ##############################

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
###############################################################################
def fix_label_tensor(data):
    '''fixing uint 8 trick'''
    out = data.cpu().numpy()
    out = out&0xffff
    out = np.array(out)+1
    return out
def mask(voxel_label,label_tensor):
    '''applying mask'''
    mask1 = torch.zeros_like(voxel_label,dtype=bool)
    mask2 = torch.zeros_like(voxel_label,dtype=bool)
    thing_list = [i for i in dataset_config['thing_class'].keys() if dataset_config['thing_class'][i]==True]
    data = fix_label_tensor(label_tensor)
    for label in thing_list:
        mask1[voxel_label == label] = True
        mask2[data==label] = True
    voxel_label[~mask1] = 0
    return voxel_label,mask1,mask2
###############################################################################
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
    
    ###################### I change the semantic-kitti.yaml to make val dataloader contain sequence 0-10
    
    # train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    # print("train_pt size = "+str(len(train_pt_dataset)))
    print("val_pt size = "+str(len(val_pt_dataset)))
    # train_dataset=voxel_dataset(train_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0,use_aug = True,max_volume_space = [51.2,25.6,4.4], min_volume_space = [0,-25.6,-2])
    val_dataset=voxel_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0,max_volume_space = [51.2,25.6,4.4], min_volume_space = [0,-25.6,-2])
    # print("train size = "+str(len(train_dataset)))
    print("val size = "+str(len(val_dataset)))
    # train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    #                                                 batch_size = batch_size,
    #                                                 collate_fn = collate_fn_BEV,
    #                                                 shuffle = False,
    #                                                 num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)
    indice = np.moveaxis(np.array(np.meshgrid(np.arange(0,256,1),np.arange(0,256,1))),(0,1,2),(2,1,0))
    indice = torch.from_numpy(np.repeat(indice[:,:,np.newaxis,:], 32, axis=2)).int().to(device)
    remap_lut = get_remap_lut()
    knn_5 = KNeighborsClassifier(n_jobs=4,weights='distance')
    knn_1 = KNeighborsClassifier(n_neighbors=1,n_jobs=4,weights='distance')
    PanopticLabelGenerator = PanopticLabelGenerator_VoxelVersion((256,256,32))
    for _,(_,val_vox_label,val_gt_center,val_gt_offset,val_grid,_,_,_,filenames) in enumerate(val_dataset_loader):
        val_vox_label = SemKITTI2train(val_vox_label)
        val_label_tensor=val_vox_label.to(device)
        # val_gt_center_tensor = val_gt_center.to(device)
        # val_gt_offset_tensor = val_gt_offset.to(device)
        
        for i in range(len(filenames)):
            voxel_label_path = filenames[i].replace('velodyne','voxels').replace('.bin', '.label')
            if(os.path.isfile(voxel_label_path)==False):
                continue
            voxel_label = np.fromfile(voxel_label_path, dtype=np.uint16).reshape(256,256,32)
            voxel_label =torch.from_numpy(remap_lut[voxel_label.astype(np.uint16)]).to(device)
            
            val_label = val_label_tensor[i]
            voxel_label,mask1,mask2 = mask(voxel_label,val_label)
            if (~mask1).all() or (~mask2).all():
                voxel_label,mask1 = voxel_label.cpu().numpy(),mask1.cpu().numpy()
                pass
            else:
                partial_label = (torch.cat((val_label[:,:,:,np.newaxis],indice),axis=3)[mask2]).reshape(-1,3)
                complete_label = (torch.cat((voxel_label[:,:,:,np.newaxis],indice),axis=3)[mask1]).reshape(-1,3)
                if complete_label.shape[0] >= 5:
                    knn_5.fit(partial_label.cpu().numpy()[:,1:], partial_label.cpu().numpy()[:,0])
                    predict = knn_5.predict(complete_label.cpu().numpy()[:,1:])
                else:
                    knn_1.fit(partial_label.cpu().numpy()[:,1:], partial_label.cpu().numpy()[:,0])
                    predict = knn_1.predict(complete_label.cpu().numpy()[:,1:])

                voxel_label,mask1 = voxel_label.cpu().numpy(),mask1.cpu().numpy()
                voxel_label[mask1]=predict
            # voxel_label = SemKITTI2train(voxel_label)
            ########################## update center offset ################################
            center, offset = PanopticLabelGenerator(voxel_label,min_bound,intervals,mask1)
            ########################## update center offset ################################
            ########################## preprocess only have center and offset ################################
            label_to_be_save= (center,offset)
            folder_path,filename = splitPath(filenames[i])
            folder_path=folder_path.replace('velodyne','preprocess')
            filename+='.pt'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path,filename)
            torch.save(label_to_be_save,save_path)
    
    return 

class PanopticLabelGenerator_VoxelVersion():
    def __init__(self,grid_size,sigma=5):
        self.grid_size = grid_size
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0,size,1,float)
        y = x[:,np.newaxis]
        x0,y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x-x0) ** 2 + (y - y0) ** 2) / (2 * sigma **2))
        self.indice = np.moveaxis(np.array(np.meshgrid(np.arange(0,256,1),np.arange(0,256,1))),(0,1,2),(2,1,0))

        
    def __call__(self, voxel_label, min_bound, intervals, mask):
        height, width = self.grid_size[0], self.grid_size[1]
        center_pts = []
        center = np.zeros((1, height, width), dtype=np.float32)
        offset = np.zeros((2, height, width), dtype=np.float32)
        inst_data = np.zeros(voxel_label.shape, dtype=np.float32)
        inst_data[mask] = (voxel_label[mask] & 0xffff0000)>>16
        inst_labels = np.unique(inst_data)
        if 0 in inst_labels:
            inst_labels = inst_labels[inst_labels!=0] #delete instance label 0, it isn't foreground instance
        if inst_labels.size == 0:
            return center, offset
        for inst_label in inst_labels:
            mask_ = np.where(inst_data==inst_label)
            center_x, center_y = np.mean(mask_[0]), np.mean(mask_[1])
            # generate center heatmap
            x, y = int(np.floor(center_x)), int(np.floor(center_y))
            center_pts.append([x, y])
            if x < 0 or y < 0 or x >= height or y >= width:
                continue
            sigma = self.sigma
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
            
            c, d = max(0,-ul[0]), min(br[0], height) - ul[0]
            a, b = max(0,-ul[1]), min(br[1], width) - ul[1]
            
            cc, dd = max(0,ul[0]), min(br[0], height)
            aa, bb = max(0, ul[1]), min(br[1], width)
            
            center[0, cc:dd, aa:bb] = np.maximum(center[0, cc:dd, aa:bb], self.g[c:d,a:b])
            offset[0,mask_[0],mask_[1]] = (center_x - mask_[0])
            offset[1,mask_[0],mask_[1]] = (center_y - mask_[1])
        return center, offset


        
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


