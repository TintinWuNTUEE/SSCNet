from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import yaml
import random
import sys
import numba as nb
import io_data as SemanticKittiIO
from process_panoptic import PanopticLabelGenerator
import pickle
import errno
class SemanticKITTI_dataloader(Dataset):

  def __init__(self, dataset, phase):
    '''

    :param dataset: The dataset configuration (data augmentation, input encoding, etc)
    :param phase_tag: To differentiate between training, validation and test phase
    '''

    yaml_path, _ = os.path.split(os.path.realpath(__file__))
    self.dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'semantic-kitti.yaml'), 'r'))
    self.nbr_classes = self.dataset_config['nbr_classes']
    self.grid_dimensions = self.dataset_config['grid_dims']   # [W, H, D] [256,32,256]
    self.remap_lut = self.get_remap_lut()
    self.rgb_mean = np.array([0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
    self.rgb_std = np.array([0.30599035, 0.3129534 , 0.31933814])   # images std:  [78.02753826 79.80311686 81.43122464]
    self.root_dir = dataset['ROOT_DIR']
    self.modalities = dataset['MODALITIES']
    self.extensions = {'3D_OCCUPANCY': '.bin', '3D_LABEL': '.label', '3D_OCCLUDED': '.occluded',
                       '3D_INVALID': '.invalid'}
    self.data_augmentation = {'FLIPS': dataset['AUGMENTATION']['FLIPS']}
    self.learning_map = self.dataset_config['learning_map']
    self.filepaths = {}
    self.phase = phase
    self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                       6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                       2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                       2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                       2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])

    self.split = {'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], 'val': [8],
                  'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}

    for modality in self.modalities:
      if self.modalities[modality]:
        self.get_filepaths(modality)
    ########################################################################
    thing_class = self.dataset_config['thing_class']
    self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]
    
    # get class distribution weight 
    epsilon_w = 0.001
    origin_class = self.dataset_config['content'].keys()
    weights = np.zeros((len(self.dataset_config['learning_map_inv'])-1,),dtype = np.float32)
    for class_num in origin_class:
      if self.dataset_config['learning_map'][class_num] != 0:
        weights[self.dataset_config['learning_map'][class_num]-1] += self.dataset_config['content'][class_num]
    self.CLS_LOSS_WEIGHT = 1/(weights + epsilon_w)
    # self.instance_pkl_path = instance_pkl_path
    
    ########################################################################
    # if self.phase != 'test':
    #   self.check_same_nbr_files()

    self.nbr_files = len(self.filepaths['3D_OCCUPANCY'])  # TODO: Pass to something generic

    return

  def get_filepaths(self, modality):
    '''
    Set modality filepaths with split according to phase (train, val, test)
    '''

    sequences = list(sorted(glob(os.path.join(self.root_dir, 'dataset', 'sequences', '*')))[i] for i in self.split[self.phase])

    if self.phase != 'test':

      if modality == '3D_LABEL':
        self.filepaths['3D_LABEL'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
        self.filepaths['3D_INVALID'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
        for sequence in sequences:
          assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
          # Scale 1:1
          self.filepaths['3D_LABEL']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label')))
          self.filepaths['3D_INVALID']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid')))
          # Scale 1:2
          # self.filepaths['3D_LABEL']['1_2'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_2')))
          # self.filepaths['3D_INVALID']['1_2'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_2')))
          # # Scale 1:4
          # self.filepaths['3D_LABEL']['1_4'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_4')))
          # self.filepaths['3D_INVALID']['1_4'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_4')))
          # # Scale 1:8
          # self.filepaths['3D_LABEL']['1_8'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_8')))
          # self.filepaths['3D_INVALID']['1_8'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_8')))

      if modality == '3D_OCCLUDED':
        self.filepaths['3D_OCCLUDED'] = []
        for sequence in sequences:
          assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
          self.filepaths['3D_OCCLUDED'] += sorted(glob(os.path.join(sequence, 'voxels', '*.occluded')))

    if modality == '3D_OCCUPANCY':
      self.filepaths['3D_OCCUPANCY'] = []
      for sequence in sequences:
        assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
        self.filepaths['3D_OCCUPANCY'] += sorted(glob(os.path.join(sequence, 'voxels', '*.bin')))
    if modality == 'PANOPTIC':
      self.filepaths['PANOPTIC'] = []
      for sequence in sequences:
        assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
        self.filepaths['PANOPTIC'] += sorted(glob(os.path.join(sequence,'labels','*.label')))
    # if modality == '2D_RGB':
    #   self.filepaths['2D_RGB'] = []
    #   for sequence in sequences:
    #     assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
    #     self.filepaths['2D_RGB'] += sorted(glob(os.path.join(sequence, 'image_2', '*.png')))[::5]

    return

  def check_same_nbr_files(self):
    '''
    Set modality filepaths with split according to phase (train, val, test)
    '''

    # TODO: Modify for nested dictionaries...
    for i in range(len(self.filepaths.keys()) - 1):
      length1 = len(self.filepaths[list(self.filepaths.keys())[i]])
      length2 = len(self.filepaths[list(self.filepaths.keys())[i+1]])
      assert length1 == length2, 'Error: {} and {} not same number of files'.format(list(self.filepaths.keys())[i],
                                                                                    list(self.filepaths.keys())[i+1])
    return

  def __getitem__(self, idx):
    '''

    '''

    data = {}

    do_flip = 0
    if self.data_augmentation['FLIPS'] and self.phase == 'train':
      do_flip = random.randint(0, 3)
    ################################################################
    annotated_data = np.fromfile(self.filepaths['PANOPTIC'][idx], dtype=np.uint32).reshape((-1,1))
    sem_data = annotated_data & 0xFFFF #delete high 16 digits binary
    sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
    inst_data = annotated_data
    point_label_tuple = (sem_data.astype(np.uint8), inst_data)#add lmscnet output as rawdata:((raw_data[:,:3], sem_data.astype(np.uint8),inst_data))
    ################################################################
    for modality in self.modalities:
      if (self.modalities[modality]) and (modality in self.filepaths):
        data[modality] = self.get_data_modality(modality, idx, do_flip)
    return data, idx, point_label_tuple

  def get_data_modality(self, modality, idx, flip):

    if modality == '3D_OCCUPANCY':
      OCCUPANCY = SemanticKittiIO._read_occupancy_SemKITTI(self.filepaths[modality][idx])
      OCCUPANCY = np.moveaxis(OCCUPANCY.reshape([self.grid_dimensions[0],
                                                 self.grid_dimensions[2],
                                                 self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
      OCCUPANCY = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCUPANCY)
      return OCCUPANCY[None, :, :, :]

    elif modality == '3D_LABEL':
      LABEL_1_1 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_1', idx))
      # LABEL_1_2 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_2', idx))
      # LABEL_1_4 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_4', idx))
      # LABEL_1_8 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_8', idx))
      return {'1_1': LABEL_1_1 } #, '1_2': LABEL_1_2, '1_4': LABEL_1_4, '1_8': LABEL_1_8}

    elif modality == '3D_OCCLUDED':
      OCCLUDED = SemanticKittiIO._read_occluded_SemKITTI(self.filepaths[modality][idx])
      OCCLUDED = np.moveaxis(OCCLUDED.reshape([self.grid_dimensions[0],
                                               self.grid_dimensions[2],
                                               self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
      OCCLUDED = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCLUDED)
      return OCCLUDED

    # elif modality == '2D_RGB':
    #   RGB = SemanticKittiIO._read_rgb_SemKITTI(self.filepaths[modality][idx])
    #   # TODO Standarize, Normalize
    #   RGB = SemanticKittiIO.img_normalize(RGB, self.rgb_mean, self.rgb_std)
    #   RGB = np.moveaxis(RGB, (0, 1, 2), (1, 2, 0)).astype(dtype='float32')  # reshaping [3xHxW]
    #   # There is a problem on the RGB images.. They are not all the same size and I used those to calculate the mapping
    #   # for the sketch... I need images all te same size..
    #   return RGB
    elif modality == 'PANOPTIC':
      pass
    else:
      assert False, 'Specified modality not found'

  def get_label_at_scale(self, scale, idx):

    scale_divide = int(scale[-1])
    INVALID = SemanticKittiIO._read_invalid_SemKITTI(self.filepaths['3D_INVALID'][scale][idx])
    LABEL = SemanticKittiIO._read_label_SemKITTI(self.filepaths['3D_LABEL'][scale][idx])
    if scale == '1_1':
      LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
    LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
    LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
                                       int(self.grid_dimensions[2] / scale_divide),
                                       int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])

    return LABEL

  def read_semantics_config(self, data_path):

    # get number of interest classes, and the label mappings
    DATA = yaml.safe_load(open(data_path, 'r'))
    self.class_strings = DATA["labels"]
    self.class_remap = DATA["learning_map"]
    self.class_inv_remap = DATA["learning_map_inv"]
    self.class_ignore = DATA["learning_ignore"]
    self.n_classes = len(self.class_inv_remap)

    return

  def get_inv_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(self.dataset_config['learning_map_inv'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map_inv'].keys())] = list(self.dataset_config['learning_map_inv'].values())

    return remap_lut

  def get_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(self.dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())

    # in completion we have to distinguish empty and invalid voxels.
    # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    return remap_lut

  def __len__(self):
    """
    Returns the length of the dataset
    """
    # Return the number of elements in the dataset
    return self.nbr_files

class voxel_dataset(Dataset):
  def __init__(self, in_dataset, args, grid_size, ignore_label = 0, return_test = False, fixed_volume_space= True, use_aug = False, max_volume_space = [50,50,1.5], min_volume_space = [-50,-50,-3]):
        'Initialization'
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

        self.panoptic_proc = PanopticLabelGenerator(self.grid_size,sigma=args['gt_generator']['sigma'])
        # if self.instance_aug:
        #     self.inst_aug = instance_augmentation(self.point_cloud_dataset.instance_pkl_path+'/instance_path.pkl',self.point_cloud_dataset.thing_list,self.point_cloud_dataset.CLS_LOSS_WEIGHT,\
        #                                         random_flip=args['inst_aug_type']['inst_global_aug'],random_add=args['inst_aug_type']['inst_os'],\
        #                                         random_rotate=args['inst_aug_type']['inst_global_aug'],local_transformation=args['inst_aug_type']['inst_loc_aug'])

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 3:
            xyz,idx,tuples = data
            labels,insts = tuples
        # elif len(data) == 4:
        #     xyz,labels,insts,feat = data
        #     if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        else: raise Exception('Return invalid data tuple')
        if len(labels.shape) == 1: labels = labels[..., np.newaxis]
        if len(insts.shape) == 1: insts = insts[..., np.newaxis]
        
        # random data augmentation by rotation
        # if self.rotate_aug:
        #     rotate_rad = np.deg2rad(np.random.random()*360)
        #     c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        #     j = np.matrix([[c, s], [-s, c]])
        #     xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        # if self.flip_aug:
        #     flip_type = np.random.choice(4,1)
        #     if flip_type==1:
        #         xyz[:,0] = -xyz[:,0]
        #     elif flip_type==2:
        #         xyz[:,1] = -xyz[:,1]
        #     elif flip_type==3:
        #         xyz[:,:2] = -xyz[:,:2]

        # random instance augmentation
        # if self.instance_aug:
        #     xyz,labels,insts,feat = self.inst_aug.instance_aug(xyz,labels.squeeze(),insts.squeeze(),feat)

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
        
        data_tuple = (voxel_position,processed_label,center,offset)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1)
        
        if len(data) == 3:
            return_fea = return_xyz
        # elif len(data) == 4:
        #     return_fea = np.concatenate((return_xyz,feat),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,insts,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,insts,return_fea)
        return data_tuple

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
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    center2stack=np.stack([d[2] for d in data])
    offset2stack=np.stack([d[3] for d in data])
    grid_ind_stack = [d[4] for d in data]
    point_label = [d[5] for d in data]
    point_inst = [d[6] for d in data]
    xyz = [d[7] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),torch.from_numpy(center2stack),torch.from_numpy(offset2stack),grid_ind_stack,point_label,point_inst,xyz

def collate_fn_BEV_test(data):    
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    center2stack=np.stack([d[2] for d in data])
    offset2stack=np.stack([d[3] for d in data])
    grid_ind_stack = [d[4] for d in data]
    point_label = [d[5] for d in data]
    point_inst = [d[6] for d in data]
    xyz = [d[7] for d in data]
    index = [d[8] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),torch.from_numpy(center2stack),torch.from_numpy(offset2stack),grid_ind_stack,point_label,point_inst,xyz,index

# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

if __name__ == '__main__':
  class CFG:
    
    def __init__(self):
      '''
      Class constructor
      :param config_path:
      '''

      # Initializing dict...
      self._dict = {}
      return

    def from_config_yaml(self, config_path):
      '''
      Class constructor
      :param config_path:
      '''

      # Reading config file
      self._dict = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

      self._dict['STATUS']['CONFIG'] = config_path

      if not 'OUTPUT_PATH' in self._dict['OUTPUT'].keys():
        self.set_output_filename()
        self.init_stats()
        self.update_config()

      return

    def from_dict(self, config_dict):
      '''
      Class constructor
      :param config_path:
      '''

      # Reading config file
      self._dict = config_dict
      return

    def set_output_filename(self):
      '''
      Set output path in the form Model_Dataset_DDYY_HHMMSS
      '''
      datetime = get_date_sting()
      model = self._dict['MODEL']['TYPE']
      dataset = self._dict['DATASET']['TYPE']
      OUT_PATH = os.path.join(self._dict['OUTPUT']['OUT_ROOT'], model + '_' + dataset + '_' + datetime)
      self._dict['OUTPUT']['OUTPUT_PATH'] = OUT_PATH
      return

    def update_config(self, resume=False):
      '''
      Save config file
      '''
      if resume:
        self.set_resume()
      yaml.dump(self._dict, open(self._dict['STATUS']['CONFIG'], 'w'))
      return

    def init_stats(self):
      '''
      Initialize training stats (i.e. epoch mean time, best loss, best metrics)
      '''
      self._dict['OUTPUT']['BEST_LOSS'] = 999999999999
      self._dict['OUTPUT']['BEST_METRIC'] = -999999999999
      self._dict['STATUS']['LAST'] = ''
      return

    def set_resume(self):
      '''
      Update resume status dict file
      '''
      if not self._dict['STATUS']['RESUME']:
        self._dict['STATUS']['RESUME'] = True
      return

    def finish_config(self):
      self.move_config(os.path.join(self._dict['OUTPUT']['OUTPUT_PATH'], 'config.yaml'))
      return

    def move_config(self, path):
      # Remove from original path
      os.remove(self._dict['STATUS']['CONFIG'])
      # Change ['STATUS']['CONFIG'] to new path
      self._dict['STATUS']['CONFIG'] = path
      # Save to routine output folder
      yaml.dump(self._dict, open(path, 'w'))

      return
  _cfg = CFG()
  _cfg.from_config_yaml("../SSC_configs/LMSCNet_SS.yaml")
  data = SemanticKITTI_dataloader(_cfg._dict["DATASET"],'train')
  xyz,idx,tuples = data.__getitem__(0)
  print(['PANOPTIC'])