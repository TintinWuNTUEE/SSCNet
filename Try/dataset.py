from torch.utils import data
from torch.utils.data import Dataset
from glob import glob
import yaml
import numpy as np


class SemanticKITTI(Dataset):
    def __init__(self, dataset_setting, phase):
        with open("semantic-kitti.yaml",'r') as stream:
            self.dataset_config = yaml.safe_load(stream)
        self.nbr_classes = self.dataset_config['nbr_classes']
        self.grid_dimensions = self.dataset_config['grid_dims']
        self.remap_lut = self.get_remap_lut()
        self.rgb_mean = np.array([0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
        self.rgb_std = np.array([0.30599035, 0.3129534 , 0.31933814])   # images std:  [78.02753826 79.80311686 81.43122464]
        self.root_dir = dataset_setting['ROOT_DIR']
        self.modalities = dataset_setting['MODALITIES']
        self.extensions = {'3D_OCCUPANCY': '.bin', '3D_LABEL': '.label', '3D_OCCLUDED': '.occluded', '3D_INVALID': '.invalid'}
        self.data_augmentation = {'FLIPS': dataset_setting['AUGMENTATION']['FLIPS']}
        self.learning_map = self.dataset_config['learning_map']
        self.filepaths = {}
        self.phase = phase
        self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                       6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                       2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                       2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                       2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
        
        self.split = {'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], 'val': [8], 'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}

        for modality in self.modalities:
            if self.modalities[modality]:
                self.get_filepaths(modality)
    


    def get_filepaths(self, modality):
        pass
    #todo

          
          
          
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