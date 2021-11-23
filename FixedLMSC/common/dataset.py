import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, dataset
from glob import glob
import yaml
import numpy as np
import os
import random
import common.io_data as SemanticKittiIO
def get_dataset(_cfg):
    if _cfg._dict['DATASET']['TYPE'] == 'SemanticKITTI':
        ds_train = SemanticKITTI(_cfg._dict['DATASET'],'train')
        ds_val = SemanticKITTI(_cfg._dict['DATASET'],'val')
        # ds_test = SemanticKITTI(_cfg._dict['DATASET'],'test')

    _cfg._dict['DATASET']['SPLIT'] = {'TRAIN': len(ds_train), 'VAL': len(ds_val)}

    dataset = {}

    train_batch_size = _cfg._dict['TRAIN']['BATCH_SIZE']
    val_batch_size = _cfg._dict['VAL']['BATCH_SIZE']
    num_workers = _cfg._dict['DATALOADER']['NUM_WORKERS']

    dataset['train'] = DataLoader(ds_train, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
    dataset['val']   = DataLoader(ds_val,   batch_size=val_batch_size, num_workers=num_workers, shuffle=False)
    # dataset['test']  = DataLoader(ds_test,   batch_size=val_batch_size, num_workers=num_workers, shuffle=False)

    return dataset
class SemanticKITTI(Dataset):
    def __init__(self, dataset_setting, phase):
        with open("configs/semantic-kitti.yaml",'r') as stream:
            self.dataset_config = yaml.safe_load(stream)
        self.nbr_classes = self.dataset_config['nbr_classes']
        self.grid_dimensions = self.dataset_config['grid_dims']
        self.remap_lut = self.get_remap_lut()
        self.rgb_mean = np.array([0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
        self.rgb_std = np.array([0.30599035, 0.3129534 , 0.31933814])   # images std:  [78.02753826 79.80311686 81.43122464]
        self.root_dir = dataset_setting['ROOT_DIR']
        self.modalities = dataset_setting['MODALITIES']
        self.extensions = {'3D_OCCUPANCY': '.bin', '3D_LABEL': '.label', '3D_OCCLUDED': '.occluded', '3D_INVALID': '.invalid', 'PREPROCESS': '.pt'}
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


        ################################################
        thing_class = self.dataset_config['thing_class']
        self.thing_list = [class_nbr for class_nbr, is_thing in thing_class.items() if is_thing]

        #get class ditribution weight
        epsilon_w = 0.001
        origin_class = self.dataset_config['content'].keys()
        weights = np.zeros((len(self.dataset_config['learning_map_inv'])-1,),dtype=np.float32)
        for class_num in origin_class:
            if self.dataset_config['learning_map'][class_num] != 0:
                weights[self.dataset_config['learning_map'][class_num]-1] += self.dataset_config['content'][class_num]
        self.CLS_LOSS_WEIGHT = 1/(weights + epsilon_w)

        ################################################

        self.nbr_files = len(self.filepaths['3D_OCCUPANCY'])

        return


    def get_filepaths(self, modality):
        print(os.path.join(self.root_dir, 'dataset', 'sequences', '*'))
        sequences = list(sorted(glob(os.path.join(self.root_dir, 'dataset', 'sequences', '*')))[i] for i in self.split[self.phase])

        if self.phase != 'test':
            if modality == '3D_LABEL':
                self.filepaths['3D_LABEL'] = []
                self.filepaths['3D_INVALID'] = []
                for sequence in sequences:
                    assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)

                    self.filepaths['3D_LABEL'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label')))
                    self.filepaths['3D_INVALID'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid')))


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

        if modality == 'PREPROCESS':
            self.filepaths['PREPROCESS'] = []
            for sequence in sequences:
                assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)

                self.filepaths['PREPROCESS'] += sorted(glob(os.path.join(sequence, 'preprocess', '*.pt')))
                print( sorted(glob(os.path.join(sequence, 'preprocess', '*.pt'))))

        #????
        # if modality == 'PANOPTIC':
        #     self.filepaths['PANOPTIC'] = []
        #     for sequence in sequences:
        #         assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)

        #         self.filepaths['PANOPTIC'] += sorted(glob(os.path.join(sequence, 'labels', '*.label')))

        return
          
          
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


    def __getitem__(self, index):
        data = {}
        do_flip = 0
        if self.data_augmentation['FLIPS'] and self.phase == 'train':
            do_flip =random.randint(0,3)
        ##########################################################
        #???
        # annotated_data = np.fromfile(self.filepaths['PANOPTIC'][index], dtype=np.int32).reshape((-1,1))
        # sem_data = annotated_data & 0XFFFF #delete high 16 digits binary
        # sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
        # inst_data = annotated_data
        # point_label_tuple = (sem_data.astype(np.uint8), inst_data)
        ##########################################################
        for modality in self.modalities:
            if (self.modalities[modality]) and (modality in self.filepaths):
                data[modality] = self.get_data_modality(modality, index, do_flip)
        # del data['PANOPTIC']
        return data, index


    def get_data_modality(self, modality, index, flip):
        if modality == '3D_OCCUPANCY':
            OCCUPANCY = SemanticKittiIO._read_occupancy_SemKITTI(self.filepaths[modality][index])
            OCCUPANCY = np.moveaxis(OCCUPANCY.reshape([self.grid_dimensions[0],
                                               self.grid_dimensions[2],
                                               self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
            OCCUPANCY = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCUPANCY)
            return OCCUPANCY[None, :, :, :]

        elif modality == '3D_LABEL':
            INVALID = SemanticKittiIO._read_invalid_SemKITTI(self.filepaths['3D_INVALID'][index])
            LABEL = SemanticKittiIO._read_label_SemKITTI(self.filepaths['3D_LABEL'][index])
            # Remap 20 classes semanticKITTI SSC
            LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(np.float32)
            # Setting all voxels which are marked invalid to unknow class
            LABEL[np.isclose(INVALID,1)] = 255
            LABEL = np.moveaxis(LABEL.reshape([self.grid_dimensions[0],
                                               self.grid_dimensions[2],
                                               self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])

            LABEL = SemanticKittiIO.data_augmentation_3Dflips(flip, LABEL)
            return LABEL

        elif modality == '3D_OCCLUDED':
            OCCLUDED = SemanticKittiIO._read_occluded_SemKITTI(self.filepaths[modality][index])
            OCCLUDED = np.moveaxis(OCCLUDED.reshape([self.grid_dimensions[0],
                                               self.grid_dimensions[2],
                                               self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
            OCCLUDED = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCLUDED)
            return OCCLUDED

        elif modality == 'PREPROCESS':
            PREPROCESS = torch.load(self.filepaths[modality][index])
            sem_label, center_label, offset_label = PREPROCESS
            # Flipping around the XS axis...
            if np.isclose(flip, 1):
                sem_label = np.flip(sem_label, axis=0).copy()
                center_label = np.flip(center_label, axis=1).copy()
                offset_label = np.flip(offset_label, axis=1).copy()

            # Flipping around the Y axis...
            if np.isclose(flip, 2):
                sem_label = np.flip(sem_label, axis=1).copy()
                center_label = np.flip(center_label, axis=2).copy()
                offset_label = np.flip(offset_label, axis=2).copy()

            # Flipping around the X and the Y axis...
            if np.isclose(flip, 3):
                sem_label = np.flip(np.flip(sem_label, axis=0), axis=1).copy()
                center_label = np.flip(np.flip(center_label, axis=1), axis=2).copy()
                offset_label = np.flip(np.flip(offset_label, axis=1), axis=2).copy()

            return [sem_label, center_label, offset_label]

        # ?????
        elif modality == 'PANOPTIC':
            pass

        else:
            assert False, 'Specified modality not found'


        pass #todo

    def __len__(self):
        """
        Return the length of the dataset
        """
        return self.nbr_files

if __name__ == '__main__':
    with open('LMSCNet_SS.yaml','r') as stream:
        config = yaml.safe_load(stream)
    dataset = SemanticKITTI(config['DATASET'],'train')
    dataloader = DataLoader(dataset,batch_size=2,num_workers=4,shuffle=False)
    for t, (data, indices) in enumerate(dataset):
        data = data['3D_LABEL']
        print(data)
        break