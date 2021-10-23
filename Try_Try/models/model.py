import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from dropblock import DropBlock2D
from models.BEV_Unet import BEV_Unet
from models.LMSCNet_SS import LMSCNet_SS
def get_model(_cfg,dataset):
    grid_dims = [256, 32, 256]
    feature_shape = [8,32,256,256]
    class_frequencies = dataset.class_frequencies
    nbr_classes = 20
    model1 = LMSCNet_SS(nbr_classes, grid_dims, class_frequencies)
    model2 = BEV_Unet(20,feature_shape[1],feature_shape[0], input_batch_norm = True, dropout = 0.5, circular_padding = False, use_vis_fea=False)
    return model1, model2

if __name__ == '__main__':
    class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                       6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                       2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                       2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                       2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
    nbr_classes = 20
    grid_dims = [256, 32, 256]  # (W, H, D)
    
    model = LMSCNet_SS(nbr_classes, grid_dims, class_frequencies).to('cuda')
    model = model.train()
    input = {'3D_OCCUPANCY':torch.from_numpy(np.ones([2,1,256,32,256],dtype=np.float32)).to('cuda')}
    print(input['3D_OCCUPANCY'].type())
    output = model(input)
    print(output['pred_semantic_1_1'].shape)
    print(output['pred_semantic_1_1_feature'].shape)

    output_shape = output['pred_semantic_1_1_feature'].shape

    output['pred_semantic_1_1'] = output['pred_semantic_1_1'].view(-1,640,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]
    output['pred_semantic_1_1_feature'] = output['pred_semantic_1_1_feature'].view(-1,256,256,256)  # [bs, C, H, W, D] -> [bs, C*H, W, D]

    second_model = BEV_Unet(20,output_shape[2],output_shape[1], input_batch_norm = True, dropout = 0.5, circular_padding = False, use_vis_fea=False).to('cuda').train()
    z, center, offset = second_model(output['pred_semantic_1_1_feature'])

    print(z.shape)
    print(center.shape)
    print(offset.shape)
