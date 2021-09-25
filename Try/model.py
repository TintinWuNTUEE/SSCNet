import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from dropblock import DropBlock2D



class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_out = self.conv_classes(x_in)

    return x_in, x_out


class LMSCNet_SS(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies):
    '''
    SSCNet architecture
    :param N: number of classes to be predicted (i.e. 12 for NYUv2)
    '''

    super().__init__()
    self.nbr_classes = class_num
    self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
    self.class_frequencies = class_frequencies
    f = self.input_dimensions[1]

    self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

    self.Encoder_block1 = nn.Sequential(
      nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block2 = nn.Sequential(
      nn.MaxPool2d(2),
      nn.Conv2d(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block3 = nn.Sequential(
      nn.MaxPool2d(2),
      nn.Conv2d(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block4 = nn.Sequential(
      nn.MaxPool2d(2),
      nn.Conv2d(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    # Treatment output 1:8
    self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
    self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
    self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

    # Treatment output 1:4
    self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
    self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)
    self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

    # Treatment output 1:2
    self.deconv1_4          = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=6, padding=2, stride=2)
    self.conv1_2            = nn.Conv2d(int(f*1.5) + int(f/4) + int(f/8), int(f*1.5), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_2 = nn.Conv2d(int(f*1.5), int(f/2), kernel_size=3, padding=1, stride=1)

    # Treatment output 1:1
    self.deconv1_2          = nn.ConvTranspose2d(int(f/2), int(f/2), kernel_size=6, padding=2, stride=2)
    self.conv1_1            = nn.Conv2d(int(f/8) + int(f/4) + int(f/2) + int(f), f, kernel_size=3, padding=1, stride=1)
    self.seg_head_1_1       = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

  def forward(self, x):

    input = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

    # Encoder block
    _skip_1_1 = self.Encoder_block1(input)
    _skip_1_2 = self.Encoder_block2(_skip_1_1)
    _skip_1_4 = self.Encoder_block3(_skip_1_2)
    _skip_1_8 = self.Encoder_block4(_skip_1_4)

    # Out 1_8
    out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)

    # Out 1_4
    out = self.deconv1_8(out_scale_1_8__2D)
    out = torch.cat((out, _skip_1_4), 1)
    out = F.relu(self.conv1_4(out))
    out_scale_1_4__2D = self.conv_out_scale_1_4(out)

    # Out 1_2
    out = self.deconv1_4(out_scale_1_4__2D)
    out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
    out = F.relu(self.conv1_2(out))
    out_scale_1_2__2D = self.conv_out_scale_1_2(out)

    # Out 1_1
    out = self.deconv1_2(out_scale_1_2__2D)
    out = torch.cat((out, _skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D), self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
    out_scale_1_1__2D = F.relu(self.conv1_1(out))
    out_scale_feature, out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)

    # Take back to [W, H, D] axis order
    # change here so that it's shape is torch.Size([2, 20, 32, 256, 256]) & torch.Size([2, 8, 32, 256, 256])

    # out_scale_1_1__3D = out_scale_1_1__3D.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    # out_scale_feature = out_scale_feature.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    scores = {'pred_semantic_1_1': out_scale_1_1__3D, 'pred_semantic_1_1_feature': out_scale_feature}

    return scores

  def weights_initializer(self, m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight)
      nn.init.zeros_(m.bias)

  def weights_init(self):
    self.apply(self.weights_initializer)

  def get_parameters(self):
    return self.parameters()

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''

    target = data['3D_LABEL']
    device, dtype = target.device, target.dtype
    class_weights = self.get_class_weights().to(device=target.device, dtype=target.dtype)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean').to(device=device)

    loss_1_1 = criterion(scores['pred_semantic_1_1'], data['3D_LABEL'].long())

    loss = {'total': loss_1_1, 'semantic_1_1': loss_1_1}

    return loss

  def get_class_weights(self):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

    return weights

  def get_target(self, data):
    '''
    Return the target to use for evaluation of the model
    '''
    return {'1_1': data['3D_LABEL']['1_1']}
    # return data['3D_LABEL']['1_1'] #.permute(0, 2, 1, 3)

  def get_scales(self):
    '''
    Return scales needed to train the model
    '''
    scales = ['1_1']
    return scales

  def get_validation_loss_keys(self):
    return ['total', 'semantic_1_1']

  def get_train_loss_keys(self):
    return ['total', 'semantic_1_1']


class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,n_feature,dilation = 1,group_conv=False,input_batch_norm = False,dropout = 0.,circular_padding = False, dropblock = True, use_vis_fea=False):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        if use_vis_fea:
            self.network = UNet(n_class*n_height,2*n_height,n_feature,dilation,group_conv,input_batch_norm,dropout,circular_padding,dropblock)
        else:
            self.network = UNet(n_class*n_height,n_height,n_feature,dilation,group_conv,input_batch_norm,dropout,circular_padding,dropblock)

    def forward(self, x):
        x,center,offset = self.network(x)
        
        x = x.permute(0,2,3,1)
        new_shape = list(x.size())[:3] + [self.n_height,self.n_class]
        x = x.view(new_shape)
        x = x.permute(0,4,1,2,3)

        return x,center,offset
    
class UNet(nn.Module):
    def __init__(self, n_class,n_height,n_feature,dilation,group_conv,input_batch_norm, dropout,circular_padding,dropblock):
        super(UNet, self).__init__()
        # encoder
        self.inc = inconv(n_height*n_feature, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)

        # semantic decoder
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        # semantic head
        self.outc = outconv(64, n_class)

        # instance decoder
        # self.i_up1 = up(1024, 256, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        # self.i_up2 = up(512, 128, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        # self.i_up3 = up(256, 64, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        # self.i_up4 = up(128, 32, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.i_up4_center = up(128, 32, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.i_up4_offset = up(128, 32, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        # instance head
        self.i_outc_center = outconv(32, 1)
        self.i_outc_offset = outconv(32, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # semantic
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        s_x = self.up4(x, x1)
        s_x = self.outc(self.dropout(s_x))
        # instance
        # i_x = self.i_up1(x5, x4)
        # i_x = self.i_up2(i_x, x3)
        # i_x = self.i_up3(i_x, x2)x

        # i_x = self.i_up4(i_x, x1)
        i_x_center = self.i_up4_center(x, x1)
        i_x_center = self.i_outc_center(self.dropout(i_x_center))

        i_x_offset = self.i_up4_offset(x, x1)
        i_x_offset = self.i_outc_offset(self.dropout(i_x_offset))

        return s_x, i_x_center, i_x_offset

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )                

    def forward(self, x):
        x = self.mpconv(x.contiguous())
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock = False, drop_p = 0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2,groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch,group_conv = group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x




class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


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
