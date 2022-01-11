import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample

class Simple_Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(Simple_Unet, self).__init__()
        self.trianing = training
        
        self.encoder1 = nn.Conv3d(in_channel,32,3,stride=1,padding=1)
        self.encoder2 = nn.Conv3d(32,64,3,stride=1,padding=1)
        self.encoder3 = nn.Conv3d(64,128,3,stride=1,padding=1)
        self.encoder4 = nn.Conv3d(128,256,3,stride=1,padding=1)
        
        self.decoder1 = nn.Conv3d(256,128,3,stride=1,padding=1)
        self.decoder2 = nn.Conv3d(128,64,3,stride=1,padding=1)
        self.decoder3 = nn.Conv3d(64,32,3,stride=1,padding=1)
        self.decoder4 = nn.Conv3d(32,2,3,stride=1,padding=1)
        
        self.map = nn.Sequential(
            nn.Conv3d(2,out_channel,1,1)
        )
        
    def forward(self,x):
        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode='trilinear'))
        out = torch.add(out,t3)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode='trilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode='trilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode='trilinear'))
        
        out = self.map(out)
        
        return out
    
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
        # x_in = [B,1,80,80,32]

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in)) # x_in = [B,2,80,80,32], all same below

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
            
        x_in = self.relu(y + x_in)  # modified
        x_in = self.conv_classes(x_in)

        return x_in

# 2016 original 3d unet
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),)
        # nn.BatchNorm3d(out_dim),)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]
        
        down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]
        
        # Bridge
        bridge = self.bridge(pool_5) # -> [1, 128, 4, 4, 4]
        
        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        
        trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]
        return out