import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample

class Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(Unet, self).__init__()
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
