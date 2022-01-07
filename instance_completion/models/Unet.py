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
        