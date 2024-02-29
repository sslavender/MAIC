import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock
from .gated_fusion import Gated
import torch.nn.functional as F
from .CSA import csablock

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, dropout_rate=0.3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.PReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        avg_pool = self.avg_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        avg_pool = self.dropout(avg_pool)
        channel_attention = self.fc(avg_pool)
        channel_attention = channel_attention.unsqueeze(2).unsqueeze(3)
        channel_attention = channel_attention.expand_as(x)
        return x * channel_attention

class MSAB(nn.Module):
    def __init__(self, in_channels):
        super(MSAB, self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels=in_channels,kernel_size=1,padding=0)
        self.conv_1 = nn.Conv2d(in_channels,out_channels=in_channels,kernel_size=1,padding=0)
        self.conv_3 = nn.Conv2d(in_channels,out_channels=in_channels,kernel_size=3,padding=1)
        self.conv_5 = nn.Conv2d(in_channels,out_channels=in_channels,kernel_size=5,padding=2)
        self.conv_7 = nn.Conv2d(in_channels,out_channels=in_channels,kernel_size=7,padding=3)
        self.bn = nn.BatchNorm2d(in_channels*4)
        self.cbam = csablock(in_channels*4)
        self.relu = nn.PReLU()
    def forward(self, x):
        x = self.relu(self.conv(x))
        x_1 = self.conv_1(x)
        x_3 = self.conv_3(x)
        x_5 = self.conv_5(x)
        x_7 = self.conv_7(x)
        c = torch.cat((x_1,x_3,x_5,x_7),dim=1)
        attention = self.cbam(c)
  
        return attention


