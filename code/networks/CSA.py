import numpy as np
import torch
from torch import nn
from torch.nn import init




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

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class   csablock(nn.Module):

    def __init__(self, channel=512,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        sa = self.sa(out)
        out=out * sa
        return out+residual


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    kernel_size=input.shape[2]
    cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
    output=cbam(input)
    print(output.shape)

    