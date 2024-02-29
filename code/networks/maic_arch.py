import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, FeatureHeatmapFusingBlock
from .modules.architecture import FeedbackHourGlass
from .srfbn_hg_arch import FeedbackBlockCustom, FeedbackBlockHeatmapAttention, merge_heatmap_5
from .msab import MSAB
from.gated_fusion import Gated
import math

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class MAIC(nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        in_channels = opt['in_channels']
        out_channels = opt['out_channels']
        num_groups = opt['num_groups']
        hg_num_feature = opt['hg_num_feature']
        hg_num_keypoints = opt['hg_num_keypoints']
        act_type = 'prelu'
        norm_type = None

        self.num_steps = opt['num_steps']
        num_features = opt['num_features']#48
        self.upscale_factor = opt['scale']
        self.detach_attention = opt['detach_attention']
        if self.detach_attention:
            print('Detach attention!')
        else:
            print('Not detach attention!')

        if self.upscale_factor == 8:
            # with PixelShuffle at start, need to upscale 4x only
            stride = 4
            padding = 2
            kernel_size = 8
        else:
            raise NotImplementedError("Upscale factor %d not implemented!" % self.upscale_factor)

        # LR feature extraction block
        self.conv_in = ConvBlock(
            in_channels,
            num_features,
            kernel_size=3,
            act_type=act_type,
            norm_type=norm_type)
        self.msab = MSAB(num_features)
        self.feat_in = nn.PixelShuffle(2)

        # basic block
        self.first_block = FeedbackBlockCustom(num_features, num_groups, self.upscale_factor,
                                   act_type, norm_type, num_features)
        self.block = FeedbackBlockHeatmapAttention(num_features, num_groups, self.upscale_factor, act_type, norm_type, 5, opt['num_fusion_block'], device=device)
        self.block.should_reset = False
        self.out = DeconvBlock(
            num_features,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_type='prelu',
            norm_type=norm_type)
        self.conv_out = ConvBlock(
            num_features,
            out_channels,
            kernel_size=3,
            act_type=None,
            norm_type=norm_type)
        
        self.up = Upsample(scale=4,num_feat=num_features)
        self.HG = FeedbackHourGlass(hg_num_feature, hg_num_keypoints)

    def forward(self, x):
        
        inter_res = nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False)

        
        x = self.conv_in(x)
        x = self.msab(x)
        x = self.feat_in(x)
        sr_outs = []
        heatmap_outs = []
        hg_last_hidden = None
        
        for step in range(self.num_steps):
            if step == 0:
                FB_out_first = self.first_block(x)
                h = torch.add(inter_res, self.conv_out(self.out(FB_out_first)))
                heatmap, hg_last_hidden = self.HG(h, hg_last_hidden)
                self.block.last_hidden = FB_out_first
                assert self.block.should_reset == False
            else:
                FB_out = self.block(x, merge_heatmap_5(heatmap, self.detach_attention))
                h = torch.add(inter_res, self.conv_out(self.out(FB_out)))
                heatmap, hg_last_hidden = self.HG(h, hg_last_hidden) 
            sr_outs.append(h)
            heatmap_outs.append(heatmap)

        return sr_outs, heatmap_outs  
