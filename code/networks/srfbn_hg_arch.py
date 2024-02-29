import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, FeatureHeatmapFusingBlock
from .modules.architecture import StackedHourGlass
from .srfbn_arch import FeedbackBlock

def merge_heatmap_5(heatmap_in, detach):
    '''
    merge 68 heatmap to 5
    heatmap: B*N*32*32
    '''
    # landmark[36:42], landmark[42:48], landmark[27:36], landmark[48:68]
    heatmap = heatmap_in.clone()
    max_heat = heatmap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    max_heat = torch.max(max_heat, torch.ones_like(max_heat) * 0.05)
    heatmap /= max_heat
    if heatmap.size(1) == 5:
        return heatmap.detach() if detach else heatmap
    elif heatmap.size(1) == 68:
        new_heatmap = torch.zeros_like(heatmap[:, :5])
        new_heatmap[:, 0] = heatmap[:, 36:42].sum(1) # left eye
        new_heatmap[:, 1] = heatmap[:, 42:48].sum(1) # right eye
        new_heatmap[:, 2] = heatmap[:, 27:36].sum(1) # nose
        new_heatmap[:, 3] = heatmap[:, 48:68].sum(1) # mouse
        new_heatmap[:, 4] = heatmap[:, :27].sum(1) # face silhouette
        return new_heatmap.detach() if detach else new_heatmap
    elif heatmap.size(1) == 194: # Helen
        new_heatmap = torch.zeros_like(heatmap[:, :5])
        tmp_id = torch.cat((torch.arange(134, 153), torch.arange(174, 193)))
        new_heatmap[:, 0] = heatmap[:, tmp_id].sum(1) # left eye
        tmp_id = torch.cat((torch.arange(114, 133), torch.arange(154, 173)))
        new_heatmap[:, 1] = heatmap[:, tmp_id].sum(1) # right eye
        tmp_id = torch.arange(41, 57)
        new_heatmap[:, 2] = heatmap[:, tmp_id].sum(1) # nose
        tmp_id = torch.arange(58, 113)
        new_heatmap[:, 3] = heatmap[:, tmp_id].sum(1) # mouse
        tmp_id = torch.arange(0, 40)
        new_heatmap[:, 4] = heatmap[:, tmp_id].sum(1) # face silhouette
        return new_heatmap.detach() if detach else new_heatmap
    else:
        raise NotImplementedError('Fusion for face landmark number %d not implemented!' % heatmap.size(1))
        
        
class FeedbackBlockHeatmapAttention(FeedbackBlock):
    def __init__(self,
                 num_features,
                 num_groups,
                 upscale_factor,
                 act_type,
                 norm_type,
                 num_heatmap,
                 num_fusion_block,
                 device=torch.device('cuda')):
        super().__init__(num_features,
                         num_groups,
                         upscale_factor,
                         act_type,
                         norm_type,
                         device)
        self.fusion_block = FeatureHeatmapFusingBlock(num_features,
                                                      num_heatmap,
                                                      num_fusion_block)
      
    def forward(self, x, heatmap):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).to(self.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)
        # fusion
        x = self.fusion_block(x, heatmap)
        output = self.swinir(x)+x
        self.last_hidden = output

        return output
    
    
class FeedbackBlockCustom(FeedbackBlock):
    def __init__(self, num_features, num_groups, upscale_factor, act_type,
                 norm_type, num_features_in):
        super(FeedbackBlockCustom, self).__init__(
            num_features, num_groups, upscale_factor, act_type, norm_type)
        self.compress_in = ConvBlock(num_features_in, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)
    def forward(self, x):
        x = self.compress_in(x)#1×1卷积
        output = self.swinir(x)+x

        return output
