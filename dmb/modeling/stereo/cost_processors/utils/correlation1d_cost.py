import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import SpatialCorrelationSampler
from dmb.modeling.stereo.cost_processors.utils.correlation_1d import correlation_1d

class Correlation1dCost(nn.Module):
    def __init__(self, max_disp, kernel_size=1, stride=1, padding=0, dilation_patch=1):
        super(Correlation1dCost, self).__init__()
        self.max_disp = max_disp
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation_patch = dilation_patch

        # for a pixel of left image at (x, y), it will calculates correlation cost volume
        # with pixel of right image at (xr, y), where xr in [x-max_disp, x+max_disp]
        # but we only need
        self.corr = SpatialCorrelationSampler(patch_size=(1, max_disp*2-1),
                                              kernel_size=kernel_size,
                                              stride=stride, padding=padding,
                                              dilation_patch=dilation_patch)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, feat1, feat2):
        # [B, 1, max_disp*2-1, H, W]
        out = self.corr(feat1, feat2)

        # [B, max_disp*2-1, H, W]
        out = out.squeeze(1)

        # [B, max_disp, H, W]
        out = out[:, :self.max_disp, :, :]

        out = self.relu(out)

        return out



# class Correlation1dCost(nn.Module):
#     def __init__(self, max_disp=192, start_disp=0, dilation=1, disp_sample=None):
#         super(Correlation1dCost, self).__init__()
#         self.max_disp = max_disp
#         self.start_disp = start_disp
#         self.dilation = dilation
#         self.disp_sample = disp_sample
#
#     def forward(self, left, right):
#         # [B, max_disp, H, W]
#         correlation_1d_cost = correlation_1d(left, right, max_disp=self.max_disp,
#                                    start_disp=self.start_disp, dilation=self.dilation,
#                                    disp_sample=self.disp_sample)
#         return correlation_1d_cost


