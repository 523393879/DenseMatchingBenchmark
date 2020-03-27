import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu, BasicBlock
from dmb.modeling.stereo.cost_processors.utils.correlation1d_cost import Correlation1dCost
from dmb.modeling.stereo.disp_predictors.faster_soft_argmin import FasterSoftArgmin


class MonoStereoAggregator(nn.Module):
    def __init__(self, max_disp, in_planes, upScale=1, batch_norm=True):
        super(MonoStereoAggregator, self).__init__()

        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv_planes_list = [in_planes, 128, 96, 64, 48, 48]
        self.dilation_list = [1, 1, 2, 4, 8, 1]
        self.aggLayers = nn.ModuleList()
        for idx in range(1, len(self.conv_planes_list)):
            self.aggLayers.append(conv_bn_relu(
                batch_norm, in_planes=self.conv_planes_list[idx - 1],
                out_planes=self.conv_planes_list[idx],
                kernel_size=3, stride=1, padding=self.dilation_list[idx],
                dilation=self.dilation_list[idx], bias=False,
            ))
        self.fullCost = nn.Sequential(
            conv_bn_relu(batch_norm, 48, self.max_disp, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.ConvTranspose2d(self.max_disp, self.max_disp, kernel_size=upScale*2, stride=upScale, padding=upScale//2, bias=False),
        )

        self.disp_predictor = FasterSoftArgmin(max_disp=max_disp)

    def forward(self, cost):
        for layer in self.aggLayers:
            cost = layer(cost)

        cost = self.fullCost(cost)

        disp = self.disp_predictor(cost)

        return disp, cost


class MonoStereoSampler(nn.Module):
    def __init__(self, max_disp, disparity_sample_number, in_planes, batch_norm=True):
        super(MonoStereoSampler, self).__init__()

        self.max_disp = max_disp
        self.disparity_sample_number = disparity_sample_number
        self.batch_norm = batch_norm

        self.correlation = Correlation1dCost(max_disp//4)

        self.aggregator_4 = MonoStereoAggregator(max_disp, in_planes, 4, batch_norm)

    def uniform_sampler(self, base_disparity):

        device = base_disparity.device
        b, c, h, w = base_disparity.shape

        # to get 'disparity_sample_number' samples around base disparity sample,
        sample_index = torch.arange(-self.disparity_sample_number, self.disparity_sample_number + 1, 1, device=device).float()
        # [B, disparity_sample_number * 2 + 1, H, W]
        sample_index =sample_index.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()

        # [B, disparity_sample_number * 2 + 1, H, W]
        disparity_sample = base_disparity + sample_index

        disparity_sample = torch.clamp(disparity_sample, min=0.0)

        return disparity_sample


    def forward(self, left, right):
        # [B, 32, H//16, W//16], [B, 32, H//8, W//8], [B, 32, H//4, W//4]
        left_16, left_8, left_4 = left
        right_16, right_8, right_4 = right

        raw_cost = self.correlation(left_4, right_4)

        pred_4, cost = self.aggregator_4(raw_cost)

        disparity_sample = self.uniform_sampler(pred_4)

        return disparity_sample, [pred_4], [cost]


