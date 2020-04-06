import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu
from dmb.modeling.stereo.cost_processors.utils.correlation1d_cost import correlation1d_cost
from dmb.modeling.stereo.disp_predictors.faster_soft_argmin import FasterSoftArgmin
from dmb.modeling.stereo.cost_processors.utils.hourglass_2d import Hourglass2D
from dmb.ops import ModulatedDeformConv, DeformConv
from dmb.modeling.stereo.disp_samplers.deep_prunner import UniformSampler

class DeformOffsetNet(nn.Module):
    def __init__(self, in_planes, sample_radius_list,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, C=16, batch_norm=True):
        super(DeformOffsetNet, self).__init__()

        self.in_planes = in_planes
        self.sample_radius_list = sample_radius_list
        self.sample_radius_length = len(self.sample_radius_list)
        self.kernel_size = kernel_size
        self.C = C

        self.conv = conv_bn_relu(batch_norm, in_planes, 2*C, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.deformable_sample_blocks = nn.ModuleList()
        for idx in range(len(self.sample_radius_list)):
            self.deformable_sample_blocks.append(
                nn.ModuleDict({
                    'context': Hourglass2D(in_planes=2*C, batch_norm=batch_norm),
                    'concat_context': conv_bn_relu(batch_norm, 4*C, 2*C, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                    'part_offset': nn.Conv2d(2 * C, 3 * kernel_size * kernel_size, kernel_size=3,
                                             stride=1, padding=1, dilation=1, bias=False),
                    'part_dcn': ModulatedDeformConv(1, 1, kernel_size, stride, padding, dilation, bias=bias),
                    'full_offset': nn.Conv2d(2 * C, 3 * kernel_size * kernel_size, kernel_size=3,
                                             stride=1, padding=1, dilation=1, bias=False),
                    'full_dcn': ModulatedDeformConv(1, 1, kernel_size, stride, padding, dilation, bias=bias)
                })
            )

    def forward(self, context, base_disparity):
        context = self.conv(context)
        pre, post = None, None
        pre_radius = 0
        offsets = []
        masks = []
        disparity_sample = base_disparity

        for idx in range(len(self.sample_radius_list)):
            radius = self.sample_radius_list[idx]
            deformable_smaple_block = self.deformable_sample_blocks[idx]
            hourglass_context, pre, post = deformable_smaple_block['context'](context, pre, post)
            context = deformable_smaple_block['concat_context'](torch.cat((context, hourglass_context), dim=1))

            # part range sample
            offset = deformable_smaple_block['part_offset'](context)
            offset = torch.sigmoid(offset)
            x, y, mask = torch.chunk(offset, 3, dim=1)
            offset = 2 * torch.cat((x, y), dim=1) - 1
            sign_offset = torch.sign(offset)
            # [-(x1-x0), (x1-x0)], -(x1-x0)-x0 = -x1, (x1-x0)+x0= x1
            offset = (offset * (radius - pre_radius)) + (sign_offset * pre_radius)
            pre_radius = radius

            offsets.append(offset)
            masks.append(mask)

            dcn_disparity = deformable_smaple_block['part_dcn'](base_disparity, offset, mask)
            disparity_sample = torch.cat((disparity_sample, dcn_disparity), dim=1)

            # full range sample
            offset = deformable_smaple_block['full_offset'](context)
            offset = torch.sigmoid(offset)
            x, y, mask = torch.chunk(offset, 3, dim=1)
            offset = 2 * torch.cat((x, y), dim=1) - 1
            offset = offset * radius
            pre_radius = radius

            offsets.append(offset)
            masks.append(mask)

            dcn_disparity = deformable_smaple_block['full_dcn'](base_disparity, offset, mask)
            disparity_sample = torch.cat((disparity_sample, dcn_disparity), dim=1)

        return disparity_sample, offsets, masks


class ProposalAggregator(nn.Module):
    def __init__(self, max_disp, in_planes, batch_norm=True):
        super(ProposalAggregator, self).__init__()

        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv = conv_bn_relu(batch_norm, in_planes, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.res = Hourglass2D(in_planes=48, batch_norm=batch_norm)
        self.lastConv = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        self.disp_predictor = FasterSoftArgmin(max_disp=max_disp)


    def forward(self, raw_cost):
        cost = self.conv(raw_cost)
        res_cost, _, _ = self.res(cost)
        cost = cost + res_cost
        cost = self.lastConv(cost)

        disp = self.disp_predictor(cost)

        return disp, cost


class MonoStereoSampler(nn.Module):
    def __init__(self, max_disp,
                 sample_radius_list,
                 disparity_sample_number=7,
                 scale=4,
                 in_planes=32,
                 C=16,
                 batch_norm=True):

        super(MonoStereoSampler, self).__init__()

        self.max_disp = max_disp
        self.batch_norm = batch_norm

        self.scale = scale
        self.sample_radius_list = sample_radius_list
        self.disparity_sample_number = disparity_sample_number

        self.in_planes = in_planes
        self.agg_planes = max_disp//scale
        self.C = C

        self.correlation = correlation1d_cost

        self.proposal_aggregator = ProposalAggregator(max_disp//scale, self.agg_planes, batch_norm=batch_norm)

        self.deformable_sampler = DeformOffsetNet(in_planes=1+in_planes+self.agg_planes,
                                                  sample_radius_list=sample_radius_list,
                                                  kernel_size=5, stride=1, padding=2, dilation=1, bias=False,
                                                  C=C, batch_norm=batch_norm)

        # self.uniform_sampler = UniformSampler(disparity_sample_number=disparity_sample_number-1)

    def forward(self, left, right):

        raw_cost = self.correlation(left, right, max_disp=self.max_disp//4)

        proposal_disp, proposal_cost = self.proposal_aggregator(raw_cost)

        offset_context = torch.cat((proposal_disp, proposal_cost, left), dim=1)

        disparity_sample, offsets, masks = self.deformable_sampler(offset_context, proposal_disp)

        return disparity_sample, [proposal_disp], [proposal_cost], offsets, masks


