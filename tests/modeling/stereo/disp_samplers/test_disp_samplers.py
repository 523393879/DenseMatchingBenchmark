import torch
import torch.nn as nn
import numpy as np
from mmcv import Config

import unittest
import time

from dmb.modeling.stereo.disp_samplers import build_disp_sampler


class TestDispSamplers(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda:4')
        self.iters = 50

        self.sampler_type = 'MONOSTEREO'


    def timeTemplate(self, module, module_name, *args, **kwargs):
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        if isinstance(module, nn.Module):
            module.eval()

        start_time = time.time()

        for i in range(self.iters):
            with torch.no_grad():
                if len(args) > 0:
                    module(*args)
                if len(kwargs) > 0:
                    module(**kwargs)
                torch.cuda.synchronize(self.device)
        end_time = time.time()
        avg_time = (end_time - start_time) / self.iters
        print('{} reference forward once takes {:.4f}s, i.e. {:.2f}fps'.format(module_name, avg_time, (1 / avg_time)))

        if isinstance(module, nn.Module):
            module.train()

    # @unittest.skip("Just skipping")
    def test_speed(self):
        max_disp = 192
        scale = 4
        SH, SW = 544, 960
        B, C, H, W = 1, 8, SH//scale, SW//scale

        left = torch.rand(B, 2*C, H, W).to(self.device)
        right = torch.rand(B, 2*C, H, W).to(self.device)

        cfg = Config(dict(
            model=dict(
                batch_norm=True,
                disp_sampler=dict(
                    type=self.sampler_type,
                    # max disparity
                    max_disp=max_disp,
                    # the down-sample scale of the input feature map
                    scale=scale,
                    # the number of diaparity samples
                    disparity_sample_number=9,
                    # the in planes of extracted feature
                    in_planes=2*C,
                    # the base channels of convolution layer in this network
                    C=C,
                    sample_radius_list = [4, 8, 16, 32]
                ),
            )
        ))

        disp_sampler = build_disp_sampler(cfg).to(self.device)

        print('*' * 60)
        print('Speed Test!')

        print('*' * 60)
        print('Correlation Speed!')
        self.timeTemplate(disp_sampler.correlation, 'Correlation', left, right)

        raw_cost = disp_sampler.correlation(left, right, max_disp//scale)

        print('*' * 60)
        print('Diaprity proposal aggregator Speed!')
        self.timeTemplate(disp_sampler.proposal_aggregator, 'Aggregator', raw_cost)

        proposal_disp, proposal_cost = disp_sampler.proposal_aggregator(raw_cost)

        print('*' * 60)
        print('Diaprity proposal sampler Speed!')
        context = torch.cat((proposal_disp, proposal_cost, left), dim=1)
        self.timeTemplate(disp_sampler.deformable_sampler, 'DeformableSampler', context, proposal_disp)

        print('*' * 60)
        print('Wholistic Module Speed!')
        self.timeTemplate(disp_sampler, self.sampler_type, left, right)


if __name__ == '__main__':
    unittest.main()

'''

Test on GTX1080Ti, 544x960

perform on full scale
MONOSTEREO reference forward once takes 0.0604s, i.e. 16.55fps
Correlation reference forward once takes 0.0604s, i.e. 296.75fps
Aggregator reference forward once takes 0.0540s, i.e. 18.52fps
NearSampler reference forward once takes 0.0034s, i.e. 293.25fps

only perform on 1/4 scale
MONOSTEREO reference forward once takes 0.0054s, i.e. 186.68fps
Correlation reference forward once takes 0.0033s, i.e. 303.29fps
Aggregator reference forward once takes 0.0033s, i.e. 305.13fps
NearSampler reference forward once takes 0.0012s, i.e. 866.98fps

MONOSTEREO reference forward once takes 0.0105s, i.e. 95.00fps
Correlation reference forward once takes 0.3979s, i.e. 2.51fps
Aggregator reference forward once takes 0.0033s, i.e. 298.66fps
DeformableSampler reference forward once takes 0.0098s, i.e. 101.99fps
'''