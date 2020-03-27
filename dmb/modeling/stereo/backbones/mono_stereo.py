import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn, conv_bn_relu, BasicBlock


class MonoStereoBackbone(nn.Module):
    """
    Backbone proposed in MonoStereo.
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True
    Inputs:
        l_img (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        r_img (Tensor): right image, in [BatchSize, 3, Height, Width] layout
    Outputs:
        l_fms (Tensor): left image feature maps, in [BatchSize, 32, Height//4, Width//4] layout

        r_fms (Tensor): right image feature maps, in [BatchSize, 32, Height//4, Width//4] layout
    """

    def __init__(self, in_planes=3, batch_norm=True):
        super(MonoStereoBackbone, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv_2x = nn.Sequential(
            conv_bn_relu(batch_norm, self.in_planes, 16, 3, 2, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 16, 16, 3, 1, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 16, 16, 3, 1, 1, 1, bias=False),
        )

        # For building Basic Block
        self.in_planes = 16

        self.conv_4x = self._make_layer(batch_norm, BasicBlock, 32, 3, 2, 1, 1)
        self.conv_8x = self._make_layer(batch_norm, BasicBlock, 64, 3, 2, 1, 1)
        self.conv_16x = self._make_layer(batch_norm, BasicBlock, 96, 3, 2, 1, 1)
        self.conv_16xd = self._make_layer(batch_norm, BasicBlock, 128, 3, 1, 2, 2)

        self.spp_branch1 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.spp_branch2 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.spp_fuse = nn.Sequential(
            conv_bn_relu(batch_norm, 32*2+128+96, 128, 3, 1, 1, 1, bias=False),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.conv_8x_fuse = nn.Sequential(
            conv_bn_relu(batch_norm, 32+64, 32, 3, 1, 1, 1, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.conv_4x_fuse = nn.Sequential(
            conv_bn_relu(batch_norm, 32+32, 32, 3, 1, 1, 1, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

    def _make_layer(self, batch_norm, block, out_planes, blocks, stride, padding, dilation):
        down_sample_module = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            down_sample_module = conv_bn(
                batch_norm, self.in_planes, out_planes * block.expansion,
                kernel_size=1, stride=stride, padding=0, dilation=1
            )

        layers = []
        layers.append(
            block(batch_norm, self.in_planes, out_planes, stride, down_sample_module, padding, dilation)
        )
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(batch_norm, self.in_planes, out_planes, 1, None, padding, dilation)
            )

        return nn.Sequential(*layers)


    def _forward(self, x):
        # in: [B, 3, H, W], out: [B, 16, H//2, W//2]
        out_2 = self.conv_2x(x)
        # in: [B, 16, H//2, W//2], out: [B, 32, H//4, W//4]
        out_4 = self.conv_4x(out_2)
        # in: [B, 32, H//4, W//4], out: [B, 64, H//8, W//8]
        out_8 = self.conv_8x(out_4)
        # in: [B, 64, H//8, W//8], out: [B, 96, H//16, W//16]
        out_16 = self.conv_16x(out_8)
        # in: [B, 96, H//16, W//16], out: [B, 128, H//16, W//16]
        out_16D = self.conv_16xd(out_16)

        spp_branch1 = self.spp_branch1(out_16D)
        spp_branch1 = F.interpolate(
            spp_branch1, (out_16D.size()[2], out_16D.size()[3]),
            mode='bilinear', align_corners=True
        )

        spp_branch2 = self.spp_branch2(out_16D)
        spp_branch2 = F.interpolate(
            spp_branch2, (out_16D.size()[2], out_16D.size()[3]),
            mode='bilinear', align_corners=True
        )

        spp_fuse = torch.cat(
            (out_16, out_16D, spp_branch1, spp_branch2), 1)
        # in: [B, 192, H//16, W//16], out: [B, 32, H//16, W//16]
        out_16_fuse = self.spp_fuse(spp_fuse)

        # in: [B, 32, H//16, W//16], out: [B, 32, H//8, W//8]
        h_8x, w_8x = out_8.shape[-2:]
        out_16_fuse_up = F.interpolate(out_16_fuse, size=(h_8x, w_8x), mode='bilinear', align_corners=False)
        # in: [B, 32+64, H//8, W//8], out: [B, 32, H//8, W//8]
        out_8_fuse = self.conv_8x_fuse(torch.cat((out_16_fuse_up, out_8), dim=1))

        # in: [B, 32, H//8, W//8], out: [B, 32, H//4, W//4]
        h_4x, w_4x = out_4.shape[-2:]
        out_8_fuse_up = F.interpolate(out_8_fuse, size=(h_4x, w_4x), mode='bilinear', align_corners=False)
        # in: [B, 32+32, H//4, W//4], out: [B, 32, H//4, W//4]
        out_4_fuse = self.conv_4x_fuse(torch.cat((out_8_fuse_up, out_4), dim=1))

        return [out_16_fuse, out_8_fuse, out_4_fuse]


    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))

        l_img, r_img = input

        l_fms = self._forward(l_img)
        r_fms = self._forward(r_img)

        return l_fms, r_fms
