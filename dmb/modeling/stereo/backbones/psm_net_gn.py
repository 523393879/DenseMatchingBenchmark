import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import consistent_padding_with_dilation
from dmb.modeling.stereo.layers.basic_layers import group_norm


def conv_gn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        ),
        group_norm(out_planes, ),
    )


def conv_gn_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias),
        group_norm(out_planes),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride, downsample, padding, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_gn_relu(
            in_planes=in_planes, out_planes=out_planes,
            kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.conv2 = conv_gn(
            in_planes=out_planes, out_planes=out_planes,
            kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out


class PSMNetGNBackbone(nn.Module):
    """
    Backbone proposed in PSMNet.
    Args:
        in_planes (int): the channels of input
        group_norm (bool): whether use batch normalization layer, default True
    Inputs:
        l_img (Tensor): left image
        r_img (Tensor): right image
    Outputs:
        l_fms (Tensor): left image feature maps
        r_fms (Tensor): right image feature maps
    """

    def __init__(self, in_planes=3):
        super(PSMNetGNBackbone, self).__init__()
        self.in_planes = in_planes

        self.firstconv = nn.Sequential(
            conv_gn_relu(self.in_planes, 32, 3, 2, 1, 1, bias=False),
            conv_gn_relu(32, 32, 3, 1, 1, 1, bias=False),
            conv_gn_relu(32, 32, 3, 1, 1, 1, bias=False),
        )

        # For building Basic Block
        self.in_planes = 32

        self.layer1 = self._make_layer(group_norm, BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(group_norm, BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(group_norm, BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(group_norm, BasicBlock, 128, 3, 1, 2, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            conv_gn_relu(128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            conv_gn_relu(128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            conv_gn_relu(128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            conv_gn_relu(128, 32, 1, 1, 0, 1, bias=False),
        )
        self.lastconv = nn.Sequential(
            conv_gn_relu(320, 128, 3, 1, 1, 1, bias=False),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, dilation=1, bias=False)
        )

    def _make_layer(self, group_norm, block, out_planes, blocks, stride, padding, dilation):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = conv_gn(
                self.in_planes, out_planes * block.expansion,
                kernel_size=1, stride=stride, padding=0, dilation=1
            )

        layers = []
        layers.append(
            block(self.in_planes, out_planes, stride, downsample, padding, dilation)
        )
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.in_planes, out_planes, 1, None, padding, dilation)
            )

        return nn.Sequential(*layers)

    def _forward(self, x):
        output_2_0 = self.firstconv(x)
        output_2_1 = self.layer1(output_2_0)
        output_4_0 = self.layer2(output_2_1)
        output_4_1 = self.layer3(output_4_0)
        output_8 = self.layer4(output_4_1)

        output_branch1 = self.branch1(output_8)
        output_branch1 = F.interpolate(
            output_branch1, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch2 = self.branch2(output_8)
        output_branch2 = F.interpolate(
            output_branch2, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch3 = self.branch3(output_8)
        output_branch3 = F.interpolate(
            output_branch3, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch4 = self.branch4(output_8)
        output_branch4 = F.interpolate(
            output_branch4, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_feature = torch.cat(
            (output_4_0, output_8, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))

        l_img, r_img = input

        l_fms = self._forward(l_img)
        r_fms = self._forward(r_img)

        return l_fms, r_fms
