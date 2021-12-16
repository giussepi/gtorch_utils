# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet3_plus/layers """

import torch
import torch.nn as nn
# import torch.nn.functional as F

from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from gtorch_utils.nns.models.segmentation.unet3_plus.init_weights import init_weights
from gtorch_utils.utils.images import apply_padding


class unetConv2(nn.Module):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/layers.py
    """

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1, init_type=UNet3InitMethod.KAIMING):
        super().__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.init_type = init_type
        s = stride
        p = padding

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type=self.init_type)

    def forward(self, inputs):
        x = inputs

        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/layers.py
    """

    def __init__(self, in_size, out_size, is_deconv, n_concat=2, init_type=UNet3InitMethod.KAIMING):
        super().__init__()
        # TODO: the issue is here, the channels must be reduced by half to perform the
        # the concatenation properly
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.init_type = init_type
        self.doubleconv3x3 = unetConv2(out_size*2, out_size, False)
        self.conv2x2 = nn.Conv2d(in_size, in_size // 2, kernel_size=2, stride=1, padding=1)

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

            # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type=self.init_type)

    def forward(self, inputs0, inputs1):
        inputs0 = self.up(inputs0)
        inputs0 = self.conv2x2(inputs0)
        # # ###################################################################
        # #                              sfsdf                              #
        # diffY = inputs1.size()[2] - inputs0.size()[2]
        # diffX = inputs1.size()[3] - inputs0.size()[3]

        # inputs0 = F.pad(
        #     inputs0,
        #     [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        # )
        # # ###################################################################
        inputs0 = apply_padding(inputs0, inputs1)

        outputs0 = torch.cat([inputs1, inputs0], dim=1)

        return self.doubleconv3x3(outputs0)


class unetUp_origin(nn.Module):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/layers.py
    """

    def __init__(self, in_size, out_size, is_deconv, n_concat=2, init_type=UNet3InitMethod.KAIMING):
        super().__init__()
        self.init_type = init_type
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type=self.init_type)

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class FinalConv(nn.Module):
    """
    Unet final 1x1 convolution

    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)
