# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet/unet_parts """

import torch

from gtorch_utils.utils.images import apply_padding


class DoubleConv(torch.nn.Module):
    """
    (convolution => [BN] => ReLU) * 2

    Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, batchnorm_cls=torch.nn.BatchNorm2d):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.batchnorm_cls = batchnorm_cls

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        if self.mid_channels is not None:
            assert isinstance(self.mid_channels, int), type(self.mid_channels)
        assert issubclass(self.batchnorm_cls, torch.nn.modules.batchnorm._BatchNorm), type(self.batchnom_cls)

        if not self.mid_channels:
            self.mid_channels = self.out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, padding=1),
            batchnorm_cls(self.mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, padding=1),
            batchnorm_cls(self.out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """
    Downscaling with maxpool then double conv

    Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, in_channels, out_channels, batchnorm_cls=torch.nn.BatchNorm2d):
        super().__init__()
        self.batchnorm_cls = batchnorm_cls

        assert issubclass(self.batchnorm_cls, torch.nn.modules.batchnorm._BatchNorm), type(self.batchnom_cls)

        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, batchnorm_cls=self.batchnorm_cls)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """
    Upscaling then double conv

    Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, in_channels, out_channels, bilinear=True, batchnorm_cls=torch.nn.BatchNorm2d):
        super().__init__()

        self.batchnorm_cls = batchnorm_cls

        assert issubclass(self.batchnorm_cls, torch.nn.modules.batchnorm._BatchNorm), type(self.batchnom_cls)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, batchnorm_cls=self.batchnorm_cls)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, batchnorm_cls=self.batchnorm_cls)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = apply_padding(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    """

    Source: Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
