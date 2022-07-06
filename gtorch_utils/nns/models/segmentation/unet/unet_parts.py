# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet/unet_parts """

from typing import Union

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from gtorch_utils.utils.images import apply_padding


__all__ = ['DoubleConv', 'Down', 'Up', 'OutConv', 'UpConcat', 'UnetDsv']


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
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, batchnorm_cls=self.batchnorm_cls)
        else:
            # FIXME: These lines will not work. It must be fixed for the non-bilinear case.
            # A very similar case is fixed at
            # nns/models/layers/disagreement_attention/layers.py -> UnetDAUp
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


class UpConcat(torch.nn.Module):
    """
    Upsampling and concatenation layer for XAttentionUNet
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True,
                 batchnorm_cls: _BatchNorm = torch.nn.BatchNorm2d):
        super().__init__()

        assert isinstance(in_channels, int), type(in_channels)
        assert isinstance(out_channels, int), type(out_channels)
        assert isinstance(bilinear, bool), type(bilinear)
        assert issubclass(batchnorm_cls, _BatchNorm), type(batchnorm_cls)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.batchnorm_cls = batchnorm_cls

        # always using upsample (following original Attention Unet implementation)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_block = DoubleConv(in_channels+out_channels, out_channels, batchnorm_cls=self.batchnorm_cls)

        # up_out_channels = self.in_channels if self.bilinear else self.in_channels // 2
        # if self.bilinear:
        #     self.up = torch.nn.Sequential(
        #         torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #         # torch.nn.Conv2d(self.in_channels, up_out_channels, kernel_size=3, padding=1),
        #         # batchnorm_cls(up_out_channels),
        #         # torch.nn.LeakyReLU(inplace=True),
        #     )
        # else:
        #     self.up = torch.nn.Sequential(
        #         torch.nn.ConvTranspose2d(self.in_channels, up_out_channels, kernel_size=2, stride=2),
        #         # batchnorm_cls(up_out_channels),
        #         # torch.nn.LeakyReLU(inplace=True),
        #     )
        # self.conv_block = DoubleConv(2*up_out_channels, out_channels, batchnorm_cls=self.batchnorm_cls)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /):
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps
        Returns:
            Union[torch.Tensor, None]
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)

        decoder_x = self.up(x)
        decoder_x = torch.cat((skip_connection, decoder_x), dim=1)
        x = self.conv_block(decoder_x)

        return x


class UnetDsv(torch.nn.Module):
    """
    Deep supervision layer for UNet
    """

    def __init__(self, in_size: int, out_size: int, scale_factor: int):
        super().__init__()
        self.dsv = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
        )

    def forward(self, x):
        return self.dsv(x)


class UnetGridGatingSignal(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, kernel_size: Union[int, tuple] = 1,
                 batchnorm_cls: _BatchNorm = torch.nn.BatchNorm2d):
        super().__init__()

        assert isinstance(in_size, int), type(in_size)
        assert isinstance(out_size, int), type(out_size)
        assert isinstance(kernel_size, (int, tuple)), type(kernel_size)
        assert issubclass(batchnorm_cls, _BatchNorm), type(batchnorm_cls)

        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.batchnorm_cls = batchnorm_cls

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_size, self.out_size, self.kernel_size, stride=1, padding=0),
            self.batchnorm_cls(self.out_size),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outputs = self.conv1(x)

        return outputs
