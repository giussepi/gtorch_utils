# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet/unet_parts """

from collections import OrderedDict
from typing import Union, Optional

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from gtorch_utils.utils.images import apply_padding


__all__ = [
    'DoubleConv', 'XConv',  'Down', 'MicroAE', 'TinyAE', 'TinyUpAE', 'MicroUpAE', 'AEDown', 'AEDown2', 'Up',
    'OutConv', 'UpConcat', 'AEUpConcat', 'AEUpConcat2', 'UnetDsv', 'UnetGridGatingSignal'
]


class DoubleConv(torch.nn.Module):
    """
    (convolution => [BN] => ReLU) * 2

    Inspired on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(
            self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None,
            batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>:
            out_channels     <int>:
            mid_channels     <int>:
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        if self.mid_channels is not None:
            assert isinstance(self.mid_channels, int), type(self.mid_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        if not self.mid_channels:
            self.mid_channels = self.out_channels
        self.double_conv = torch.nn.Sequential(
            convxd(self.in_channels, self.mid_channels, kernel_size=3, padding=1),
            self.batchnorm_cls(self.mid_channels),
            torch.nn.ReLU(inplace=True),
            convxd(self.mid_channels, self.out_channels, kernel_size=3, padding=1),
            self.batchnorm_cls(self.out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


class XConv(torch.nn.Module):
    """
    (convolution => [BN] => ReLU) * X
    """

    def __init__(
            self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None,
            batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2,
            conv_layers: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>:
            out_channels     <int>:
            mid_channels     <int>:
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
            conv_layers      <int>: Number of convolutional layers to stack. Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions
        self.conv_layers = conv_layers

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        if self.mid_channels is not None:
            assert isinstance(self.mid_channels, int), type(self.mid_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert isinstance(self.conv_layers, int), type(self.conv_layers)
        assert self.conv_layers > 0, self.conv_layers

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        if not self.mid_channels:
            self.mid_channels = self.out_channels

        layers = OrderedDict()

        if self.conv_layers == 1:
            layers['0'] = convxd(self.in_channels, self.out_channels, kernel_size=3, padding=1)
            layers['1'] = self.batchnorm_cls(self.out_channels)
            layers['2'] = torch.nn.ReLU(inplace=True)

        else:
            for i in range(0, self.conv_layers*3, 3):
                if i == 0:
                    layers[str(i)] = convxd(self.in_channels, self.mid_channels, kernel_size=3, padding=1)
                    layers[str(i+1)] = self.batchnorm_cls(self.mid_channels)
                    layers[str(i+2)] = torch.nn.ReLU(inplace=True)
                elif i + 3 < self.conv_layers*3:
                    layers[str(i)] = convxd(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)
                    layers[str(i+1)] = self.batchnorm_cls(self.mid_channels)
                    layers[str(i+2)] = torch.nn.ReLU(inplace=True)
                else:
                    layers[str(i)] = convxd(self.mid_channels, self.out_channels, kernel_size=3, padding=1)
                    layers[str(i+1)] = self.batchnorm_cls(self.out_channels)
                    layers[str(i+2)] = torch.nn.ReLU(inplace=True)

        self.x_conv = torch.nn.Sequential(layers)

    def forward(self, x: torch.Tensor):
        return self.x_conv(x)


class Down(torch.nn.Module):
    """
    Downscaling with maxpool then double conv

    Inspired on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(
            self, in_channels: int, out_channels: int, batchnorm_cls: Optional[_BatchNorm] = None,
            data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>:
            out_channels     <int>:
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d

        self.maxpool_conv = torch.nn.Sequential(
            maxpoolxd(2),
            DoubleConv(
                self.in_channels, self.out_channels, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )
        )

    def forward(self, x: torch.Tensor):
        return self.maxpool_conv(x)


class MicroAE(torch.nn.Module):
    """
    Micro AE following XAttentionUnet layers
    """

    def __init__(self, in_channels: int, out_channels: int, batchnorm_cls: Optional[_BatchNorm] = None,
                 data_dimensions: int = 2):
        """
        Kwargs:
            in_channels      <int>: encoder's in channels (decoder's out channels)
            out_channels     <int>: encoder's out channels (decoder's in channels)
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d
        self.encoder = torch.nn.Sequential(
            maxpoolxd(2),
            DoubleConv(
                self.in_channels, self.out_channels, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )
        )
        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'
        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
            DoubleConv(
                self.out_channels, self.in_channels, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class TinyAE(torch.nn.Module):
    """
    Tiny Convolutional AutoEncoder
    """

    def __init__(self, in_channels: int, batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2):
        """
        Kwargs:
            in_channels      <int>: encoder's in channels (decoder's out channels)
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d
        convtranspose = torch.nn.ConvTranspose2d if self.data_dimensions == 2 else torch.nn.ConvTranspose3d

        self.encoder = torch.nn.Sequential(
            convxd(self.in_channels, 2 * self.in_channels, kernel_size=3, stride=2, padding=1),
            self.batchnorm_cls(2 * self.in_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = convtranspose(
            2 * self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class TinyUpAE(torch.nn.Module):
    """
    Tiny upsampling Convolutional AutoEncoder
    """

    def __init__(self, in_channels: int, batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2):
        """
        Kwargs:
            in_channels      <int>: encoder's in channels (decoder's out channels)
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d
        convtranspose = torch.nn.ConvTranspose2d if self.data_dimensions == 2 else torch.nn.ConvTranspose3d

        self.encoder = torch.nn.Sequential(
            convtranspose(
                self.in_channels, self.in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.batchnorm_cls(self.in_channels // 2),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = convxd(self.in_channels // 2, self.in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class MicroUpAE(torch.nn.Module):
    """
    Micro Up AE following XAttentionUnet layers
    """

    def __init__(self, in_channels: int, out_channels: int, batchnorm_cls: Optional[_BatchNorm] = None,
                 data_dimensions: int = 2):
        """
        Kwargs:
            in_channels      <int>: encoder's in channels (decoder's out channels)
            out_channels     <int>: encoder's out channels (decoder's in channels)
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'
        self.encoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
            DoubleConv(
                self.in_channels, self.out_channels, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )
        )

        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d
        self.decoder = torch.nn.Sequential(
            maxpoolxd(2),
            DoubleConv(
                self.out_channels, self.in_channels, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class AEDown(torch.nn.Module):
    """
    Downscaling with maxpool then double conv using a micro AE
    """

    def __init__(
            self, in_channels: int, out_channels: int, batchnorm_cls: Optional[_BatchNorm] = None,
            data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>:
            out_channels     <int>:
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.ae = MicroAE(in_channels, out_channels, batchnorm_cls, data_dimensions)

    def forward(self, x: torch.Tensor):
        downsampled, downupsampled = self.ae(x)

        return downsampled, downupsampled


class AEDown2(torch.nn.Module):
    """
    Downscaling with maxpool then double conv using an isolated Tiny AE
    """

    def __init__(
            self, in_channels: int, out_channels: int, batchnorm_cls: Optional[_BatchNorm] = None,
            data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>:
            out_channels     <int>:
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        maxpoolxd = torch.nn.MaxPool2d if self.data_dimensions == 2 else torch.nn.MaxPool3d
        self.ae = TinyAE(self.in_channels, self.batchnorm_cls, self.data_dimensions)
        self.down = maxpoolxd(2)
        self.out = DoubleConv(
            3 * self.in_channels, self.out_channels, batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions
        )

    def forward(self, x: torch.Tensor):
        encoded, decoded = self.ae(x.detach())
        enriched_downsample = self.out(torch.cat([encoded.detach(), self.down(x)], dim=1))

        return enriched_downsample, decoded


class Up(torch.nn.Module):
    """
    Upscaling then double conv

    Inspired on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(
            self, in_channels: int, out_channels: int, bilinear: bool = True, batchnorm_cls: Optional[_BatchNorm] = None,
            data_dimensions: int = 2):
        """
        Kwargs:
            in_channels      <int>:
            out_channels     <int>:
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            bilinear        <bool>: If true upsample is used, else convtranspose.
                                    Default True
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_channels, int), type(self.in_channels)
        assert isinstance(self.out_channels, int), type(self.out_channels)
        assert isinstance(self.bilinear, bool), type(self.bilinear)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnom_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'
            self.up = torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, batchnorm_cls=self.batchnorm_cls,
                data_dimensions=self.data_dimensions
            )
        else:
            # FIXME: These lines will not work. It must be fixed for the non-bilinear case.
            # A very similar case is fixed at
            # nns/models/layers/disagreement_attention/layers.py -> UnetDAUp
            convtranspose = torch.nn.ConvTranspose2d if self.data_dimensions == 2 else torch.nn.ConvTranspose3d
            self.up = convtranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, out_channels, batchnorm_cls=self.batchnorm_cls, data_dimensions=self.data_dimensions)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = apply_padding(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    """

    Inspired on: Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, in_channels: int, out_channels: int, data_dimensions: int = 2):
        """
        Kwargs:
            in_channels      <int>: in channels
            out_channels     <int>: out channels
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_dimensions = data_dimensions

        assert isinstance(in_channels, int), type(in_channels)
        assert isinstance(out_channels, int), type(out_channels)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convx = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d
        self.conv = convx(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.conv(x)


class UpConcat(torch.nn.Module):
    """
    Upsampling and concatenation layer for XAttentionUNet
    """

    def __init__(
            self, in_channels: int, out_channels: int, bilinear: bool = True,
            batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>: in channels
            out_channels     <int>: out channels
            bilinear        <bool>: [DEPRECATED] If true upsample is used, else convtranspose.
                                    Currently upsample is always employed.
                                    Default True
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(in_channels, int), type(in_channels)
        assert isinstance(out_channels, int), type(out_channels)
        assert isinstance(bilinear, bool), type(bilinear)
        assert issubclass(batchnorm_cls, _BatchNorm), type(batchnorm_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        # always using upsample (following original Attention Unet implementation)
        self.up = torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.conv_block = DoubleConv(
            in_channels+out_channels, out_channels, batchnorm_cls=self.batchnorm_cls, data_dimensions=self.data_dimensions)

        # up_out_channels = self.in_channels if self.bilinear else self.in_channels // 2
        # convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d
        # convtransposexd = torch.nn.ConvTranspose2d if self.data_dimensions == 2 else torch.nn.ConvTranspose3d
        # if self.bilinear:
        #     self.up = torch.nn.Sequential(
        #         torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
        #         # convxd(self.in_channels, up_out_channels, kernel_size=3, padding=1),
        #         # batchnorm_cls(up_out_channels),
        #         # torch.nn.LeakyReLU(inplace=True),
        #     )
        # else:
        #     self.up = torch.nn.Sequential(
        #         convtransposexd(self.in_channels, up_out_channels, kernel_size=2, stride=2),
        #         # batchnorm_cls(up_out_channels),
        #         # torch.nn.LeakyReLU(inplace=True),
        #     )
        # self.conv_block = DoubleConv(
        #     2*up_out_channels, out_channels, batchnorm_cls=self.batchnorm_cls, data_dimensions=self.data_dimensions)

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


class AEUpConcat(torch.nn.Module):
    """
    Upsampling and concatenation layer for XAttentionUNet using a micro autoencoder
    """

    def __init__(
            self, in_channels: int, out_channels: int, bilinear: bool = True,
            batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>: in channels
            out_channels     <int>: out channels
            bilinear        <bool>: [DEPRECATED] If true upsample is used, else convtranspose.
                                    Currently upsample is always employed.
                                    Default True
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()

        self.down = Down(out_channels, out_channels,
                         batchnorm_cls=batchnorm_cls, data_dimensions=data_dimensions)
        self.ae = MicroUpAE(in_channels+out_channels, out_channels, batchnorm_cls, data_dimensions)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /):
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps
        Returns:
            decoder_x_upsampled <torch.Tensor>, ae_output <torch.Tensor>, ae_input <torch.Tensor>
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)

        decoder_x = torch.cat((self.down(skip_connection), x), dim=1)
        decoder_x_upsampled, decoder_x_updownsampled = self.ae(decoder_x)

        return decoder_x_upsampled, decoder_x_updownsampled, decoder_x


class AEUpConcat2(torch.nn.Module):
    """
    Upsampling and concatenation layer for XAttentionUNet
    """

    def __init__(
            self, in_channels: int, out_channels: int, bilinear: bool = True,
            batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_channels      <int>: in channels
            out_channels     <int>: out channels
            bilinear        <bool>: [DEPRECATED] If true upsample is used, else convtranspose.
                                    Currently upsample is always employed.
                                    Default True
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(in_channels, int), type(in_channels)
        assert isinstance(out_channels, int), type(out_channels)
        assert isinstance(bilinear, bool), type(bilinear)
        assert issubclass(batchnorm_cls, _BatchNorm), type(batchnorm_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        self.up = torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.up_ae = TinyUpAE(in_channels, self.batchnorm_cls, self.data_dimensions)
        self.conv_block = DoubleConv(
            in_channels+out_channels+in_channels//2, out_channels, batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /):
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps
        Returns:
            decoder_x <torch.Tensor>, decoded <torch.Tensor>
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)

        encoded, decoded = self.up_ae(x.detach())
        decoder_x = torch.cat((skip_connection, encoded.detach(), self.up(x)), dim=1)
        decoder_x = self.conv_block(decoder_x)

        return decoder_x, decoded


class UnetDsv(torch.nn.Module):
    """
    Deep supervision layer for UNet
    """

    def __init__(self, in_size: int, out_size: int, scale_factor: int, data_dimensions: int = 2):
        """
        Kwargs:
            in_size         <int>: convolutional in channels
            out_size        <int>: convolutional out channels
            scale_factor    <int>: upsample scale factor
            data_dimensions <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                       3 for 3D [batch, channel, depth, height, width]. This argument will
                                       determine to use conv2d or conv3d.
                                       Default 2
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.scale_factor = scale_factor
        self.data_dimensions = data_dimensions

        assert isinstance(self.in_size, int), type(self.in_size)
        assert isinstance(self.out_size, int), type(self.out_size)
        assert isinstance(self.scale_factor, int), type(self.scale_factor)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d
        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        self.dsv = torch.nn.Sequential(
            convxd(self.in_size, self.out_size, kernel_size=1, stride=1, padding=0),
            torch.nn.Upsample(scale_factor=self.scale_factor, mode=mode, align_corners=False),
        )

    def forward(self, x: torch.Tensor):
        return self.dsv(x)


class UnetGridGatingSignal(torch.nn.Module):
    def __init__(
            self, in_size: int, out_size: int, kernel_size: Union[int, tuple] = 1,
            batchnorm_cls: Optional[_BatchNorm] = None, data_dimensions: int = 2
    ):
        """
        Kwargs:
            in_size          <int>:
            out_size         <int>:
            kernel_size <int, tuple>:
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                    torch.nn.BatchNorm3d
            data_dimensions  <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                    3 for 3D [batch, channel, depth, height, width]. This argument will
                                    determine to use conv2d or conv3d.
                                    Default 2
        """
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(self.in_size, int), type(self.in_size)
        assert isinstance(self.out_size, int), type(self.out_size)
        assert isinstance(self.kernel_size, (int, tuple)), type(self.kernel_size)
        assert issubclass(self.batchnorm_cls, _BatchNorm), type(self.batchnorm_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'

        convxd = torch.nn.Conv2d if self.data_dimensions == 2 else torch.nn.Conv3d

        self.conv1 = torch.nn.Sequential(
            convxd(self.in_size, self.out_size, self.kernel_size, stride=1, padding=0),
            self.batchnorm_cls(self.out_size),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        outputs = self.conv1(x)

        return outputs
