# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet3_plus/init_weights """

from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod
from torch.nn import init


def weights_init_normal(m):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/init_weights.py
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/init_weights.py
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/init_weights.py
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/init_weights.py
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type=UNet3InitMethod.NORMAL):
    """
    Source: https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/init_weights.py
    """
    UNet3InitMethod.validate(init_type)
    #print('initialization method [%s]' % init_type)
    if init_type == UNet3InitMethod.NORMAL:
        net.apply(weights_init_normal)
    elif init_type == UNet3InitMethod.XAVIER:
        net.apply(weights_init_xavier)
    elif init_type == UNet3InitMethod.KAIMING:
        net.apply(weights_init_kaiming)
    elif init_type == UNet3InitMethod.ORTHOGONAL:
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
