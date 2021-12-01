# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/visualisation/plot_img_and_mask """

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_img_and_mask(img, mask):
    """
    Plots the image along with its mask

    Args:
        img  <torch.Tensor, np.array>: Image with shape <height, width, channels>
        mask <torch.Tensor, np.array>: Masks with shape <height, width> or <height, width, classes>

    Source: https://github.com/milesial/Pytorch-UNet/blob/master/utils/utils.py
    """
    assert isinstance(img, np.ndarray) or torch.is_tensor(img), type(img)
    assert isinstance(mask, np.ndarray) or torch.is_tensor(mask), type(img)

    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    _, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)

    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title('Output mask')
        ax[1].imshow(mask)

    plt.xticks([]), plt.yticks([])
    plt.show()
