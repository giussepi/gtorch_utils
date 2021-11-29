# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/iou """

from gtorch_utils.segmentation.metrics import IOU


class IOU_Loss(IOU):
    """ Source: https://github.com/giussepi/UNet-Version/blob/master/loss/iouLoss.py """

    def forward(self, pred, target):
        """ Return the IoU loss that is: 1 - IoU """
        return 1 - super().forward(pred, target)


def IOU_loss(pred, label):
    """ Source: https://github.com/giussepi/UNet-Version/blob/master/loss/iouLoss.py """
    iou_out = IOU_Loss()(pred, label)
    # print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out
