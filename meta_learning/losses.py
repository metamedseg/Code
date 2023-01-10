from torch import nn
from torch.nn import functional as F
import torch


def approximate_iou(outputs, masks, smooth=1):
    outputs = nn.Sigmoid()(outputs)
    outputs = outputs.squeeze(1)
    masks = masks.squeeze(1)
    intersection = (outputs * masks).sum()
    total_overlap = (outputs + masks).sum()
    union = total_overlap - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


class IoULoss(nn.Module):
    """IoU Loss implementation that follows http: // cs.umanitoba.ca / ~ywang / papers / isvc16.pdf"""

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, outputs, masks, smooth=1):
        iou = approximate_iou(outputs, masks, smooth)
        return 1 - iou.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss approximation. https://arxiv.org/pdf/1706.05721.pdf
    Larger beta places more emphasis on false negatives -> more importance to recall.
    """

    def __init__(self):
        super(TverskyLoss, self).__init__()

    def forward(self, outputs, masks, alpha=0.2, beta=0.8, smooth=1):
        outputs = nn.Sigmoid()(outputs)
        outputs = outputs.squeeze(1)
        masks = masks.squeeze(1)
        true_pos = (outputs * masks).sum()
        false_pos = ((1 - masks) * outputs).sum()
        fals_neg = (masks * (1 - outputs)).sum()
        tversky = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * fals_neg + smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combined loss that combines IoU and BCE loss"""

    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, outputs, masks, smooth=1, alpha=0.5):
        iou = approximate_iou(outputs, masks, smooth)
        iou_loos = 1 - iou
        bce_loss = nn.BCELoss()(outputs, masks)
        combined_loss = alpha * iou_loos + (1 - alpha) * bce_loss
        return combined_loss


class CombinedLoss2(nn.Module):
    def __init__(self, pos_weight):
        super(CombinedLoss2, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, outputs, masks, smooth=1):
        iou = approximate_iou(outputs, masks, smooth)
        modified_dice = (2 * iou) / (iou + 1)
        bce = F.binary_cross_entropy_with_logits(outputs, masks,
                                                 pos_weight=self.pos_weight)
        combined = bce - torch.log(modified_dice)
        return combined


LOSSES = {"bce": nn.BCELoss, "bce_weighted": nn.BCEWithLogitsLoss, "iou": IoULoss, "tversky": TverskyLoss,
          "combined": CombinedLoss, "combined2": CombinedLoss2}
