import mmcv
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.losses import MSELoss, SmoothL1Loss, CrossEntropyLoss

@LOSSES.register_module()
class SurfaceLoss(nn.Module):
    """Surface loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0, func_type='MSELoss'):
        super(SurfaceLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.func_type = func_type
        if self.func_type == 'MSELoss':
            self.loss_func = MSELoss(self.reduction, self.loss_weight)
        elif self.func_type == 'SmoothL1Loss':
            self.loss_func = SmoothL1Loss(self.beta, self.reduction, self.loss_weight)
        elif self.func_type == 'CrossEntropyLoss':
            self.loss_func = CrossEntropyLoss(reduction=self.reduction, loss_weight=self.loss_weight)

    def forward(self,
                pred,
                target,
                scale,
                center,
                prob,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        target = Bbox2Surface(target)
        if self.func_type == 'MSELoss':
            loss_surface = self.loss_func(
                pred, target, weight, avg_factor, reduction_override
            )
        elif self.func_type == 'SmoothL1Loss':
            loss_surface = self.loss_func(
                pred, target, weight, avg_factor, reduction_override, **kwargs
            )
        elif self.func_type == 'CrossEntropyLoss':
            target = target.reshape(-1, 6)
            target_t = TransformSurface(target, center, scale)
            left_prob, right_prob, left_weight, right_weight = Surface2Prob(target_t, prob)
            loss_left_surface = self.loss_func(
                cls_score = prob.transpose(2, 1), 
                label = left_prob.reshape(prob.shape).transpose(2, 1), 
                weight = weight,
                avg_factor = avg_factor, 
                reduction_override = 'none', 
                **kwargs
            )
            loss_right_surface = self.loss_func(
                cls_score = prob.transpose(2, 1), 
                label = right_prob.reshape(prob.shape).transpose(2, 1), 
                weight = weight,
                avg_factor = avg_factor, 
                reduction_override = 'none', 
                **kwargs
            )
            loss_surface = loss_left_surface * left_weight + loss_right_surface * right_weight
            loss_surface = loss_surface.sum()
        return loss_surface

def Bbox2Surface(bbox2):
    center = bbox2[...,:3]
    size = bbox2[...,3:6]
    surface = torch.zeros_like(bbox2[...,:6])
    surface[..., 0] = center[..., 0] - 0.5 * size[..., 0]
    surface[..., 1] = center[..., 1] - 0.5 * size[..., 1]
    surface[..., 2] = center[..., 2] - 0.5 * size[..., 2]
    surface[..., 3] = center[..., 0] + 0.5 * size[..., 0]
    surface[..., 4] = center[..., 1] + 0.5 * size[..., 1]
    surface[..., 5] = center[..., 2] + 0.5 * size[..., 2]
    return surface

def TransformSurface(surface, center, scale):
    surface_t = torch.zeros_like(surface)
    surface_t[..., 0] = -surface[..., 0] + center[..., 0]
    surface_t[..., 1] = -surface[..., 1] + center[..., 1]
    surface_t[..., 2] = -surface[..., 2] + center[..., 2]
    surface_t[..., 3] = surface[..., 3] - center[..., 0]
    surface_t[..., 4] = surface[..., 4] - center[..., 1]
    surface_t[..., 5] = surface[..., 5] - center[..., 2]
    surface_t = surface_t/scale
    return surface_t

def Surface2Prob(target, prob):
    reg_max = prob.shape[-1] - 1
    left_num = torch.div(target, (1 / reg_max), rounding_mode='floor')
    right_num = left_num + 1
    left_weight = (1 / reg_max - target % (1 / reg_max)) / (1 / reg_max)
    right_weight = target % (1 / reg_max) / (1 / reg_max)
    mask_left = left_num < 0
    mask_right = right_num > reg_max
    left_num[mask_left] = 0.0
    right_num[mask_left] = 1.0
    left_num[mask_right] = 0.0
    right_num[mask_right] = 1.0
    left_weight[mask_left] = 1.0
    right_weight[mask_left] = 0.0
    right_weight[mask_right] = 1.0
    left_weight[mask_right] = 0.0
    left_prob = torch.zeros_like(prob).reshape(-1, prob.shape[-1])
    right_prob = torch.zeros_like(prob).reshape(-1, prob.shape[-1])
    left_prob = left_prob.scatter_(1, left_num.reshape(-1, 1).to(int), 1)
    right_prob = right_prob.scatter_(1, right_num.reshape(-1, 1).to(int), 1)
    return left_prob, right_prob, left_weight, right_weight