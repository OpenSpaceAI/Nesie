import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_consistency_loss(end_points, ema_end_points, config):
    """
    Args:
        end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
                flip_x_axis, flip_y_axis, rot_mat
            }
        ema_end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
            }
    Returns:
        consistency_loss: pytorch scalar tensor
        end_points: dict
    """
    center_consistency_loss, map_ind = compute_center_consistency_loss(end_points, ema_end_points)
    class_consistency_loss = compute_class_consistency_loss(end_points, ema_end_points, map_ind)
    size_consistency_loss = compute_size_consistency_loss(end_points, ema_end_points, map_ind, config)

    consistency_loss =  center_consistency_loss +class_consistency_loss + size_consistency_loss

    end_points['center_consistency_loss'] = center_consistency_loss
    end_points['class_consistency_loss'] = class_consistency_loss
    end_points['size_consistency_loss'] = size_consistency_loss
    end_points['consistency_loss'] = consistency_loss

    return consistency_loss, end_points

def compute_center_consistency_loss(end_points, ema_end_points):
    center = end_points['center'] #(B, num_proposal, 3)
    ema_center = ema_end_points['center'] #(B, num_proposal, 3)
    flip_x_axis = end_points['flip_x_axis'] #(B,)
    flip_y_axis = end_points['flip_y_axis'] #(B,)
    rot_mat = end_points['rot_mat'] #(B,3,3)
    scale_ratio = end_points['scale'] #(B,1,3)

    # align ema_center with center based on the input augmentation steps
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    ema_center[inds_to_flip_x_axis, :, 0] = -ema_center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]

    ema_center = torch.bmm(ema_center, rot_mat.transpose(1,2)) #(B, num_proposal, 3)

    ema_center = ema_center * scale_ratio

    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #ind1 (B, num_proposal): ema_center index closest to center

    #TODO: use both dist1 and dist2 or only use dist1
    dist = dist1 + dist2
    return torch.mean(dist), ind2


def compute_class_consistency_loss(end_points, ema_end_points, map_ind):
    cls_scores = end_points['sem_cls_scores'] #(B, num_proposal, num_class)
    ema_cls_scores = ema_end_points['sem_cls_scores'] #(B, num_proposal, num_class)

    cls_log_prob = F.log_softmax(cls_scores, dim=2) #(B, num_proposal, num_class)
    # cls_log_prob = F.softmax(cls_scores, dim=2)
    ema_cls_prob = F.softmax(ema_cls_scores, dim=2) #(B, num_proposal, num_class)

    cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])

    class_consistency_loss = F.kl_div(cls_log_prob_aligned, ema_cls_prob)
    # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)

    return class_consistency_loss*2


def compute_size_consistency_loss(end_points, ema_end_points, map_ind, config):
    mean_size_arr = config.mean_size_arr
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda() #(num_size_cluster,3)
    B, K = map_ind.shape

    scale_ratio = end_points['scale'] #(B,1,3)
    size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    size_residual = torch.gather(end_points['size_residuals'], 2, size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    size_residual.squeeze_(2)

    ema_size_class = torch.argmax(ema_end_points['size_scores'], -1) # B,num_proposal
    ema_size_residual = torch.gather(ema_end_points['size_residuals'], 2, ema_size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    ema_size_residual.squeeze_(2)

    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(B,K,3)
    size = size_base + size_residual

    ema_size_base = torch.index_select(mean_size_arr, 0, ema_size_class.view(-1))
    ema_size_base = ema_size_base.view(B,K,3)
    ema_size = ema_size_base + ema_size_residual
    ema_size = ema_size * scale_ratio

    size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(size, map_ind)])

    size_consistency_loss = F.mse_loss(size_aligned, ema_size)

    return size_consistency_loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).expand(-1,-1,M,-1)
    pc2_expand_tile = pc2.unsqueeze(1).expand(-1,N,-1,-1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    return dist1, idx1, dist2, idx2

def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss