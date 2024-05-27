import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from torch.nn import functional as F
import numpy as np

from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.builder import build_loss
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.models.model_utils import VoteModule
from mmdet3d.ops import build_sa_module, furthest_point_sample
from mmdet3d.ops.rotated_iou import cal_giou_3d, cal_iou_3d
from mmdet.core import multi_apply
from mmdet.models import HEADS
from .reliable_conv_bbox_module import ReliableConvBboxHead
from .quelity_estimation_module import QualityEstimation
from mmdet3d.core import DepthInstance3DBoxes

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1)/self.reg_max)

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 6*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 6).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 6)
        return x

class AngleIntegral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(AngleIntegral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1)/self.reg_max)

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, (n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 1).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 1)
        return x

@HEADS.register_module()
class SAQEHead(BaseModule):
    r"""Bbox head of `Votenet <https://arxiv.org/abs/1904.09664>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_class_loss (dict): Config of size classification loss.
        size_res_loss (dict): Config of size residual regression loss.
        semantic_loss (dict): Config of point-wise semantic segmentation loss.
    """

    def __init__(self,
                 num_classes,
                 reg_max=16,
                 reg_channels=128,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 alpha=0.5,
                 objectness_loss=None,
                 center_loss=None,
                 semantic_loss=None,
                 iou_loss=None,
                 iou_pred_loss=None,
                 surface_loss=None,
                 angle_loss=None,
                 angle_pred_loss=None,
                 side_loss=None,
                 init_cfg=None,
                 grid_conv_cfg=None,
                 sizes=[3.0,3.0,2.5]):
        super(SAQEHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.reg_channels = reg_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gt_per_seed = vote_module_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']
        self.alpha = alpha
        self.sizes=sizes

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.iou_loss = build_loss(iou_loss)
        self.iou_pred_loss = build_loss(iou_pred_loss)
        self.surface_loss = build_loss(surface_loss)
        self.angle_loss = build_loss(angle_loss)
        self.angle_pred_loss = build_loss(angle_pred_loss)
        self.side_loss = build_loss(side_loss)
        if semantic_loss is not None:
            self.semantic_loss = build_loss(semantic_loss)

        self.vote_module = VoteModule(**vote_module_cfg)
        self.vote_aggregation = build_sa_module(vote_aggregation_cfg)
        self.fp16_enabled = False

        # Bbox classification and regression
        self.n_reg_outs = 6 * (self.reg_max + 1)
        self.head_reg_outs = 12
        self.conv_pred = ReliableConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self.num_classes + 2,
            num_bbox_out_channels=self.n_reg_outs + 3,
            num_heading_out_channels=self.head_reg_outs,
            reg_max=self.reg_max)
        self.integral = Integral(self.reg_max)
        self.angle_integral = AngleIntegral(self.head_reg_outs - 1)
        
        # quality estimation module
        self.grid_conv = QualityEstimation(**grid_conv_cfg)

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.
        Args:
            feat_dict (dict): Feature dict from backbone.
        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        seed_points = feat_dict['fp_xyz'][-1]
        seed_features = feat_dict['fp_features'][-1]
        seed_indices = feat_dict['fp_indices'][-1]
        return seed_points, seed_features, seed_indices

    def side2box(self, aggregated_points, bbox_pred, results):
        B, proposal_num = bbox_pred.shape[:2]
        surface_pred_res = self.integral(bbox_pred[..., :self.n_reg_outs]).reshape(B, proposal_num, -1)
        scale_x = torch.exp(bbox_pred[..., self.n_reg_outs + 0])
        scale_y = torch.exp(bbox_pred[..., self.n_reg_outs + 1])
        scale_z = torch.exp(bbox_pred[..., self.n_reg_outs + 2])
        x1 = aggregated_points[..., 0] - surface_pred_res[..., 0] * scale_x     
        y1 = aggregated_points[..., 1] - surface_pred_res[..., 1] * scale_y
        z1 = aggregated_points[..., 2] - surface_pred_res[..., 2] * scale_z
        x2 = aggregated_points[..., 0] + surface_pred_res[..., 3] * scale_x
        y2 = aggregated_points[..., 1] + surface_pred_res[..., 4] * scale_y
        z2 = aggregated_points[..., 2] + surface_pred_res[..., 5] * scale_z
        results['surface_pred'] = torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)
        results['surface_scale'] = torch.stack((scale_x, scale_y, scale_z, scale_x, scale_y, scale_z), dim=-1)

        angles = self.angle_integral(bbox_pred[..., self.n_reg_outs + 3:]).reshape(B, proposal_num) * 2 * torch.pi
        angles[angles > torch.pi] -= 2 * torch.pi

        results['bbox_preds'] = torch.stack((
            (x1 + x2)/2.0,
            (y1 + y2)/2.0,
            (z1 + z2)/2.0,
            x2 - x1,
            y2 - y1,
            z2 - z1,
            angles
        ), dim=-1)
        return results

    def jitter_bbox_preds(self, results, dataset_name):
        center, size, heading = results['bbox_preds'][..., :3], results['bbox_preds'][..., 3:6], results['bbox_preds'][..., -1]
        B = heading.shape[0]
        factor = 1
        center_jitter = center.unsqueeze(2).expand(-1, -1, factor, -1).contiguous().view(B, -1, 3)
        size_jitter = size.unsqueeze(2).expand(-1, -1, factor, -1).contiguous().view(B, -1, 3)
        heading_jitter = heading.unsqueeze(2).expand(-1, -1, factor).contiguous().view(B, -1)
        center_jitter = center_jitter + size_jitter * (torch.randn(size_jitter.shape).cuda() * 0.5)
        size_jitter = size_jitter + size_jitter * (torch.randn(size_jitter.shape).cuda() * 0.5 + 0.2)
        size_jitter = torch.clamp(size_jitter, min=1e-8)

        center = torch.cat([center, center_jitter], dim=1)
        size = torch.cat([size, size_jitter], dim=1)

        if dataset_name == "ScanNet":
            heading = torch.cat([heading, heading_jitter], dim=1)
            heading_ = torch.zeros_like(heading)
        else:
            heading_ = torch.cat([heading, heading_jitter], dim=1)
        
        jitter_center = center_jitter
        jitter_size = size_jitter
        jitter_heading = heading_jitter
        results['jitter_bbox_preds'] = torch.stack((
            jitter_center[..., 0], 
            jitter_center[..., 1], 
            jitter_center[..., 2], 
            jitter_size[..., 0], 
            jitter_size[..., 1], 
            jitter_size[..., 2], 
            jitter_heading), dim=-1)

        results['jitter_surface_preds'] = Bbox2Surface(results['jitter_bbox_preds'])
        return center, size, heading_, results

    def forward(self, feat_dict, sample_mod, dataset_name = 'ScanNet'):
        assert sample_mod in ['vote', 'seed', 'random', 'spec']
        seed_points, seed_features, seed_indices = self._extract_input(feat_dict)

        # 1. generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(seed_points, seed_features)
        results = dict(seed_points=seed_points,
            seed_features=seed_features,
            seed_indices=seed_indices,
            vote_points=vote_points,
            vote_features=vote_features,
            vote_offset=vote_offset)

        # 2. aggregate vote_points
        if sample_mod == 'vote':
            # use fps in vote_aggregation
            aggregation_inputs = dict(points_xyz=vote_points, features=vote_features)
        elif sample_mod == 'seed':
            # FPS on seed and choose the votes corresponding to the seeds
            sample_indices = furthest_point_sample(seed_points, self.num_proposal)
            aggregation_inputs = dict(points_xyz=vote_points, features=vote_features, indices=sample_indices)
        elif sample_mod == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = seed_points.new_tensor(torch.randint(0, num_seed, (batch_size, self.num_proposal)), dtype=torch.int32)
            aggregation_inputs = dict(points_xyz=vote_points, features=vote_features, indices=sample_indices)
        elif sample_mod == 'spec':
            # Specify the new center in vote_aggregation
            aggregation_inputs = dict(points_xyz=seed_points, features=seed_features, target_xyz=vote_points)
        else:
            raise NotImplementedError(f'Sample mode {sample_mod} is not supported!')

        aggregated_points, features, aggregated_indices = self.vote_aggregation(**aggregation_inputs)
        results['aggregated_points'] = aggregated_points
        results['aggregated_features'] = features
        results['aggregated_indices'] = aggregated_indices

        # 3. predict bbox and score
        cls_predictions, reg_predictions = self.conv_pred(features)
        origin_proposal_num = cls_predictions.shape[-1]

        # 4. decode predictions
        cls_preds_trans = cls_predictions.transpose(2, 1)
        results['obj_scores'] = cls_preds_trans[..., :2]
        results['sem_scores'] = cls_preds_trans[..., 2:]
        results = self.side2box(aggregated_points, reg_predictions.transpose(2, 1), results)
        B = reg_predictions.shape[0]
        probs = reg_predictions[:, :self.n_reg_outs, :]
        prob = F.softmax(probs.reshape(B, 6, self.reg_max+1, -1), dim=2)
        results['bbox_probs'] = prob

        # 5. quality score
        center, size, heading, results = self.jitter_bbox_preds(results, dataset_name)
        results = self.grid_conv(center.detach(), size.detach(), heading.detach(), results)

        # rotation prediction
        results['rotate_scores'] = results['rotate_scores'].sigmoid()
        results['rotate_scores_jitter'] = results['rotate_scores'][:, origin_proposal_num:]
        results['rotate_scores'] = results['rotate_scores'][:, :origin_proposal_num]

        # obj prediction
        results['R_obj_scores_jitter'] = results['R_obj_scores'][:, origin_proposal_num:]
        results['R_obj_scores'] = results['R_obj_scores'][:, :origin_proposal_num]

        # iou prediction
        results['iou_scores'] = results['iou_scores'].sigmoid()
        results['iou_scores_jitter'] = results['iou_scores'][:, origin_proposal_num:]
        results['iou_scores'] = results['iou_scores'][:, :origin_proposal_num]

        # side prediction
        results['side_scores'] = results['side_scores'].sigmoid().permute(1,3,0,2)
        results['side_scores_jitter'] = results['side_scores'][:, origin_proposal_num:]
        results['side_scores'] = results['side_scores'][:, :origin_proposal_num]
        return results

    @force_fp32(apply_to=('bbox_preds', ))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of vote head.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            ret_target (Bool): Return targets or not.

        Returns:
            dict: Losses of Votenet.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (vote_targets, vote_target_masks, center_targets, bbox_targets, mask_targets, valid_gt_masks,
         objectness_targets, objectness_weights, box_loss_weights,
         valid_gt_weights, assignment) = targets

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds['seed_points'],
                                              bbox_preds['vote_points'],
                                              bbox_preds['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate objectness loss
        objectness_loss_1 = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)
        
        objectness_loss_2 = self.objectness_loss(
            bbox_preds['R_obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        objectness_loss_3 = self.objectness_loss(
            bbox_preds['R_obj_scores_jitter'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        objectness_loss = objectness_loss_1 + (objectness_loss_2 + objectness_loss_3) * 0.5

        # calculate center loss
        source2target_loss, target2source_loss = self.center_loss(
            bbox_preds['bbox_preds'][..., :3],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        center_loss = source2target_loss + target2source_loss

        # calculate surface loss
        surface_weight = box_loss_weights.reshape(-1).unsqueeze(-1).repeat(1,6)
        surface_loss = self.surface_loss(
            bbox_preds['surface_pred'].reshape(-1, 6),
            torch.cat(bbox_targets, dim=0),
            bbox_preds['surface_scale'].reshape(-1, 6),
            bbox_preds['aggregated_points'].reshape(-1, 3),
            bbox_preds['bbox_probs'].permute(0,3,1,2).reshape(-1, 6, self.reg_max+1),
            weight=surface_weight,
            reduction_override='none',
        )
        # N_class = bbox_preds['side_scores'].shape[-1]
        # side_pred = bbox_preds['side_scores'].reshape(-1, 6, N_class)
        # side_scores = torch.stack([side_pred[i,:,indx[i]] for i in range(indx.shape[0])]).reshape(-1, 6)
        # sigma = 0.8 * side_scores * side_scores - 1.8 * side_scores + torch.ones_like(side_scores)
        # surface_loss = torch.exp(-sigma) * surface_loss + self.alpha * sigma * surface_weight
        surface_loss = surface_loss.sum()

        # calculate angle loss
        pred_angle = bbox_preds['bbox_preds'][..., -1].reshape(-1)
        target_angle = torch.cat(bbox_targets, dim=0)[..., -1]
        pred_angle_sin = torch.sin(pred_angle)
        pred_angle_cos = torch.cos(pred_angle)
        target_angle_sin = torch.sin(target_angle)
        target_angle_cos = torch.cos(target_angle)
        angle_sin_loss = self.angle_loss(pred_angle_sin, target_angle_sin, weight=box_loss_weights.reshape(-1), reduction_override='none')
        angle_cos_loss = self.angle_loss(pred_angle_cos, target_angle_cos, weight=box_loss_weights.reshape(-1), reduction_override='none')
        angle_loss = angle_sin_loss + angle_cos_loss
        angle_loss = angle_loss.sum()

        angle_score_labels = (angle_sin_loss + angle_cos_loss).detach() / ( box_loss_weights.max())
        pred_angle = bbox_preds['bbox_preds'][..., -1].reshape(-1)
        N_class = bbox_preds['rotate_scores'].shape[-1]
        indx = bbox_preds['sem_scores'].max(dim=-1)[1].reshape(-1)
        angle_pred_scores = bbox_preds['rotate_scores'].reshape(-1, N_class)
        angle_scores = torch.stack([angle_pred_scores[i,indx[i]] for i in range(indx.shape[0])]).reshape(-1)
        angle_pred_loss = self.angle_pred_loss(angle_scores, angle_score_labels, weight=box_loss_weights.reshape(-1))

        jitter_angle_pred_scores = bbox_preds['rotate_scores_jitter'].reshape(-1, N_class)
        jitter_angle_scores = torch.stack([jitter_angle_pred_scores[i,indx[i]] for i in range(indx.shape[0])]).reshape(-1)

        angle_pred_loss += self.angle_pred_loss(jitter_angle_scores, angle_score_labels, weight=box_loss_weights.reshape(-1))

        # angle_sigma = 0.8 * angle_scores * angle_scores - 1.8 * angle_scores + torch.ones_like(angle_scores)
        # angle_loss = torch.exp(-angle_sigma) * angle_loss + self.alpha * angle_sigma * box_loss_weights.reshape(-1)

        # calculate semantic loss
        semantic_loss = self.semantic_loss(
            bbox_preds['sem_scores'].transpose(2, 1),
            mask_targets,
            weight=box_loss_weights)

        # calculate iou loss
        iou_weight = box_loss_weights.reshape(-1)
        iou_loss = self.iou_loss(
            bbox_preds['bbox_preds'].reshape(-1, 7),
            torch.cat(bbox_targets, dim=0),
            weight=iou_weight,
            reduction_override='none',
        ).reshape(-1)
        # sigma_mean = sigma.mean(dim=-1)
        # iou_loss = torch.exp(-sigma_mean) * iou_loss + self.alpha * sigma_mean * iou_weight
        iou_loss = iou_loss.sum()

        # calculate iou pred loss
        targets = torch.cat(bbox_targets, dim=0)
        targets = targets.view_as(bbox_preds['bbox_preds'])
        label_iou = cal_iou_3d(bbox_preds['bbox_preds'], targets).detach().view(-1)
        label_iou_jitter = cal_iou_3d(bbox_preds['jitter_bbox_preds'], targets).detach().view(-1)
        label_cls = mask_targets.reshape(-1)
        label_cls_jitter = mask_targets.reshape(-1)

        loss_iou = self.iou_pred_loss(
            bbox_preds['iou_scores'].reshape(-1, self.num_classes), (label_cls, label_iou),
            weight=box_loss_weights.reshape(-1))
        loss_iou_jitter = self.iou_pred_loss(
            bbox_preds['iou_scores_jitter'].reshape(-1, self.num_classes), (label_cls_jitter, label_iou_jitter),
            weight=box_loss_weights.reshape(-1))
        iou_pred_loss = loss_iou + loss_iou_jitter

        # calculate side pred loss
        side_pred = bbox_preds['side_scores'].reshape(-1, 6, self.num_classes)
        side_pred = torch.stack([side_pred[i,:,label_cls[i]] for i in range(label_cls.shape[0])])
        bbox_side_gt = torch.cat(bbox_targets, dim=0)
        side_loss = self.side_loss(
            side_pred,
            bbox_preds['surface_pred'].reshape(-1, 6).detach(),
            bbox_side_gt,
            bbox_preds['surface_scale'].reshape(-1, 6),
            bbox_preds['aggregated_points'].reshape(-1, 3),
            bbox_preds['bbox_probs'].permute(0,3,1,2).reshape(-1, 6, self.reg_max+1),
            weight=surface_weight,
        )

        side_pred_jitter = bbox_preds['side_scores_jitter'].reshape(-1, 6, self.num_classes)
        side_pred_jitter = torch.stack([side_pred_jitter[i,:,label_cls_jitter[i]] for i in range(label_cls_jitter.shape[0])])
        bbox_side_gt = torch.cat(bbox_targets, dim=0)
        side_jitter_loss = self.side_loss(
            side_pred_jitter,
            bbox_preds['jitter_surface_preds'].reshape(-1, 6).detach(),
            bbox_side_gt,
            bbox_preds['surface_scale'].reshape(-1, 6),
            bbox_preds['aggregated_points'].reshape(-1, 3),
            bbox_preds['bbox_probs'].permute(0,3,1,2).reshape(-1, 6, self.reg_max+1),
            weight=surface_weight,
        )
        side_loss = side_loss + side_jitter_loss
        
        # total loss
        losses = dict(
            vote_loss=vote_loss,
            objectness_loss=objectness_loss,
            semantic_loss=semantic_loss,
            center_loss=center_loss,
            surface_loss=surface_loss,
            angle_loss=angle_loss,
            angle_pred_loss=angle_pred_loss,
            iou_loss=iou_loss,
            iou_pred_loss=iou_pred_loss,
            side_loss=side_loss)

        if ret_target:
            losses['targets'] = targets

        return losses

    @force_fp32(apply_to=('bbox_preds', ))
    def sup_loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of vote head.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            ret_target (Bool): Return targets or not.

        Returns:
            dict: Losses of Votenet.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (vote_targets, vote_target_masks, center_targets, bbox_targets, mask_targets, valid_gt_masks,
         objectness_targets, objectness_weights, box_loss_weights,
         valid_gt_weights, assignment) = targets

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds['seed_points'],
                                              bbox_preds['vote_points'],
                                              bbox_preds['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate objectness loss
        objectness_loss_1 = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)
        
        objectness_loss_2 = self.objectness_loss(
            bbox_preds['R_obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        objectness_loss_3 = self.objectness_loss(
            bbox_preds['R_obj_scores_jitter'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        objectness_loss = objectness_loss_1 + (objectness_loss_2 + objectness_loss_3) * 0.5

        # calculate center loss
        source2target_loss, target2source_loss = self.center_loss(
            bbox_preds['bbox_preds'][..., :3],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        center_loss = source2target_loss + target2source_loss

        # calculate surface loss
        surface_weight = box_loss_weights.reshape(-1).unsqueeze(-1).repeat(1,6)
        surface_loss = self.surface_loss(
            bbox_preds['surface_pred'].reshape(-1, 6),
            torch.cat(bbox_targets, dim=0),
            bbox_preds['surface_scale'].reshape(-1, 6),
            bbox_preds['aggregated_points'].reshape(-1, 3),
            bbox_preds['bbox_probs'].permute(0,3,1,2).reshape(-1, 6, self.reg_max+1),
            weight=surface_weight,
            reduction_override='none',
        )
        indx = bbox_preds['sem_scores'].max(dim=-1)[1].reshape(-1)
        N_class = bbox_preds['side_scores'].shape[-1]
        side_pred = bbox_preds['side_scores'].reshape(-1, 6, N_class)
        side_scores = torch.stack([side_pred[i,:,indx[i]] for i in range(indx.shape[0])]).reshape(-1, 6)
        sigma = 0.8 * side_scores * side_scores - 1.8 * side_scores + torch.ones_like(side_scores)
        surface_loss = torch.exp(-sigma.detach()) * surface_loss
        surface_loss = surface_loss.sum()

        # calculate angle loss
        pred_angle = bbox_preds['bbox_preds'][..., -1].reshape(-1)
        target_angle = torch.cat(bbox_targets, dim=0)[..., -1]
        pred_angle_sin = torch.sin(pred_angle)
        pred_angle_cos = torch.cos(pred_angle)
        target_angle_sin = torch.sin(target_angle)
        target_angle_cos = torch.cos(target_angle)
        angle_sin_loss = self.angle_loss(pred_angle_sin, target_angle_sin, weight=box_loss_weights.reshape(-1), reduction_override='none')
        angle_cos_loss = self.angle_loss(pred_angle_cos, target_angle_cos, weight=box_loss_weights.reshape(-1), reduction_override='none')
        angle_loss = angle_sin_loss + angle_cos_loss
        N_class = bbox_preds['rotate_scores'].shape[-1]
        angle_pred_scores = bbox_preds['rotate_scores'].reshape(-1, N_class)
        angle_scores = torch.stack([angle_pred_scores[i,indx[i]] for i in range(indx.shape[0])]).reshape(-1)
        angle_sigma = 0.8 * angle_scores * angle_scores - 1.8 * angle_scores + torch.ones_like(angle_scores)
        angle_loss = torch.exp(-angle_sigma.detach()) * angle_loss
        angle_loss = angle_loss.sum()

        # calculate semantic loss
        semantic_loss = self.semantic_loss(
            bbox_preds['sem_scores'].transpose(2, 1),
            mask_targets,
            weight=box_loss_weights)

        # calculate iou loss
        iou_weight = box_loss_weights.reshape(-1)
        iou_loss = self.iou_loss(
            bbox_preds['bbox_preds'].reshape(-1, 7),
            torch.cat(bbox_targets, dim=0),
            weight=iou_weight,
            reduction_override='none',
        ).reshape(-1)
        sigma_mean = sigma.mean(dim=-1)
        iou_loss = torch.exp(-sigma_mean.detach()) * iou_loss
        iou_loss = iou_loss.sum()

        # calculate iou pred loss
        targets = torch.cat(bbox_targets, dim=0)
        targets = targets.view_as(bbox_preds['bbox_preds'])
        label_iou = cal_iou_3d(bbox_preds['bbox_preds'], targets).detach().view(-1)
        label_iou_jitter = cal_iou_3d(bbox_preds['jitter_bbox_preds'], targets).detach().view(-1)
        label_cls = mask_targets.reshape(-1)
        label_cls_jitter = mask_targets.reshape(-1)

        loss_iou = self.iou_pred_loss(
            bbox_preds['iou_scores'].reshape(-1, self.num_classes), (label_cls, label_iou),
            weight=box_loss_weights.reshape(-1))
        loss_iou_jitter = self.iou_pred_loss(
            bbox_preds['iou_scores_jitter'].reshape(-1, self.num_classes), (label_cls_jitter, label_iou_jitter),
            weight=box_loss_weights.reshape(-1))
        iou_pred_loss = loss_iou + loss_iou_jitter

        # calculate side pred loss
        side_pred = bbox_preds['side_scores'].reshape(-1, 6, self.num_classes)
        side_pred = torch.stack([side_pred[i,:,label_cls[i]] for i in range(label_cls.shape[0])])
        bbox_side_gt = torch.cat(bbox_targets, dim=0)
        side_loss = self.side_loss(
            side_pred,
            bbox_preds['surface_pred'].reshape(-1, 6),
            bbox_side_gt,
            bbox_preds['surface_scale'].reshape(-1, 6),
            bbox_preds['aggregated_points'].reshape(-1, 3),
            bbox_preds['bbox_probs'].permute(0,3,1,2).reshape(-1, 6, self.reg_max+1),
            weight=surface_weight,
        )

        side_pred_jitter = bbox_preds['side_scores_jitter'].reshape(-1, 6, self.num_classes)
        side_pred_jitter = torch.stack([side_pred_jitter[i,:,label_cls_jitter[i]] for i in range(label_cls_jitter.shape[0])])
        bbox_side_gt = torch.cat(bbox_targets, dim=0)
        side_jitter_loss = self.side_loss(
            side_pred_jitter,
            bbox_preds['jitter_surface_preds'].reshape(-1, 6).detach(),
            bbox_side_gt,
            bbox_preds['surface_scale'].reshape(-1, 6),
            bbox_preds['aggregated_points'].reshape(-1, 3),
            bbox_preds['bbox_probs'].permute(0,3,1,2).reshape(-1, 6, self.reg_max+1),
            weight=surface_weight,
        )
        side_loss = side_loss + side_jitter_loss
        
        # total loss
        losses = dict(
            vote_loss=vote_loss,
            objectness_loss=objectness_loss,
            semantic_loss=semantic_loss,
            center_loss=center_loss,
            surface_loss=surface_loss,
            angle_loss=angle_loss,
            iou_loss=iou_loss,
            iou_pred_loss=iou_pred_loss,
            side_loss=side_loss)

        if ret_target:
            losses['targets'] = targets

        return losses

    @force_fp32(apply_to=('bbox_preds', ))
    def unsup_loss(self,
             bbox_preds,
             points,
             pseudo_boxes,
             pseudo_label,
             img_metas=None,
             pseudo_quality_score=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of vote head.
            points (list[torch.Tensor]): Input points.
            pseudo_boxes (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            pseudo_label (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.
            ret_target (Bool): Return targets or not.

        Returns:
            dict: Losses of Votenet.
        """
        targets = self.get_targets(points, pseudo_boxes, pseudo_label, bbox_preds=bbox_preds)
        (vote_targets, vote_target_masks, center_targets, bbox_targets, mask_targets, valid_gt_masks,
         objectness_targets, objectness_weights, box_loss_weights,
         valid_gt_weights, assignment) = targets
        B, num_proposal = assignment.shape
        pseudo_quality_side = []
        for i in range(B):
            if pseudo_quality_score[i].shape[0] != 0:
                pseudo_quality_side.append(pseudo_quality_score[i][assignment[i]])
            else:
                pseudo_quality_side.append(torch.zeros(num_proposal, 6).to(assignment.device))
        pseudo_quality_side = torch.stack(pseudo_quality_side)
        pseudo_quality_mean = pseudo_quality_side.mean(dim=-1)
        # calculate center loss
        unsup_source2target_loss, unsup_target2source_loss = self.center_loss(
            bbox_preds['bbox_preds'][..., :3],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        unsup_center_loss = unsup_source2target_loss + unsup_target2source_loss

        # calculate semantic loss
        unsup_semantic_loss = self.semantic_loss(
            bbox_preds['sem_scores'].transpose(2, 1),
            mask_targets,
            weight=box_loss_weights)

        # calculate iou loss
        iou_weight = (box_loss_weights * pseudo_quality_mean).reshape(-1) 
        unsup_iou_loss = self.iou_loss(
            bbox_preds['bbox_preds'].reshape(-1, 7),
            torch.cat(bbox_targets, dim=0),
            weight=iou_weight,
            reduction_override='none',
        )

        indx = bbox_preds['sem_scores'].max(dim=-1)[1].reshape(-1)
        N_class = bbox_preds['side_scores'].shape[-1]
        side_pred = bbox_preds['side_scores'].reshape(-1, 6, N_class)
        side_scores = torch.stack([side_pred[i,:,indx[i]] for i in range(indx.shape[0])]).reshape(-1, 6)
        sigma = 0.8 * side_scores * side_scores - 1.8 * side_scores + torch.ones_like(side_scores)
        sigma_mean = sigma.mean(dim=-1)
        unsup_iou_loss = torch.exp(-sigma_mean.detach()) * unsup_iou_loss
        unsup_iou_loss = unsup_iou_loss.sum()

        # calculate surface loss
        surface_weight = box_loss_weights.reshape(-1).unsqueeze(-1).repeat(1,6)* pseudo_quality_side.reshape(-1, 6)
        unsup_surface_loss = self.surface_loss(
            bbox_preds['surface_pred'].reshape(-1, 6),
            torch.cat(bbox_targets, dim=0),
            bbox_preds['surface_scale'].reshape(-1, 6),
            bbox_preds['aggregated_points'].reshape(-1, 3),
            bbox_preds['bbox_probs'].permute(0,3,1,2).reshape(-1, 6, self.reg_max+1),
            weight=surface_weight,
            reduction_override='none',
        )
        unsup_surface_loss = torch.exp(-sigma.detach()) * unsup_surface_loss
        unsup_surface_loss = unsup_surface_loss.sum()     

        un_label_weight = 2.0

        unsup_losses = dict(
            unsup_semantic_loss=un_label_weight*unsup_semantic_loss,
            unsup_center_loss=un_label_weight*unsup_center_loss,
            unsup_iou_loss=un_label_weight*unsup_iou_loss,
            unsup_surface_loss=un_label_weight*unsup_surface_loss)
            
        return unsup_losses
    
    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of vote head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of vote head.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, vote_target_masks, center_targets, bbox_targets,
         mask_targets, objectness_targets, objectness_masks, assignment) = multi_apply(
             self.get_targets_single, points,
             gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask,
             aggregated_points)

        # pad targets as original code of votenet.
        for index in range(len(gt_labels_3d)):
            pad_num = max_gt_num - gt_labels_3d[index].shape[0]
            center_targets[index] = F.pad(center_targets[index],
                                          (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)
        center_targets = torch.stack(center_targets)
        valid_gt_masks = torch.stack(valid_gt_masks)
        assignment = torch.stack(assignment)

        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)
        valid_gt_weights = valid_gt_masks.float() / (
            torch.sum(valid_gt_masks.float()) + 1e-6)
        mask_targets = torch.stack(mask_targets)

        return (vote_targets, vote_target_masks, center_targets, bbox_targets, mask_targets,
                valid_gt_masks, objectness_targets, objectness_weights,
                box_loss_weights, valid_gt_weights, assignment)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None):
        """Generate targets of vote head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                vote aggregation layer.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        # assert self.bbox_coder.with_rot or pts_semantic_mask is not None

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]

        vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
        vote_target_masks = points.new_zeros([num_points],
                                                dtype=torch.long)
        vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
        box_indices_all = gt_bboxes_3d.points_in_boxes(points)
        for i in range(gt_labels_3d.shape[0]):
            box_indices = box_indices_all[:, i]
            indices = torch.nonzero(
                box_indices, as_tuple=False).squeeze(-1)
            selected_points = points[indices]
            vote_target_masks[indices] = 1
            vote_targets_tmp = vote_targets[indices]
            votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                0) - selected_points[:, :3]

            for j in range(self.gt_per_seed):
                column_indices = torch.nonzero(
                    vote_target_idx[indices] == j,
                    as_tuple=False).squeeze(-1)
                vote_targets_tmp[column_indices,
                                    int(j * 3):int(j * 3 +
                                                3)] = votes[column_indices]
                if j == 0:
                    vote_targets_tmp[column_indices] = votes[
                        column_indices].repeat(1, self.gt_per_seed)

            vote_targets[indices] = vote_targets_tmp
            vote_target_idx[indices] = torch.clamp(
                vote_target_idx[indices] + 1, max=2)


        center_targets = gt_bboxes_3d.gravity_center

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        objectness_targets[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
        objectness_masks[
            euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

        mask_targets = gt_labels_3d[assignment]
        bbox_targets = torch.cat((center_targets[assignment], gt_bboxes_3d.tensor[assignment, 3:]), dim=-1)

        return (vote_targets, vote_target_masks, center_targets, bbox_targets,
                mask_targets.long(), objectness_targets, objectness_masks, assignment)

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True,
                   use_iou_for_nms=True):
        """Generate bboxes from vote head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from vote head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.
            use_nms (bool): Whether to apply NMS, skip nms postprocessing
                while using vote head in rpn stage.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        obj_scores = F.softmax(bbox_preds['R_obj_scores'], dim=-1)[..., -1]
        sem_scores = F.softmax(bbox_preds['sem_scores'], dim=-1)
        bbox3d = bbox_preds['bbox_preds']  # self.bbox_coder.decode(bbox_preds)
        if use_iou_for_nms:
            B, num_proposal = bbox_preds['iou_scores'].shape[:2]
            _, indx = bbox_preds['sem_scores'].reshape(-1, self.num_classes).max(dim=-1)
            iou_logits = bbox_preds['iou_scores'].reshape(-1, self.num_classes)
            iou_logits = torch.stack([iou_logits[i,indx[i]] for i in range(B*num_proposal)]).reshape(B, -1)
            obj_scores = obj_scores * iou_logits

        if use_nms:
            batch_size = bbox3d.shape[0]
            results = list()
            for b in range(batch_size):
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3],
                                               input_metas[b])
                bbox = input_metas[b]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=True)  # self.bbox_coder.with_rot)
                results.append((bbox, score_selected, labels))

            return results
        else:
            return bbox3d

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=True,  # self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def forward_onlyiou_faster(self, results, center, size, heading, dataset_name = 'ScanNet'):
        if dataset_name == 'ScanNet':
            heading_ = torch.zeros_like(heading)
        else:
            heading_ = heading
        B = center.shape[0]
        results = self.grid_conv(center, size, heading_, results)
        _, indx = results['sem_scores'].reshape(-1, self.num_classes).max(dim=-1)
        iou_scores = results['iou_scores'].reshape(-1, self.num_classes)
        results['iou_scores'] = torch.stack([iou_scores[i,indx[i]] for i in range(iou_scores.shape[0])]).reshape(B, -1)
        return results

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