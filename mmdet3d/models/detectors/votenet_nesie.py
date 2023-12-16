import torch
from collections import Counter

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d, DepthInstance3DBoxes
from mmdet.models import DETECTORS
from .single_stage import SingleStage3DDetector
from torch.nn import functional as F
import mmcv
from mmcv.runner import HOOKS, Hook
from mmcv.runner import Priority, get_priority
import torch
import numpy as np

@DETECTORS.register_module()
class VoteNetNesie(SingleStage3DDetector):
    r"""Nesie for 3D detection."""

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                custom_config=None,):
        super(VoteNetNesie, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)
        self._hooks = []
        self.register_custom_hooks(custom_config)
    
    def combine_data(self, label_data, unlabel_datas):
        ratio = len(unlabel_datas)
        for i in range(ratio):
            for key, val in label_data.items():
                if key not in ['use_label_t', 'use_label_s']:
                    val.extend(unlabel_datas[i][key])
                else:
                    label_data[key] = torch.concat((val,unlabel_datas[i][key]),dim=0)
        return label_data

    def choose_sup_item(self, bbox_preds, loss_inputs, use_label):
        inds = [i for i in range(len(use_label)) if use_label[i] == True]
        sup_bbox_preds = {}
        sup_loss_inputs = []
        for key, val in bbox_preds.items():
            sup_bbox_preds[key] = val[inds]
        for item in loss_inputs:
            if item != None:
                sup_loss_inputs.append([item[d] for d in range(len(item)) if d in inds])
            else:
                sup_loss_inputs.append(item)
        return sup_bbox_preds, sup_loss_inputs

    def choose_unsup_item(self, bbox_preds, loss_inputs, use_label):
        inds = [i for i in range(len(use_label)) if use_label[i] != True]
        unsup_bbox_preds = {}
        unsup_loss_inputs = []
        for key, val in bbox_preds.items():
            unsup_bbox_preds[key] = val[inds]
        for item in loss_inputs:
            unsup_loss_inputs.append([item[d] for d in range(len(item)) if d in inds])
        return unsup_bbox_preds, unsup_loss_inputs

    def forward_train(self, label_data, unlabel_datas, img_metas):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        data = self.combine_data(label_data, unlabel_datas)
        points_s = data['points_s']
        points_t = data['points_t']
        gt_bboxes_3d_s = data['gt_bboxes_3d_s']
        gt_bboxes_3d_t = data['gt_bboxes_3d_t']
        gt_labels_3d_s = data['gt_labels_3d_s']
        gt_labels_3d_t = data['gt_labels_3d_t']
        if "pts_semantic_mask_s" in data.keys():
            pts_semantic_mask_s = data['pts_semantic_mask_s']
            pts_instance_mask_s = data['pts_instance_mask_s']
        else:
            pts_semantic_mask_s = None
            pts_instance_mask_s = None
        img_metas_s = data['img_metas_s']
        img_metas_t = data['img_metas_t']
        use_label_s = data['use_label_s'].tolist()
        use_label_t = data['use_label_t'].tolist()
        points_cat_s = torch.stack(points_s)
        points_cat_t = torch.stack(points_t)

        x_s = self.extract_feat(points_cat_s)
        bbox_preds_s = self.bbox_head(x_s, self.train_cfg.sample_mod, self.train_cfg.dataset_name)
        loss_inputs_s = (points_s, gt_bboxes_3d_s, gt_labels_3d_s, pts_semantic_mask_s,
                       pts_instance_mask_s, img_metas_s)

        with torch.no_grad():
            self.call_hook("switch_to_teacher")
            x_t = self.extract_feat(points_cat_t)
            bbox_preds_t = self.bbox_head(x_t, self.train_cfg.sample_mod, self.train_cfg.dataset_name)
            pseudo_label, pseudo_boxes, pseudo_quality_score = self.get_pseudo_labels(bbox_preds_t, self.train_cfg.dataset_name)
            pseudo_boxes = self.transformation_bbox_preds(pseudo_boxes, img_metas_t, img_metas_s)
            self.call_hook("switch_to_student")

        sup_bbox_preds, sup_loss_inputs = self.choose_sup_item(bbox_preds_s, loss_inputs_s, use_label_s)
        sup_losses = self.bbox_head.loss(sup_bbox_preds, *sup_loss_inputs)
        loss_inputs = (points_s, pseudo_boxes, pseudo_label, img_metas_s, pseudo_quality_score)
        unsup_bbox_preds, unsup_loss_inputs = self.choose_unsup_item(bbox_preds_s, loss_inputs, use_label_s)
        self.ulb_update(unsup_loss_inputs[2], unsup_loss_inputs[3])
        unsup_losses = self.bbox_head.unsup_loss(unsup_bbox_preds, *unsup_loss_inputs)
        losses = {**sup_losses, **unsup_losses}
        return losses
    
    def get_pseudo_labels(self, unsup_bbox_preds, dataset_name="ScanNet"):
        '''
        'obj_threshold': 0.9, 'cls_threshold': 0.9, 'iou_threshold': 0.25
        '''
        selected_label = self.ulb_list
        ulb_count = 10 * self.ulb_flag.sum() * len(self.lb_map) / len(self.ulb_map)
        pseudo_counter = selected_label.sum(dim=0)
        sorted, indices = torch.sort(pseudo_counter, descending=True)
        # quality_score_exp = torch.exp(-unsup_bbox_preds['quality_score'])
        # var_weight = variance(unsup_bbox_preds['bbox_probs'], unsup_bbox_preds['surface_scale'])
        classwise_acc = torch.zeros((len(self.CLASSES),)).cuda()
        if self.train_cfg.thresh_warmup:
            for i in indices:
                classwise_acc[i] = sorted[i] / max(max(sorted),ulb_count)
                classwise_acc[i] = (classwise_acc[i] / (2. - classwise_acc[i]))
        else:
            for i in indices:
                classwise_acc[i] = sorted[i] / max(sorted)
                classwise_acc[i] = (classwise_acc[i] / (2. - classwise_acc[i]))

        unsup_bbox_preds['bbox_preds'][:,:,2] = unsup_bbox_preds['bbox_preds'][:,:,2] - unsup_bbox_preds['bbox_preds'][:,:,5] * 0.5
        pred_center = unsup_bbox_preds['bbox_preds'][:,:,:3]
        pred_size = unsup_bbox_preds['bbox_preds'][:,:,3:6]
        pred_heading = unsup_bbox_preds['bbox_preds'][:,:,6:7]
        batch_size, num_proposal = pred_center.shape[:2]

        # cls score threshold
        pred_sem_cls = unsup_bbox_preds['sem_scores']
        max_cls, argmax_cls = torch.max(pred_sem_cls, dim=2)
        B, proposal_num = argmax_cls.shape
        argmax_cls_flatten = argmax_cls.reshape(-1)
        if self.train_cfg.use_cbl:
            threshold = torch.tensor([classwise_acc[argmax_cls_flatten[i]] for i in argmax_cls_flatten])\
                        .reshape(B, proposal_num).to(pred_sem_cls.device)
            cls_threshold = 0.7 + 0.3 * threshold
            cls_threshold[cls_threshold>0.95] = 0.95
        else:
            cls_threshold = 0.9
        cls_mask = max_cls > cls_threshold

        # obj score threshold
        pred_objectness = torch.nn.Softmax(dim=2)(unsup_bbox_preds['obj_scores'])
        # the second element is positive score
        pos_obj = pred_objectness[:, :, 1]
        neg_obj = pred_objectness[:, :, 0]
        if self.train_cfg.use_cbl:
            # obj_threshold = 0.8 + 0.2 * threshold
            obj_threshold = 0.9
            # obj_threshold[obj_threshold>0.95] = 0.95
        else:
            obj_threshold = 0.9
        objectness_mask = pos_obj > obj_threshold
        neg_objectness_mask = neg_obj > obj_threshold

        # IoU score threshold
        indx = argmax_cls.reshape(-1)
        pred_iou_scores = unsup_bbox_preds['iou_scores']
        iou_logits = pred_iou_scores.reshape(-1, len(self.CLASSES))
        iou_pred = torch.stack([iou_logits[i,indx[i]] for i in range(B*num_proposal)]).reshape(B, -1)        
        if self.train_cfg.use_cbl:
            iou_threshold = 0.25 + threshold * 0.5
            iou_threshold[iou_threshold>0.35] = 0.35
            # iou_threshold[iou_threshold>0.35] = 0.35
        else:
            iou_threshold = 0.25
        iou_mask = iou_pred > iou_threshold
        before_iou_mask = torch.logical_and(cls_mask, objectness_mask)
        final_mask = torch.logical_and(before_iou_mask, iou_mask)

        # side score
        side_pred = unsup_bbox_preds['side_scores'].reshape(-1, 6, len(self.CLASSES)).detach()
        side_scores = torch.stack([side_pred[i,:,indx[i]] for i in range(indx.shape[0])]).reshape(B, -1, 6)
        quality_score = 5/3* side_scores * side_scores - 8/3* side_scores + torch.ones_like(side_scores)

        # we only keep MAX_NUM_OBJ=64 predictions
        # however, after filtering the number can still exceed this
        # so we keep the ones with larger pos_obj * max_cls
        MAX_NUM_OBJ = 64
        inds = torch.argsort(pos_obj * iou_pred * final_mask, dim=1, descending=True)
        inds = inds[:, :MAX_NUM_OBJ].long()
        final_mask_sorted = torch.gather(final_mask, dim=1, index=inds)
        neg_objectness_mask = torch.gather(neg_objectness_mask, dim=1, index=inds)

        use_lhs = True
        if use_lhs:
            pred_center_ = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
            pred_size_ = torch.gather(pred_size, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
            pred_heading_ = torch.gather(pred_heading, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 1))
            num_proposal = pred_center_.shape[1]
            bsize = pred_center_.shape[0]
            pred_box_parameters = np.zeros((bsize, num_proposal, 7), dtype=np.float32)
            pred_box_parameters[:, :, 0:3] = pred_center_.detach().cpu().numpy()
            pred_box_parameters[:, :, 3:6] = pred_size_.detach().cpu().numpy()
            pred_box_parameters[:, :, 6:7] = pred_heading_.detach().cpu().numpy()
            pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3), dtype=np.float32)
            pred_center_upright_camera = flip_axis_to_camera(pred_center_.detach().cpu().numpy())
            for i in range(bsize):
                for j in range(num_proposal):
                    if dataset_name == "ScanNet":
                        pred_box_parameters[i, j, 6] = 0.0
                    corners_3d_upright_camera = get_3d_box(pred_box_parameters[i, j, 3:6], pred_box_parameters[i, j, 6], pred_center_upright_camera[i, j, :])
                    pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

            # pred_corners_3d_upright_camera, _ = predictions2corners3d(end_points, config_dict)
            pred_mask = np.ones((batch_size, MAX_NUM_OBJ))
            nonempty_box_mask = np.ones((batch_size, MAX_NUM_OBJ))
            pos_obj_numpy = torch.gather(pos_obj, dim=1, index=inds).detach().cpu().numpy()
            pred_sem_cls_numpy = torch.gather(argmax_cls, dim=1, index=inds).detach().cpu().numpy()
            iou_numpy = torch.gather(iou_pred, dim=1, index=inds).detach().cpu().numpy()
            for i in range(batch_size):
                boxes_3d_with_prob = np.zeros((MAX_NUM_OBJ, 8))
                for j in range(MAX_NUM_OBJ):
                    boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                    boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                    boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                    boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                    boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                    boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                    boxes_3d_with_prob[j, 6] = pos_obj_numpy[i, j] * iou_numpy[i, j]
                    boxes_3d_with_prob[j, 7] = pred_sem_cls_numpy[
                        i, j]  # only suppress if the two boxes are of the same class!!
                nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]

                # here we do not consider orientation, in accordance to test time nms
                nms_iou = 0.25
                use_old_type_nms = False
                pick = lhs_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                            nms_iou, use_old_type_nms)
                assert (len(pick) > 0)
                pred_mask[i, nonempty_box_inds[pick]] = 0
            # end_points['pred_mask'] = pred_mask
            final_mask_sorted[torch.from_numpy(pred_mask).bool().cuda()] = 0

        label_mask = torch.zeros((batch_size, MAX_NUM_OBJ), dtype=torch.long).cuda()
        label_mask[final_mask_sorted] = 1
        heading_label = torch.gather(pred_heading, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 1))
        size_label = torch.gather(pred_size, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        # var_weight_label = torch.gather(var_weight, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 6))
        quality_score = torch.gather(quality_score, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 6))
        sem_cls_label = torch.gather(argmax_cls, dim=1, index=inds)
        center_label = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        center_label[(1 - label_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000
        pred_vote_xyz = unsup_bbox_preds['vote_points']
        false_center_label = torch.gather(pred_vote_xyz, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        false_center_label[torch.logical_not(neg_objectness_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000

        pseudo_label = []
        pseudo_boxes = []
        # pseudo_var_weight = []
        pseudo_quality_score = []
        for i in range(batch_size):
            select_indx = label_mask[i]
            if select_indx.sum() != 0:
                center_label_item = torch.stack([center_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])
                size_label_item = torch.stack([size_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])
                heading_label_item = torch.stack([heading_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])
                box_label_item = torch.cat((center_label_item,size_label_item,heading_label_item),dim=-1)
                sem_cls_label_item = torch.stack([sem_cls_label[i,j] for j in range(len(select_indx)) if select_indx[j]==1])
                # var_weight_label_item = torch.stack([var_weight_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])            
                quality_score_label_item = torch.stack([quality_score[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])            
            else:
                sem_cls_label_item = torch.ones((0)).to(center_label.device)
                box_label_item = torch.ones((0,7)).to(center_label.device)
                # var_weight_label_item = torch.ones((0,6)).to(center_label.device)
                quality_score_label_item = torch.ones((0,6)).to(center_label.device)
            # bbox = DepthInstance3DBoxes(box_label_item, box_dim=box_label_item.shape[-1], with_yaw=True)
            pseudo_label.append(sem_cls_label_item)
            pseudo_boxes.append(box_label_item)
            # pseudo_var_weight.append(var_weight_label_item)
            pseudo_quality_score.append(quality_score_label_item)
        return pseudo_label, pseudo_boxes, pseudo_quality_score
    
    def ulb_update(self, pseudo_label, img_metas):
        ulb_num = len(img_metas)
        for idx in range(ulb_num):
            item = img_metas[idx]
            p=self.ulb_map.index(item['sample_idx'])
            self.ulb_flag[p] = 0.0
            pseudo_lb = Counter(pseudo_label[idx].tolist())
            self.ulb_list[p] = torch.tensor([pseudo_lb[i] for i in range(self.ulb_list.shape[-1])])

    def transformation_bbox_preds(self, bbox3d, img_metas_t=None, img_metas_s=None):
        """transform bbox predictions."""
        batch_size = len(bbox3d)
        bboxes_3d = []
        for b in range(batch_size):
            bboxes = DepthInstance3DBoxes(
                bbox3d[b].to('cpu'),
                box_dim = bbox3d[b].shape[-1],
                with_yaw=True)
            if img_metas_t != None:  
                self.untransformation(bboxes,img_metas_t[b])
            if img_metas_s != None:
                self.transformation(bboxes,img_metas_s[b])
            bboxes_3d.append(bboxes)
        return bboxes_3d

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        if self.test_cfg.iou_opt:
            bbox_results = self.iou_opt_test(points,img_metas,imgs,rescale)
        else:
            points_cat = torch.stack(points)
            x = self.extract_feat(points_cat)
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod, self.test_cfg.dataset_name)
            bbox_list = self.bbox_head.get_bboxes(
                points_cat, bbox_preds, img_metas, rescale=rescale, use_iou_for_nms=self.test_cfg.use_iou_for_nms)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
        if self.test_cfg.add_info:
            pseudo_label, pseudo_boxes, pseudo_var_weight = self.test_pseudo_labels(bbox_preds.copy(), self.test_cfg.dataset_name)
            bbox_results[0] = dict(
                **bbox_results[0], 
                quality_score = bbox_preds['side_scores'].to('cpu'),
                bbox_pred = bbox_preds['bbox_preds'].to('cpu'),
                aggregated_points = bbox_preds['aggregated_points'].to('cpu'),
                obj_scores = bbox_preds['obj_scores'].to('cpu'),
                sem_scores = bbox_preds['sem_scores'].to('cpu'),
                surface_pred = bbox_preds['surface_pred'].to('cpu'),
                surface_scale = bbox_preds['surface_scale'].to('cpu'),
                bbox_probs = bbox_preds['bbox_probs'].to('cpu'),
                iou_scores = bbox_preds['iou_scores'].to('cpu'),
                pseudo_boxes = pseudo_boxes[0].to('cpu'),
                pseudo_label = pseudo_label[0].to('cpu'),
                pseudo_var_weight = pseudo_var_weight[0].to('cpu'),
                )
        return bbox_results


    def test_pseudo_labels(self, unsup_bbox_preds, dataset_name="ScanNet"):
        '''
        'obj_threshold': 0.9, 'cls_threshold': 0.9, 'iou_threshold': 0.25
        '''
        unsup_bbox_preds['bbox_preds'][:,:,2] = unsup_bbox_preds['bbox_preds'][:,:,2] - unsup_bbox_preds['bbox_preds'][:,:,5] * 0.5
        pred_center = unsup_bbox_preds['bbox_preds'][:,:,:3]
        pred_size = unsup_bbox_preds['bbox_preds'][:,:,3:6]
        pred_heading = unsup_bbox_preds['bbox_preds'][:,:,6:7]
        batch_size, num_proposal = pred_center.shape[:2]

        # cls score threshold
        pred_sem_cls = unsup_bbox_preds['sem_scores']
        max_cls, argmax_cls = torch.max(pred_sem_cls, dim=2)
        B, proposal_num = argmax_cls.shape
        argmax_cls_flatten = argmax_cls.reshape(-1)
        cls_threshold = 0.70
        cls_mask = max_cls > cls_threshold

        # obj score threshold
        pred_objectness = torch.nn.Softmax(dim=2)(unsup_bbox_preds['obj_scores'])
        # the second element is positive score
        pos_obj = pred_objectness[:, :, 1]
        neg_obj = pred_objectness[:, :, 0]
        obj_threshold = 0.80
        objectness_mask = pos_obj > obj_threshold
        neg_objectness_mask = neg_obj > obj_threshold

        # IoU score threshold
        indx = argmax_cls.reshape(-1)
        pred_iou_scores = unsup_bbox_preds['iou_scores']
        iou_logits = pred_iou_scores.reshape(-1, len(self.CLASSES))
        iou_pred = torch.stack([iou_logits[i,indx[i]] for i in range(B*num_proposal)]).reshape(B, -1)        
        iou_threshold = 0.15
        iou_mask = iou_pred > iou_threshold
        before_iou_mask = torch.logical_and(cls_mask, objectness_mask)
        final_mask = torch.logical_and(before_iou_mask, iou_mask)

        # side score
        side_pred = unsup_bbox_preds['side_scores'].reshape(-1, 6, len(self.CLASSES)).detach()
        side_scores = torch.stack([side_pred[i,:,indx[i]] for i in range(indx.shape[0])]).reshape(B, -1, 6)
        quality_score = 5/3* side_scores * side_scores - 8/3* side_scores + torch.ones_like(side_scores)

        MAX_NUM_OBJ = 64
        inds = torch.argsort(pos_obj * iou_pred * final_mask, dim=1, descending=True)
        inds = inds[:, :MAX_NUM_OBJ].long()
        final_mask_sorted = torch.gather(final_mask, dim=1, index=inds)
        neg_objectness_mask = torch.gather(neg_objectness_mask, dim=1, index=inds)

        use_lhs = True
        if use_lhs:
            pred_center_ = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
            pred_size_ = torch.gather(pred_size, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
            pred_heading_ = torch.gather(pred_heading, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 1))
            num_proposal = pred_center_.shape[1]
            bsize = pred_center_.shape[0]
            pred_box_parameters = np.zeros((bsize, num_proposal, 7), dtype=np.float32)
            pred_box_parameters[:, :, 0:3] = pred_center_.detach().cpu().numpy()
            pred_box_parameters[:, :, 3:6] = pred_size_.detach().cpu().numpy()
            pred_box_parameters[:, :, 6:7] = pred_heading_.detach().cpu().numpy()
            pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3), dtype=np.float32)
            pred_center_upright_camera = flip_axis_to_camera(pred_center_.detach().cpu().numpy())
            for i in range(bsize):
                for j in range(num_proposal):
                    if dataset_name == "ScanNet":
                        pred_box_parameters[i, j, 6] = 0.0
                    corners_3d_upright_camera = get_3d_box(pred_box_parameters[i, j, 3:6], pred_box_parameters[i, j, 6], pred_center_upright_camera[i, j, :])
                    pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

            # pred_corners_3d_upright_camera, _ = predictions2corners3d(end_points, config_dict)
            pred_mask = np.ones((batch_size, MAX_NUM_OBJ))
            nonempty_box_mask = np.ones((batch_size, MAX_NUM_OBJ))
            pos_obj_numpy = torch.gather(pos_obj, dim=1, index=inds).detach().cpu().numpy()
            pred_sem_cls_numpy = torch.gather(argmax_cls, dim=1, index=inds).detach().cpu().numpy()
            iou_numpy = torch.gather(iou_pred, dim=1, index=inds).detach().cpu().numpy()
            for i in range(batch_size):
                boxes_3d_with_prob = np.zeros((MAX_NUM_OBJ, 8))
                for j in range(MAX_NUM_OBJ):
                    boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                    boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                    boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                    boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                    boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                    boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                    boxes_3d_with_prob[j, 6] = pos_obj_numpy[i, j] * iou_numpy[i, j]
                    boxes_3d_with_prob[j, 7] = pred_sem_cls_numpy[
                        i, j]  # only suppress if the two boxes are of the same class!!
                nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]

                # here we do not consider orientation, in accordance to test time nms
                nms_iou = 0.25
                use_old_type_nms = False
                pick = lhs_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                            nms_iou, use_old_type_nms)
                assert (len(pick) > 0)
                pred_mask[i, nonempty_box_inds[pick]] = 0
            # end_points['pred_mask'] = pred_mask
            final_mask_sorted[torch.from_numpy(pred_mask).bool().cuda()] = 0

        label_mask = torch.zeros((batch_size, MAX_NUM_OBJ), dtype=torch.long).cuda()
        label_mask[final_mask_sorted] = 1
        heading_label = torch.gather(pred_heading, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 1))
        size_label = torch.gather(pred_size, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        quality_score_label = torch.gather(quality_score, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 6))
        sem_cls_label = torch.gather(argmax_cls, dim=1, index=inds)
        center_label = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        center_label[(1 - label_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000
        pred_vote_xyz = unsup_bbox_preds['vote_points']
        false_center_label = torch.gather(pred_vote_xyz, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        false_center_label[torch.logical_not(neg_objectness_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000

        pseudo_label = []
        pseudo_boxes = []
        pseudo_quality_score = []
        for i in range(batch_size):
            select_indx = label_mask[i]
            if select_indx.sum() != 0:
                center_label_item = torch.stack([center_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])
                size_label_item = torch.stack([size_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])
                heading_label_item = torch.stack([heading_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])
                box_label_item = torch.cat((center_label_item,size_label_item,heading_label_item),dim=-1)
                sem_cls_label_item = torch.stack([sem_cls_label[i,j] for j in range(len(select_indx)) if select_indx[j]==1])
                quality_score_label_item = torch.stack([quality_score_label[i,j,:] for j in range(len(select_indx)) if select_indx[j]==1])            
            else:
                sem_cls_label_item = torch.ones((0)).to(center_label.device)
                box_label_item = torch.ones((0,7)).to(center_label.device)
                quality_score_label_item = torch.ones((0,6)).to(center_label.device)
            # bbox = DepthInstance3DBoxes(box_label_item, box_dim=box_label_item.shape[-1], with_yaw=True)
            pseudo_label.append(sem_cls_label_item)
            pseudo_boxes.append(box_label_item)
            pseudo_quality_score.append(quality_score_label_item)
        return pseudo_label, pseudo_boxes, pseudo_quality_score
    
    def iou_opt_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with iou opt.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """        
        with torch.enable_grad():
            if hasattr(self,'optimizer') == False:
                self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-3)

            self.optimizer.zero_grad()

            points_cat = torch.stack(points)
            points_cat.requires_grad = True
            x = self.extract_feat(points_cat)
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod, self.test_cfg.dataset_name) 
            center = bbox_preds['bbox_preds'][:,:,:3].detach()
            size = bbox_preds['bbox_preds'][:,:,3:6].detach()
            heading = bbox_preds['bbox_preds'][:,:,6:7].detach()
            center.requires_grad = True
            size.requires_grad = True  

            bbox_preds = self.bbox_head.forward_onlyiou_faster(bbox_preds, center, size, heading, self.test_cfg.dataset_name)
            iou = bbox_preds['iou_scores'].contiguous().view(-1)
            iou.backward(torch.ones_like(iou))
            # max_iou = iou_gathered.view(center.shape[:2])
            center_grad = center.grad
            size_grad = size.grad
            mask = torch.ones_like(center)
            count = 0    
            
            for k in bbox_preds.keys():
                bbox_preds[k] = bbox_preds[k].detach()
            while True:
                center_ = center.detach() + self.test_cfg.opt_rate * center_grad * mask
                size_ = size.detach() + self.test_cfg.opt_rate * size_grad * mask
                heading_ = heading.detach()
                self.optimizer.zero_grad()
                center_.requires_grad = True
                size_.requires_grad = True
                end_points_ = self.bbox_head.forward_onlyiou_faster(bbox_preds, center_, size_, heading_, self.test_cfg.dataset_name)
                iou = end_points_['iou_scores'].contiguous().view(-1)
                iou.backward(torch.ones_like(iou))
                center_grad = center_.grad
                size_grad = size_.grad
                # cur_iou = iou_gathered.view(center.shape[:2])
                # mask[cur_iou < max_iou - 0.1] = 0
                # mask[torch.abs(cur_iou - max_iou) < 0.001] = 0
                # print(mask.sum().float() /mask.view(-1).shape[-1])
                count += 1
                if count > self.test_cfg.opt_step:
                    break
                center = center_
                size = size_

            bbox_preds['bbox_preds'][:,:,:3] = center_
            bbox_preds['bbox_preds'][:,:,3:6] = size_

            self.optimizer.zero_grad()
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod, self.test_cfg.dataset_name)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def untransformation(self, bboxes_3d, img_metas):
        re_flow = img_metas['transformation_3d_flow'][::-1]
        for op in re_flow:
            if op == 'T':
                if 'pcd_trans' in img_metas.keys():
                    pcd_trans = -1.0 * img_metas['pcd_trans']
                    bboxes_3d.translate(pcd_trans)
            elif op == 'R':
                if 'pcd_rotation' in img_metas.keys():
                    pcd_rotation = img_metas['pcd_rotation']
                    bboxes_3d.rotate(pcd_rotation)
            elif op == 'S':
                if 'pcd_trans' in img_metas.keys():
                    pcd_scale_factor = 1.0 / img_metas['pcd_scale_factor']
                    bboxes_3d.scale(pcd_scale_factor)
            elif op == 'VF':
                bboxes_3d.flip("vertical")
            elif op == 'HF':
                bboxes_3d.flip("horizontal")

    def transformation(self, bboxes_3d, img_metas):
        au_flow = img_metas['transformation_3d_flow']
        for op in au_flow:
            if op == 'T':
                if 'pcd_trans' in img_metas.keys():
                    pcd_trans = img_metas['pcd_trans']
                    bboxes_3d.translate(pcd_trans)
            elif op == 'R':
                if 'pcd_rotation' in img_metas.keys():
                    pcd_rotation = img_metas['pcd_rotation'].T
                    bboxes_3d.rotate(pcd_rotation)
            elif op == 'S':
                if 'pcd_scale_factor' in img_metas.keys():
                    pcd_scale_factor = img_metas['pcd_scale_factor']
                    bboxes_3d.scale(pcd_scale_factor)
            elif op == 'VF':
                bboxes_3d.flip("vertical")
            elif op == 'HF':
                bboxes_3d.flip("horizontal")
    
    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    def register_hook(self, hook, priority="NORMAL"):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg):
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Notes:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop("priority", "NORMAL")
        hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def get_hook_info(self):
        # Get hooks info in each stage
        stage_hook_map = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name
            except ValueError:
                priority = hook.priority
            classname = hook.__class__.__name__
            hook_info = f"({priority:<12}) {classname:<35}"
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f"{stage}:\n"
                info += "\n".join(hook_infos)
                info += "\n -------------------- "
                stage_hook_infos.append(info)
        return "\n".join(stage_hook_infos)

    def register_custom_hooks(self, custom_config):
        if custom_config is None:
            return

        if not isinstance(custom_config, list):
            custom_config = [custom_config]

        for item in custom_config:
            if isinstance(item, dict):
                self.register_hook_from_cfg(item)
            else:
                self.register_hook(item, priority="NORMAL")



def lhs_3d_faster_samecls(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    cls = boxes[:,7]
    area = (x2-x1)*(y2-y1)*(z2-z1) + 1e-8

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])
        cls1 = cls[i]
        cls2 = cls[I[:last-1]]

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)
        o = o * (cls1 == cls2)

        inds = np.where(o>overlap_threshold)[0]
        len_inds = len(inds)

        for count in range((len_inds) // 2):
            pick.append(I[inds[len_inds - count - 1]])

        I = np.delete(I, np.concatenate(([last-1], inds)))

    return pick

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
            6 -------- 5
           /|         /|
          7 -------- 4 .
          | |        | |
          . 2 -------- 1
          |/         |/
          3 -------- 0

    '''
    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2];
    y_corners = [h/2, h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def variance(probs, scales):
    probs = probs.permute(0,3,1,2)
    B, num_proposal, k, reg_max = probs.shape
    scales = scales.unsqueeze(-1).repeat(1, 1, 1, reg_max)
    values = torch.linspace(0, reg_max-1, reg_max)/(reg_max-1)
    values = values.to(probs.device).repeat(B*num_proposal*k).reshape((B, num_proposal, k, -1))
    values = values * scales
    mean = torch.sum(values * probs, dim=-1)
    var = torch.sum(values * values * probs, dim=-1) - mean * mean
    var_weight = torch.exp(-var).detach()
    return var_weight
