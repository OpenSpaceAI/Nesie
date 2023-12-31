import re
import numpy as np
import random
import tempfile
import warnings
from os import path as osp

from mmdet3d.core import show_result, show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.datasets import DATASETS
from mmseg.datasets import DATASETS as SEG_DATASETS
from .simi_dataset import SimiCustom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class SimiScanNet3DDataset(SimiCustom3DDataset):
    r"""ScanNet two streams Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 label_list_file,
                 labeled_ratio=0.2,
                 ratio=2,
                 pipeline=None,
                 pipeline_weakly=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            label_list_file=label_list_file,
            labeled_ratio=labeled_ratio,
            ratio=ratio,
            pipeline=pipeline,
            pipeline_weakly=pipeline_weakly,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_unlabel_data_info(self, index):
        """Get unlabeled data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.unlabel_data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)

        if not self.test_mode:
            annos = self.get_unlabel_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for \
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def get_unlabel_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for \
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.unlabel_data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def prepare_label_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        input_dict_weakly = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        self.pre_pipeline(input_dict_weakly)
        example = self.pipeline(input_dict)
        example_weakly = self.pipeline_weakly(input_dict_weakly)
        example['use_label'] = True
        example_weakly['use_label'] = True
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        if self.filter_empty_gt and \
                (example_weakly is None or
                    ~(example_weakly['gt_labels_3d']._data != -1).any()):
            return None
        return self.combine_data(example, example_weakly)

    def prepare_unlabel_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_unlabel_data_info(index)
        input_dict_weakly = self.get_unlabel_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        self.pre_pipeline(input_dict_weakly)
        example = self.pipeline(input_dict)
        example_weakly = self.pipeline_weakly(input_dict_weakly)
        example['use_label'] = False
        example_weakly['use_label'] = False
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        if self.filter_empty_gt and \
                (example_weakly is None or
                    ~(example_weakly['gt_labels_3d']._data != -1).any()):
            return None
        return self.combine_data(example, example_weakly)

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        label_data = self.prepare_label_data(index)
        if len(self.unlabel_data_infos)!=0:
            unlabel_inds = random.sample(range(0, len(self.unlabel_data_infos)), self.ratio)
            unlabel_datas = []
            for unlabel_ind in unlabel_inds:
                unlabel_data = self.prepare_unlabel_data(unlabel_ind)
                unlabel_datas.append(unlabel_data)
            example = {"label_data":label_data,"unlabel_datas":unlabel_datas,'img_metas':label_data['img_metas_s']}
        else:
            example = {"label_data":label_data,'img_metas':label_data['img_metas_s']}
        return example

    def combine_data(self, example, example_weakly):
        results = {}
        for key, val in example.items():
            results[key+'_s'] = val
            results[key+'_t'] = example_weakly[key]
        return results

    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need \
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        input_dict['ann_info'] = dict(
            axis_align_matrix=self._get_axis_align_matrix(
                self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result(points, gt_bboxes, gt_labels,
                        pred_bboxes, pred_labels, out_dir, file_name, False)

