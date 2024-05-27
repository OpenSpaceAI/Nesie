# model settings
model = dict(
    type="VoteNetSAQE",
    backbone=dict(
        type="PointNet2SASSG",
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256), (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type="BN2d"),
        sa_cfg=dict(
            type="PointSAModule", pool_mod="max", use_xyz=True, normalize_xyz=True
        ),
    ),
    bbox_head=dict(
        type="SAQEHead",
        num_classes=18,
        reg_max=32,
        alpha=1.0,
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type="Conv1d"),
            norm_cfg=dict(type="BN1d"),
            norm_feats=True,
            vote_loss=dict(
                type="ChamferDistance",
                mode="l1",
                reduction="none",
                loss_dst_weight=10.0,
            ),
        ),
        vote_aggregation_cfg=dict(
            type="PointSAModule",
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True,
        ),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True
        ),
        objectness_loss=dict(
            type="CrossEntropyLoss",
            class_weight=[0.2, 0.8],
            reduction="sum",
            loss_weight=5.0,
        ),
        center_loss=dict(
            type="ChamferDistance",
            mode="l2",
            reduction="sum",
            loss_src_weight=10.0,
            loss_dst_weight=10.0,
        ),
        iou_loss=dict(type="IoU3DLoss", reduction="sum", loss_weight=3.0),
        semantic_loss=dict(type="CrossEntropyLoss", reduction="sum", loss_weight=1.0),
        iou_pred_loss=dict(
            type="GeneralQualityFocalLoss",
            reduction="sum",
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0,
        ),
        surface_loss=dict(
            type="SurfaceLoss",
            func_type="MSELoss",
            beta=5.0,
            reduction="sum",
            loss_weight=10.0,
        ),
        angle_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        angle_pred_loss=dict(
            type='MSELoss', reduction='sum', loss_weight=1.0),
        side_loss=dict(
            type="SidePredLoss",
            label_func_type="SmoothL1Loss",
            loss_func_type="MSELoss",
            beta=5.0,
            reduction="sum",
            loss_weight=1.0,
        ),
        grid_conv_cfg=dict(
            num_class=18,
            num_heading_bin=1,
            num_size_cluster=18,
            mean_size_arr_path="data/scannet/meta_data/scannet_means.npz",
            num_proposal=256,
            sampling="seed_fps",
            query_feats="seed",
        ),
    ),
    custom_config=[
        dict(
            type="SimiTeacherHook",
            momentum=0.001,
            interval=1,
            warm_up=10,
            resume_from=None,
        )
    ],
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=0.3,
        neg_distance_thr=0.6,
        sample_mod="vote",
        dataset_name="ScanNet",
        thresh_warmup=True,
        use_cbl=True,
    ),
    test_cfg=dict(
        sample_mod="seed",
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True,
        dataset_name="ScanNet",
        use_iou_for_nms=True,
        iou_opt=False,
        add_info=True,
        opt_rate=5e-4,
        opt_step=10,
    ),
)


lr = 0.008  # max learning rate
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy="step", warmup=None, step=[24, 32])
# runtime settings
runner = dict(type="SimiEpochBasedRunner", max_epochs=36)
custom_hooks = [
    dict(type="SimiRunnerHook", interval=1, by_epoch=True, save_optimizer=True)
]

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
load_from = None
resume_from = None
workflow = [("train", 1)]


# dataset settings
dataset_type = "ScanNetDataset"
simi_dataset_type = "SimiScanNet3DDataset"
data_root = "/data1/Dataset/ScanNet_multi_modal/"
class_names = (
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "showercurtrain",
    "toilet",
    "sink",
    "bathtub",
    "garbagebin",
)
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True,
    ),
    dict(type="GlobalAlignment", rotation_axis=2),
    dict(
        type="PointSegClassMapping",
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39),
        max_cat_id=40,
    ),
    dict(type="IndoorPointSample", num_points=40000),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.1415926 / 36, 3.1415926 / 36],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=True,
    ),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "points",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
        ],
    ),
]
train_pipeline_weakly = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True,
    ),
    dict(type="GlobalAlignment", rotation_axis=2),
    dict(
        type="PointSegClassMapping",
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39),
        max_cat_id=40,
    ),
    dict(type="IndoorPointSample", num_points=40000),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[0, 0],
        scale_ratio_range=[1.0, 1.0],
        translation_std=[0, 0, 0],
        shift_height=False,
    ),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "points",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
        ],
    ),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
    ),
    dict(type="GlobalAlignment", rotation_axis=2),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(
                type="RandomFlip3D",
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
            ),
            dict(type="IndoorPointSample", num_points=40000),
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2],
    ),
    dict(type="GlobalAlignment", rotation_axis=2),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["points"]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=20,
        dataset=dict(
            type=simi_dataset_type,
            data_root=data_root,
            ann_file=data_root + "scannet_infos_train.pkl",
            label_list_file="data/scannet/meta_data/scannetv2_train_0.1.txt",
            ratio=2,
            pipeline=train_pipeline,
            pipeline_weakly=train_pipeline_weakly,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d="Depth",
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "scannet_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d="Depth",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "scannet_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d="Depth",
    ),
)

evaluation = dict(pipeline=eval_pipeline)
