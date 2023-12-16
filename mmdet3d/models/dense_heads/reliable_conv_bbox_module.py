from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_conv_layer
from mmcv.runner import BaseModule
from torch import nn as nn
import torch
from mmdet.models.builder import HEADS


@HEADS.register_module()
class ReliableConvBboxHead(BaseModule):
    r"""More general bbox head, with shared conv layers and two optional
    separated branches.

    .. code-block:: none

                     /-> cls convs -> cls_score
        shared convs
                     \-> bbox convs -> bbox_pred[:n]
                     \-> heading convs -> bbox_pred[n:]
    """

    def __init__(self,
                 in_channels=0,
                 shared_conv_channels=(),
                 cls_conv_channels=(),
                 num_cls_out_channels=0,
                 bbox_conv_channels=(),
                 num_bbox_out_channels=0,
                 heading_conv_channels=(),
                 num_heading_out_channels=0,
                 reg_max=16,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ReliableConvBboxHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        assert in_channels > 0
        assert num_cls_out_channels > 0
        assert num_bbox_out_channels > 0
        assert num_heading_out_channels > 0
        self.in_channels = in_channels
        self.shared_conv_channels = shared_conv_channels
        self.cls_conv_channels = cls_conv_channels
        self.num_cls_out_channels = num_cls_out_channels
        self.bbox_conv_channels = bbox_conv_channels
        self.num_bbox_out_channels = num_bbox_out_channels
        self.heading_conv_channels = heading_conv_channels
        self.num_heading_out_channels = num_heading_out_channels
        self.reg_max = reg_max
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias

        # add shared convs
        if len(self.shared_conv_channels) > 0:
            self.shared_convs = self._add_conv_branch(
                self.in_channels, self.shared_conv_channels)
            out_channels = self.shared_conv_channels[-1]
        else:
            out_channels = self.in_channels

        # add cls specific branch
        prev_channel = out_channels
        if len(self.cls_conv_channels) > 0:
            self.cls_convs = self._add_conv_branch(prev_channel,
                                                   self.cls_conv_channels)
            prev_channel = self.cls_conv_channels[-1]

        self.conv_cls = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_cls_out_channels,
            kernel_size=1)
        # add bbox specific branch
        prev_channel = out_channels
        if len(self.bbox_conv_channels) > 0:
            self.bbox_convs = self._add_conv_branch(prev_channel,
                                                   self.bbox_conv_channels)
            prev_channel = self.bbox_conv_channels[-1]

        self.conv_bbox = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_bbox_out_channels,
            kernel_size=1)
        # add heading specific branch
        prev_channel = out_channels
        if len(self.heading_conv_channels) > 0:
            self.heading_convs = self._add_GN_conv_branch(prev_channel,
                                                   self.heading_conv_channels, self.reg_max)
            prev_channel = self.heading_conv_channels[-1]

        self.conv_heading = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_heading_out_channels,
            kernel_size=1)

    def _add_conv_branch(self, in_channels, conv_channels):
        """Add shared or separable branch."""
        conv_spec = [in_channels] + list(conv_channels)
        # add branch specific conv layers
        conv_layers = nn.Sequential()
        for i in range(len(conv_spec) - 1):
            conv_layers.add_module(
                f'layer{i}',
                ConvModule(
                    conv_spec[i],
                    conv_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return conv_layers

    def _add_GN_conv_branch(self, in_channels, conv_channels, reg_max):
        """Add shared or separable branch."""
        conv_spec = [in_channels] + list(conv_channels)
        # add branch specific conv layers
        conv_layers = nn.Sequential()
        for i in range(len(conv_spec) - 1):
            conv_layers.add_module(
                f'layer{i}',
                ConvModule(
                    conv_spec[i],
                    conv_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=dict(type='GN', num_groups=reg_max, requires_grad=True),
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return conv_layers

    def forward(self, feats):
        """Forward.

        Args:
            feats (Tensor): Input features

        Returns:
            Tensor: Class scores predictions
            Tensor: Regression predictions
        """
        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(feats)

        # separate branches
        x_cls = x
        x_bbox = x
        x_heading = x

        if len(self.cls_conv_channels) > 0:
            x_cls = self.cls_convs(x_cls)
        cls_score = self.conv_cls(x_cls)

        if len(self.bbox_conv_channels) > 0:
            x_bbox = self.bbox_convs(x_bbox)
        bbox_pred_bbox = self.conv_bbox(x_bbox)

        if len(self.heading_conv_channels) > 0:
            x_heading = self.heading_convs(x_heading)
        bbox_pred_heading = self.conv_heading(x_heading)

        bbox_pred = torch.cat((bbox_pred_bbox, bbox_pred_heading), dim = 1)

        return cls_score, bbox_pred
