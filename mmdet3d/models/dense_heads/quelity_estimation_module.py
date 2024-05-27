""" Grid interpolation and convolution module for IoU estimation
Written by Yezhen Cong, 2020
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import three_nn

class QualityEstimation(nn.Module):
    def __init__(
        self,
        num_class,
        num_heading_bin,
        num_size_cluster,
        mean_size_arr_path,
        num_proposal,
        sampling,
        seed_feat_dim=256,
        query_feats="seed",
        iou_class_depend=True,
    ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = np.load(mean_size_arr_path)["arr_0"]
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.query_feats = query_feats
        self.iou_class_depend = iou_class_depend
        self.reg_topk = 4
        self.grid_size = 3
        self.left_mask = [
            i // self.grid_size * self.grid_size * self.grid_size + i % self.grid_size
            for i in range(self.grid_size * self.grid_size)
        ]
        self.right_mask = [
            i // self.grid_size * self.grid_size * self.grid_size
            + i % self.grid_size
            + self.grid_size * (self.grid_size - 1)
            for i in range(self.grid_size * self.grid_size)
        ]

        # class dependent IoU
        self.iou_size = num_class if self.iou_class_depend else 1

        self.mlps_before = []
        self.mlps_head = []
        # side 
        for _ in range(6):
            self.mlps_before.append(MiniPointNet(self.seed_feat_dim + 3, 128, hide_dim=128))
            self.mlps_head.append(
                nn.Sequential(
                    nn.Conv1d(128 + 33 + 4 + 1, 128, 1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, self.iou_size, 1),
                )
            )
        # IoU
        self.mlps_head.append(
            nn.Sequential(
                nn.Conv1d((128 + 33 + 4 + 1) * 6, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, self.iou_size * 2 + 2, 1),
            )
        )
        self.mlps_before = nn.ModuleList(self.mlps_before)
        self.mlps_head = nn.ModuleList(self.mlps_head)

    def extract_features(self, end_points):
        origin_xyz = end_points["seed_points"].detach().contiguous()
        origin_features = end_points["seed_features"].detach().contiguous()
        return origin_xyz, origin_features

    def generate_grid(self, size: torch.tensor):
        """generate grids for every side

        Args:
            center (torch.tensor): B K 3
            size (torch.tensor): B K 3
            heading (torch.tensor): B K

        Returns:
            whole_grid: B K grid_size *  grid_size * grid_size 3
        """
        B, K = size.shape[:2]

        grid_step = torch.linspace(-1, 1, self.grid_size).cuda()
        grid_step_x = grid_step.view(self.grid_size, 1, 1).repeat(
            1, self.grid_size, self.grid_size
        )
        grid_step_x = grid_step_x.view(1, 1, -1).expand(B, K, -1)
        grid_step_y = grid_step.view(1, self.grid_size, 1).repeat(
            self.grid_size, 1, self.grid_size
        )
        grid_step_y = grid_step_y.view(1, 1, -1).expand(B, K, -1)
        grid_step_z = grid_step.view(1, 1, self.grid_size).repeat(
            self.grid_size, self.grid_size, 1
        )
        grid_step_z = grid_step_z.view(1, 1, -1).expand(B, K, -1)

        x_grid = grid_step_x * size[:, :, 0:1] / 2
        y_grid = grid_step_y * size[:, :, 1:2] / 2
        z_grid = grid_step_z * size[:, :, 2:3] / 2

        whole_grid = torch.cat(
            [x_grid.unsqueeze(-1), y_grid.unsqueeze(-1), z_grid.unsqueeze(-1)], dim=-1
        )

        return whole_grid

    def grid_for_side(
        self, whole_grid: torch.tensor, center: torch.tensor, heading: torch.tensor
    ):
        """generate grids for every side

        Args:
            whole_grid (torch.tensor): B K grid_size *  grid_size * grid_size 3
            center (torch.tensor): B K 3
            heading (torch.tensor): B K

        Returns:
            side_grid: B K grid_size *  grid_size * 6 3
        """
        B, K = center.shape[:2]

        front_grid = whole_grid[:, :, 0 : self.grid_size * self.grid_size, :]
        back_grid = whole_grid[:, :, -self.grid_size * self.grid_size :, :]
        top_grid = whole_grid[:, :, self.grid_size - 1 :: self.grid_size, :]
        down_grid = whole_grid[:, :, :: self.grid_size, :]
        left_grid = whole_grid[:, :, self.left_mask, :]
        right_grid = whole_grid[:, :, self.right_mask, :]

        front_grid_zero = front_grid * 0.1
        front_grid_zero[..., 1:] = 0.0
        front_grid_jitter = torch.cat((front_grid - front_grid_zero, front_grid, front_grid + front_grid_zero), dim=-2)

        back_grid_zero = back_grid * 0.1
        back_grid_zero[..., 1:] = 0.0
        back_grid_jitter = torch.cat((back_grid - back_grid_zero, back_grid, back_grid + back_grid_zero), dim=-2)

        top_grid_zero = top_grid * 0.1
        top_grid_zero[..., :2] = 0.0
        top_grid_jitter = torch.cat((top_grid - top_grid_zero, top_grid, top_grid + top_grid_zero), dim=-2)

        down_grid_zero = down_grid * 0.1
        down_grid_zero[..., :2] = 0.0
        down_grid_jitter = torch.cat((down_grid - down_grid_zero, down_grid, down_grid + down_grid_zero), dim=-2)

        left_grid_zero = left_grid * 0.1
        left_grid_zero[..., ::2] = 0.0
        left_grid_jitter = torch.cat((left_grid - left_grid_zero, left_grid, left_grid + left_grid_zero), dim=-2)

        right_grid_zero = right_grid * 0.1
        right_grid_zero[..., ::2] = 0.0
        right_grid_jitter = torch.cat((right_grid - right_grid_zero, right_grid, right_grid + right_grid_zero), dim=-2)

        side_grid = torch.cat(
            [front_grid_jitter, back_grid_jitter, top_grid_jitter, down_grid_jitter, left_grid_jitter, right_grid_jitter], dim=-2
        )

        rot_mat = rot_gpu(heading).view(-1, 3, 3)  # [B * S, 3, 3]
        side_grid = torch.bmm(
            side_grid.view(B * K, -1, 3), rot_mat.transpose(1, 2)
        ).view(B, K, -1, 3)
        side_grid = side_grid + center.unsqueeze(2).expand(
            -1, -1, side_grid.shape[2], -1
        )
        return side_grid

    def grid_for_bbox(
        self, whole_grid: torch.tensor, center: torch.tensor, heading: torch.tensor
    ):
        """generate grids for every side

        Args:
            whole_grid (torch.tensor): B K grid_size *  grid_size * grid_size 3
            center (torch.tensor): B K 3
            heading (torch.tensor): B K

        Returns:
            side_grid: B K grid_size *  grid_size * grid_size 3
        """
        B, K = center.shape[:2]

        rot_mat = rot_gpu(heading).view(-1, 3, 3)  # [B * S, 3, 3]
        bbox_grid = torch.bmm(
            whole_grid.view(B * K, -1, 3), rot_mat.transpose(1, 2)
        ).view(B, K, -1, 3)
        bbox_grid = bbox_grid + center.unsqueeze(2).expand(
            -1, -1, bbox_grid.shape[2], -1
        )
        return bbox_grid

    def grid_features(
        self,
        origin_xyz: torch.tensor,
        origin_features: torch.tensor,
        whole_grid: torch.tensor,
        center: torch.tensor,
    ):
        """interpolated features from origin point clouds

        Args:
            origin_xyz (torch.tensor): B N_points 3
            origin_features (torch.tensor): B C N_points
            whole_grid (torch.tensor): B K*N_grids 3
            center (torch.tensor): B K 3

        Returns:
            _type_: B, C, K, N_grids
        """
        B, K = center.shape[:2]
        feat_size = origin_features.shape[1]
        grid_size = whole_grid.shape[1] // K
        _, idx = three_nn(whole_grid, origin_xyz)  # B, K*96, 3

        interp_points = torch.gather(
            origin_xyz, dim=1, index=idx.view(B, -1, 1).expand(-1, -1, 3).long()
        )  # B, K*96*3, 3
        expanded_whole_grid = (
            whole_grid.unsqueeze(2).expand(-1, -1, 3, -1).contiguous().view(B, -1, 3)
        )  # B, K*96*3, 3
        dist = interp_points - expanded_whole_grid
        dist = torch.sqrt(torch.sum(dist * dist, dim=2))
        relative_grid = whole_grid - center.unsqueeze(2).expand(
            -1, -1, grid_size, -1
        ).contiguous().view(B, -1, 3)

        weight = 1 / (dist + 1e-8)
        weight = weight.view(B, -1, 3)
        norm = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / norm
        weight = weight.contiguous()

        interpolated_feats = torch.cat(
            [
                torch.index_select(a, 0, i).unsqueeze(0)
                for a, i in zip(origin_features.transpose(1, 2), idx.view(B, -1).long())
            ],
            0,
        )
        interpolated_feats = torch.sum(
            interpolated_feats.view(B, -1, 3, feat_size) * weight.unsqueeze(-1), dim=2
        )
        interpolated_feats = interpolated_feats.transpose(1, 2)
        interpolated_feats = interpolated_feats.view(B, -1, K, grid_size)
        interpolated_feats = torch.cat(
            [
                relative_grid.transpose(1, 2).contiguous().view(B, -1, K, grid_size),
                interpolated_feats,
            ],
            dim=1,
        )
        return interpolated_feats

    def dist_feature(self, end_points, prefix=''):
        """distribution feature

        Args:
            end_points (dict): prob [12, 6, 33, 256]
        """
        prob = end_points[f"{prefix}bbox_probs"].detach()
        stat = (
            torch.cat(
                [
                    prob,
                    prob.topk(self.reg_topk, dim=2)[0],
                    prob.var(dim=2, keepdim=True),
                ],
                dim=2,
            )
            .permute(1, 0, 2, 3)
            .repeat(1, 1, 1, 2)
        )
        return stat

    def forward(
        self,
        center: torch.tensor,
        size: torch.tensor,
        heading: torch.tensor,
        end_points: dict,
        prefix='',
    ):
        """_summary_

        Args:
            center (torch.tensor): B K 3
            size (torch.tensor): B K 3
            heading (torch.tensor): B K
            end_points (dict): dict

        Returns:
            end_points: dict
        """

        B, K = size.shape[:2]
        origin_xyz, origin_features = self.extract_features(end_points)

        # generate grid
        whole_grid = self.generate_grid(size)
        side_grid = self.grid_for_side(whole_grid, center, heading)
        # bbox_grid = self.grid_for_bbox(whole_grid, center, heading)

        # interpolate features for the side grids
        side_grid = side_grid.view(B, -1, 3).contiguous()
        # bbox_grid = bbox_grid.view(B, -1, 3).contiguous()
        origin_xyz = origin_xyz.contiguous()
        origin_features = origin_features.contiguous()
        side_interpolated_feats = self.grid_features(
            origin_xyz, origin_features, side_grid, center
        )
        side_interpolated_feats = torch.split(
            side_interpolated_feats, self.grid_size * self.grid_size * 3, dim=-1
        )

        # get the distribution features
        dist_feature = self.dist_feature(end_points, prefix)

        # get the geometric features
        end_points[f"{prefix}side_scores"] = []
        side_feature_list = []
        for i in range(6):
            interpolated_feats = side_interpolated_feats[i]
            interpolated_feats = self.mlps_before[i](interpolated_feats)
            interpolated_feats = torch.cat((interpolated_feats, dist_feature[i]), dim=1)
            side_feature_list.append(interpolated_feats)
            end_points[f"{prefix}side_scores"].append(self.mlps_head[i](interpolated_feats))
        
        end_points[f"{prefix}side_scores"] = torch.stack(end_points[f"{prefix}side_scores"], 0)
        side_features = torch.cat(side_feature_list, dim=1)
        global_scores = self.mlps_head[6](side_features).transpose(2, 1)
        end_points[f"{prefix}iou_scores"] = global_scores[..., :self.iou_size]
        end_points[f"{prefix}rotate_scores"] = global_scores[..., self.iou_size:self.iou_size*2]
        end_points[f"{prefix}R_obj_scores"] = global_scores[..., self.iou_size*2:]
        return end_points


def rot_gpu(t):
    """Rotation about the upright axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3])).cuda()
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 1] = s
    output[..., 2, 2] = 1
    output[..., 1, 0] = -s
    output[..., 1, 1] = c
    return output


class MiniPointNet(nn.Module):
    def __init__(self, channels: int, feature_dim: int, hide_dim=256):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(channels, hide_dim, 1, bias=False),
            nn.BatchNorm2d(hide_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_dim, hide_dim//2, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv2d(hide_dim, hide_dim, 1, bias=False),
            nn.BatchNorm2d(hide_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_dim, feature_dim, 1),
        )

    def forward(self, points) -> torch.Tensor:
        # points: (B, N, C)
        feature = self.first_conv(points)  # (B, 256, N_p, N)
        feature_global = torch.max(feature, dim=-1, keepdim=True).values  # (B, 256, N_p, 1)
        # concating global features to each point features
        feature = torch.cat(
            [feature_global.expand(-1, -1, -1, feature.shape[-1]), feature], dim=1
        )  # (B, 512, N_p, N)
        feature = self.second_conv(feature)  # (B, feature_dim, N_p, N)
        feature_global = torch.max(feature, dim=-1).values  # (B, feature_dim, N_p)
        return feature_global