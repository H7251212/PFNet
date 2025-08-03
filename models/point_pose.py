
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
from .model_utils import *   # 包含 index_points, knn_point, square_distance 等函数

# ---------------- FPS + k-NN ----------------
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        # xyz: [B, N, 3] ; x: [B, N, C]
        B, N, _ = xyz.shape
        fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)
        return lc_xyz, lc_x, knn_xyz, knn_x

# ---------------- Gaussian Positional Encoding for Raw-point Embedding ----------------
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        """
        in_dim : 输入维度（通常为3，对应 x,y,z）
        out_dim: 输出维度，必须能被 in_dim 整除
        sigma  : 高斯核标准差
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.num_ref = out_dim // in_dim
        self.register_buffer('ref', torch.linspace(-1, 1, steps=self.num_ref).view(1, self.num_ref))
        
    def forward(self, xyz):
        # xyz: [B, in_dim, N]
        B, D, N = xyz.shape
        ref = self.ref.expand(B, -1)  # [B, num_ref]
        outs = []
        for d in range(D):
            coord = xyz[:, d:d+1, :]                # [B, 1, N]
            coord_exp = coord.expand(-1, self.num_ref, -1)
            ref_exp = ref.unsqueeze(-1)
            embed = torch.exp(- ((coord_exp - ref_exp)**2) / (2 * (self.sigma ** 2)))
            outs.append(embed)
        position_embed = torch.cat(outs, dim=1)        # [B, D*num_ref, N]
        return position_embed

# ---------------- Multi-Scale Gaussian Positional Encoding for Local Geometry ----------------
class PosE_Geo_MS(nn.Module):
    def __init__(self, in_dim, out_dim, sigma_list):
        """
        in_dim : 输入坐标维度（通常为3）
        out_dim: 输出维度，要求为 in_dim*num_ref, 这里num_ref = out_dim/in_dim 对每个尺度保持一致
        sigma_list: 一个列表，包含多个尺度的sigma值，例如 [0.4, 0.6, 0.8]
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_list = sigma_list
        self.num_ref = out_dim // in_dim  
        # 对每个尺度初始化参考点（保持一致即可）
        self.register_buffer('ref', torch.linspace(-1, 1, steps=self.num_ref).view(1, self.num_ref))
        # 可学习的融合权重，用 1x1 卷积融合多尺度编码，输入通道 = len(sigma_list)*in_dim*num_ref
        self.fuse_conv = nn.Conv2d(len(sigma_list)*in_dim*self.num_ref, in_dim*self.num_ref, kernel_size=1)

    def forward(self, knn_xyz, knn_x):
        # knn_xyz: [B, in_dim, G, K]
        # knn_x  : [B, C, G, K]
        B, D, G, K = knn_xyz.shape
        ref = self.ref.expand(B, -1)  # [B, num_ref]
        multi_scale_encodings = []
        for sigma in self.sigma_list:
            outs = []
            for d in range(D):
                coord = knn_xyz[:, d:d+1, :, :]                # [B, 1, G, K]
                coord_exp = coord.expand(-1, self.num_ref, -1, -1)  # [B, num_ref, G, K]
                ref_exp = ref.unsqueeze(-1).unsqueeze(-1)        # [B, num_ref, 1, 1]
                embed = torch.exp(- ((coord_exp - ref_exp)**2) / (2 * (sigma ** 2)))
                outs.append(embed)
            # 合并每个坐标通道，shape: [B, in_dim*num_ref, G, K]
            multi_scale_encodings.append(torch.cat(outs, dim=1))
        # 拼接所有尺度，得到 shape: [B, len(sigma_list)*in_dim*num_ref, G, K]
        concat_enc = torch.cat(multi_scale_encodings, dim=1)
        # 融合多尺度编码（这里加入可学习的1x1卷积）
        pos_embed = self.fuse_conv(concat_enc)
        # 融合方式可以采用加法（也可在此处尝试其他融合方式）
        knn_x_w = knn_x + pos_embed
        return knn_x_w

# ---------------- Local Geometry Aggregation ----------------
class LGA(nn.Module):
    def __init__(self, out_dim, sigma_list):
        super().__init__()
        # 使用多尺度位置编码
        self.geo_extract = PosE_Geo_MS(3, out_dim, sigma_list)
        # 增加一个局部归一化层：对每个group的邻域内数据做归一化（例如LayerNorm）
        self.lnorm = nn.LayerNorm(out_dim)  # 注意：输入形状需要匹配，这里后续会reshape

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # knn_x: [B, G, K, C] 以及 knn_xyz: [B, G, K, 3]
        B, G, K, C = knn_x.shape
        # 对每个局部邻域进行归一化：沿 "K" 维度独立归一化
        knn_x = knn_x - knn_x.mean(dim=2, keepdim=True)
        knn_x = knn_x / (knn_x.std(dim=2, keepdim=True) + 1e-5)
        knn_xyz = knn_xyz - knn_xyz.mean(dim=2, keepdim=True)
        knn_xyz = knn_xyz / (knn_xyz.std(dim=2, keepdim=True) + 1e-5)
        
        # 扩展局部中心特征到每个邻域，并与邻域特征拼接
        knn_x = torch.cat([knn_x, lc_x.view(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # [B, G, K, C']
        # 调整排列，适应后续卷积，变成[B, C', G, K]
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        # 融合多尺度位置编码信息（本质上结合了几何信息和邻域特征）
        knn_x_w = self.geo_extract(knn_xyz, knn_x)
        # 对融合后的特征在每个局部邻域进行归一化（LayerNorm要求最后为 [B, G, K, C]）
        knn_x_w = knn_x_w.permute(0, 2, 3, 1).contiguous()  # [B, G, K, C_new]
        # 将最后一维作为通道进行归一化
        knn_x_w = self.lnorm(knn_x_w)
        # 恢复格式 [B, C_new, G, K]
        knn_x_w = knn_x_w.permute(0, 3, 1, 2)
        return knn_x_w

# ---------------- Pooling模块 ----------------
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, knn_x_w):
        # knn_x_w: [B, C, G, K] 对“K”方向做max和mean聚合
        lc_x = knn_x_w.max(dim=-1)[0] + knn_x_w.mean(dim=-1)
        lc_x = self.out_transform(lc_x)
        return lc_x

# ---------------- Non-Parametric Encoder ----------------

class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, sigma):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        
        # 如果 sigma 不是列表，则转换为列表
        if not isinstance(sigma, list):
            sigma_list = [sigma]
        else:
            sigma_list = sigma

        self.raw_point_embed = PosE_Initial(3, self.embed_dim, sigma_list[0])  # 初始编码使用其中一个尺度

        self.FPS_kNN_list = nn.ModuleList()
        self.LGA_list = nn.ModuleList()
        self.Pooling_list = nn.ModuleList()

        out_dim = self.embed_dim
        group_num = self.input_points
        for i in range(self.num_stages):
            out_dim = out_dim * 2       # 每阶段翻倍
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, sigma_list))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):
        x = self.raw_point_embed(x[:, :3, :])
        xyz_list = [xyz]
        x_list = [x]
        for i in range(self.num_stages):
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            x = self.Pooling_list[i](knn_x_w)
            xyz_list.append(xyz)
            x_list.append(x)
        return xyz_list, x_list



# ---------------- Non-Parametric Decoder ----------------
class DecNP(nn.Module):  
    def __init__(self, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors

    def propagate(self, xyz1, xyz2, points1, points2):
        # xyz1: [B, N, 3]; xyz2: [B, S, 3]
        # points2: [B, C, S] -> 先转为 [B, S, C]
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)
            interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)  # 恢复至 [B, C, N]
        return new_points

    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(self.num_stages):
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
        return x

# ---------------- Non-Parametric Segmentation Network (编码器+解码器) ----------------
class Point_NN_Seg(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, embed_dim=144, 
                 k_neighbors=128, de_neighbors=6, sigma=0.6):
        super().__init__()
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors, sigma)
        self.DecNP = DecNP(num_stages, de_neighbors)

    def forward(self, xyz):
        # xyz: [B, C, N]，假设前三个通道为坐标
        xyz_in = xyz.permute(0, 2, 1)  # [B, N, C]
        xyz_list, x_list = self.EncNP(xyz_in, xyz)
        x_out = self.DecNP(xyz_list, x_list)
        return x_out  # 每个点的最终特征表示

# ---------------- 非参数分割头 ----------------
class NonParamSegHead(nn.Module):
    def __init__(self, in_channels, num_parts=18):
        super().__init__()
        self.num_parts = num_parts
        # 固定生成原型，这里随机初始化后归一化；实际可根据先验信息设置
        prototypes = torch.randn(num_parts, in_channels)
        prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)
        self.register_buffer("prototypes", prototypes)

    def forward(self, features):
        # features: [B, in_channels, N]
        f_norm = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-6)
        scores = torch.einsum("pc, bcn -> bpn", self.prototypes, f_norm)
        seg_labels = scores.argmax(dim=1)  # [B, N]
        return seg_labels

# ---------------- 非参数位姿估计模块 ----------------
class NonParamPoseEstimator(nn.Module):  
    def __init__(self, num_parts=18):
        super().__init__()
        self.num_parts = num_parts

    def forward(self, xyz, seg_labels):
        # xyz: [B, N, 3]；seg_labels: [B, N]
        B, N, _ = xyz.shape
        device = xyz.device
        poses = []
        for b in range(B):
            bone_positions = []
            for label in range(self.num_parts):
                mask = (seg_labels[b] == label)  # [N]
                if mask.sum() > 0:
                    part_xyz = xyz[b][mask]  # [n_points, 3]
                    center = part_xyz.mean(dim=0)
                    distances = torch.norm(part_xyz - center.unsqueeze(0), dim=1)
                    median_dist = distances.median()
                    filtered = part_xyz[distances <= median_dist]
                    if filtered.shape[0] > 0:
                        bone_pos = filtered.mean(dim=0)
                    else:
                        bone_pos = center
                else:
                    bone_pos = torch.zeros(3, device=device)
                bone_positions.append(bone_pos)
            poses.append(torch.stack(bone_positions, dim=0))
        poses = torch.stack(poses, dim=0)   # [B, num_parts, 3]
        return poses

# ---------------- 整体无参位姿估计网络 ----------------
class NonParamPoseNet(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, embed_dim=144, 
                 k_neighbors=128, de_neighbors=6, sigma=0.6, num_parts=18):
        super().__init__()
        self.seg_net = Point_NN_Seg(input_points, num_stages, embed_dim, k_neighbors, de_neighbors, sigma)
        # 这里 in_channels = embed_dim * (2**(num_stages+1) - 1)，可根据编码器输出特征拼接策略调整
        self.seg_head = NonParamSegHead(in_channels=embed_dim * (2**(num_stages+1) - 1), num_parts=num_parts)
        self.pose_estimator = NonParamPoseEstimator(num_parts=num_parts)

    def forward(self, x):
        # x: [B, C, N]，假设前三个通道为坐标
        features = self.seg_net(x)           # features: [B, in_channels, N]
        seg_labels = self.seg_head(features)   # [B, N]
        xyz = x[:, :3, :].permute(0, 2, 1)      # [B, N, 3]
        poses = self.pose_estimator(xyz, seg_labels)  # [B, num_parts, 3]
        return seg_labels, poses

