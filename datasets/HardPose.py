#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    """
    对点云进行归一化处理：
       1. 先减去点云的质心
       2. 再除以所有点到原点的最大距离，达到尺度归一化
       返回:
         - 归一化后的点云
         - 质心 (用于反归一化)
         - 最大距离 (缩放因子)
    """
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_centered ** 2, axis=1)))
    pc_normalized = pc_centered / m if m > 0 else pc_centered
    return pc_normalized, centroid, m

class HardPoseMultiSubjectDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False, fixed_seed=True, return_normalization_params=True):
        """
        参数说明:
          npoints   : 每个样本点云随机采样的点数
          split     : 数据集类型，可选 'train', 'test', 或 'valid'
          normalize : 是否对点云进行归一化处理
          fixed_seed: 是否使用固定种子确保采样的可重复性
          return_normalization_params: 是否返回归一化参数（质心和尺度）
        """
        self.npoints = npoints
        self.normalize = normalize
        self.fixed_seed = fixed_seed
        self.return_normalization_params = return_normalization_params

        # 根据 split 参数设置对应的文件夹路径和允许的被试编号
        if split == 'train':
            self.root_dir = r"data/hard_pose/output_train_txt_cloud"
            # 训练数据：被试编号 S1 ~ S60，但没有 S6
            self.allowed_subjects = {"S{}".format(i) for i in range(1, 61) if i != 6}
        elif split == 'test':
            self.root_dir = r"/data/coding/upload-data/data/Point-NN-main/data/hard_pose/output_test_txt_cloud"
            # 测试数据：被试编号 S1 ~ S20，但没有 S6
            self.allowed_subjects = {"S{}".format(i) for i in range(1, 21) if i != 6}
        elif split == 'valid':
            self.root_dir = r"data/hard_pose/output_valid_txt_cloud"
            # 验证数据：被试编号 S1 ~ S19，但没有 S6
            self.allowed_subjects = {"S{}".format(i) for i in range(1, 20) if i != 6}
        else:
            raise ValueError("split 参数错误，请选择 'train', 'test' 或 'valid'")

        # 构造数据文件列表，文件名格式例如：
        # hard_pose_train_S19_sample1001.txt
        # 文件列表中同时存储文件路径和被试编号，用元组 (file_path, subject) 表示
        self.datapaths = []
        pattern = re.compile(r"hard_pose_{}_((S\d+))_sample\d+\.txt".format(split))
        for fn in sorted(os.listdir(self.root_dir)):
            m = pattern.match(fn)
            if m:
                subject_id = m.group(1)  # 如 "S19"
                if subject_id in self.allowed_subjects:
                    full_path = os.path.join(self.root_dir, fn)
                    self.datapaths.append((full_path, subject_id))

        if len(self.datapaths) == 0:
            raise RuntimeError("在 {} 目录下没有找到符合条件的数据文件，请检查文件路径和文件名格式。".format(self.root_dir))

        # 可选：缓存加载的数据以加快后续读取速度
        self.cache = {}
        self.cache_size = 20000

        print(f"数据集初始化完成: {split} split, 总样本数: {len(self.datapaths)}")
        print(f"固定种子模式: {'开启' if self.fixed_seed else '关闭'}")
        print(f"返回归一化参数: {'是' if self.return_normalization_params else '否'}")

    def __len__(self):
        return len(self.datapaths)

    def get_filename(self, index):
        """获取文件名，用于匹配GT"""
        file_path, _ = self.datapaths[index]
        return os.path.basename(file_path)

    def get_sample_info(self, index):
        """获取样本的详细信息，用于匹配GT"""
        file_path, subject_id = self.datapaths[index]
        filename = os.path.basename(file_path)
        
        # 解析文件名：hard_pose_test_S19_sample1001.txt
        pattern = r"hard_pose_\w+_(S\d+)_sample(\d+)\.txt"
        match = re.match(pattern, filename)
        if match:
            subject = match.group(1)  # S19
            sample_id = int(match.group(2))  # 1001
            return {
                'filename': filename,
                'subject': subject,
                'sample_id': sample_id,
                'file_path': file_path,
                'index': index
            }
        return None

    def __getitem__(self, index):
        """
        读取单个样本数据：
          加载 txt 文件内容，格式假定为：
              x y z seg_label
          返回随机采样的 npoints 个点、分割标签，同时构造全零的法向量数组做占位符，
          类别标签根据文件名中的被试编号确定（例如 S4 类别标签就是 4）。
          
          如果设置了 return_normalization_params，则额外返回归一化参数：
            - centroid: 质心坐标 [3]
            - scale: 缩放因子 [1]
        """
        file_path, subject_id = self.datapaths[index]
        # 类别标签：将 subject_id 中的数字部分转换为 int 类型
        label = np.array([int(subject_id[1:])], dtype=np.int32)

        # 初始化归一化参数
        centroid = np.zeros(3, dtype=np.float32)
        scale = 1.0

        if index in self.cache:
            # 从缓存中获取数据
            cache_data = self.cache[index]
            pts = cache_data['pts']
            seg = cache_data['seg']
            if self.return_normalization_params:
                centroid = cache_data['centroid']
                scale = cache_data['scale']
        else:
            try:
                # 加载数据，假设每行包含4个数字
                data = np.loadtxt(file_path).astype(np.float32)
            except Exception as e:
                raise IOError("读取文件 {} 出错，错误信息：{}".format(file_path, str(e)))
            
            pts = data[:, 0:3]               # 点坐标
            seg = data[:, -1].astype(np.int32) # 分割标签
            
            # 计算归一化参数
            if self.return_normalization_params:
                # 计算整个点云的质心和最大距离
                centroid = np.mean(pts, axis=0)
                centered_pts = pts - centroid
                distances = np.sqrt(np.sum(centered_pts ** 2, axis=1))
                scale = np.max(distances) if len(distances) > 0 else 1.0
                
                # 如果设置了归一化，则应用归一化
                if self.normalize:
                    pts = centered_pts / scale if scale > 0 else centered_pts
            elif self.normalize:
                # 只归一化但不返回参数
                pts, _, _ = pc_normalize(pts)
            
            # 保存到缓存
            if len(self.cache) < self.cache_size:
                cache_data = {
                    'pts': pts,
                    'seg': seg
                }
                if self.return_normalization_params:
                    cache_data['centroid'] = centroid
                    cache_data['scale'] = scale
                self.cache[index] = cache_data

        # 如果未从缓存加载且未计算归一化参数
        if not self.return_normalization_params and self.normalize and index not in self.cache:
            # 只归一化但不返回参数
            pts, _, _ = pc_normalize(pts)

        num_points = pts.shape[0]
        
        # 固定随机种子确保可重复性
        if self.fixed_seed:
            np.random.seed(index)
        
        # 随机采样 self.npoints 个点（支持重复采样）
        choice = np.random.choice(num_points, self.npoints, replace=True)
        pts_sampled = pts[choice, :]
        seg_sampled = seg[choice]
        
        # 构造全零的法向量数组
        normals_sampled = np.zeros((self.npoints, 3), dtype=np.float32)
        
        # 返回文件名用于匹配GT
        filename = self.get_filename(index)
        
        # 准备返回的数据
        return_data = [
            pts_sampled,           # 点云数据 [npoints, 3]
            label,                 # 类别标签 [1]
            seg_sampled,           # 分割标签 [npoints]
            normals_sampled,       # 法向量 [npoints, 3] (占位符)
            filename               # 文件名 (字符串)
        ]
        
        # 如果需要返回归一化参数
        if self.return_normalization_params:
            return_data.append(centroid)  # 质心 [3]
            return_data.append(scale)     # 缩放因子 [1]
        
        return tuple(return_data)