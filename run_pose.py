
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib
# 指定无界面后端（其他绘图模块如果还需要留用时，保持不变）
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入新封装的 GT 提取模块和辅助函数
from ground_truth_extractor import MultiGroundTruthExtractor, get_sample_index

# 假定数据集与模型都在对应模块中
from datasets.HardPose import HardPoseMultiSubjectDataset
from models import NonParamPoseNet

################################################################################
# 位姿评估器，用于计算 MPJPE、MSE、MAE、RMSE、PCK 等指标
################################################################################

class PoseEvaluator:
    def __init__(self):
        self.metrics = {}
        self.predictions = []               # 存储预测结果 (NumPy数组)
        self.groundtruths = []              # 存储真实值 (NumPy数组)
        self.file_mappings = []             # 存储样本映射信息
        self.denormalized_predictions = []  # 存储反归一化后的预测结果 (NumPy数组)

    def calculate_pose_metrics(self, pred_poses, gt_poses):
        """
        计算主要指标：
          - MPJPE: 每个关节平均误差；
          - MSE、MAE、RMSE；
          - 不同阈值下的 PCK（单位 m）；
          - 关节相关系数（Correlation）。
        """
        if isinstance(pred_poses, torch.Tensor):
            pred_poses = pred_poses.cpu().numpy()
        if isinstance(gt_poses, torch.Tensor):
            gt_poses = gt_poses.cpu().numpy()

        min_batch = min(pred_poses.shape[0], gt_poses.shape[0])
        pred_poses = pred_poses[:min_batch]
        gt_poses = gt_poses[:min_batch]

        # 1. 计算每个关节的误差 (欧氏距离)
        joint_errors = np.linalg.norm(pred_poses - gt_poses, axis=2)  # shape: [B, num_joints]
        mpjpe = np.mean(joint_errors)

        # 2. 全局指标 MSE、MAE、RMSE
        mse = mean_squared_error(gt_poses.reshape(-1), pred_poses.reshape(-1))
        mae = mean_absolute_error(gt_poses.reshape(-1), pred_poses.reshape(-1))
        rmse = np.sqrt(mse)

        # 3. PCK 指标（不同阈值下）
        pck_results = {}
        for th in [0.05, 0.1, 0.2, 0.5]:
            correct = (joint_errors < th).astype(np.float32)
            pck = np.mean(correct) * 100
            pck_results[f'PCK_{int(th*100)}cm'] = pck

        # 4. 计算相关系数
        pred_flat = pred_poses.reshape(-1)
        gt_flat = gt_poses.reshape(-1)
        if len(pred_flat) > 1 and len(gt_flat) > 1:
            correlation = np.corrcoef(pred_flat, gt_flat)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        metrics = {
            'MPJPE': mpjpe,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Per_Joint_MPJPE': np.mean(joint_errors, axis=0),
            'Correlation': correlation
        }
        metrics.update(pck_results)
        return metrics

    def update_metrics(self, pred_poses, gt_poses, file_mapping=None):
        if isinstance(pred_poses, torch.Tensor):
            pred_poses = pred_poses.detach().cpu().numpy()
        if isinstance(gt_poses, torch.Tensor):
            gt_poses = gt_poses.detach().cpu().numpy()

        batch_metrics = self.calculate_pose_metrics(pred_poses, gt_poses)
        for key, val in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(val)

        self.predictions.append(pred_poses)
        self.groundtruths.append(gt_poses)

        if file_mapping is not None:
            self.file_mappings.extend(file_mapping)

    def denormalize_prediction(self, pred_poses, centers, scales):
        scales = scales[:, None, None]
        centers = centers[:, None, :]
        denormalized = pred_poses * scales + centers
        return denormalized

    def get_final_metrics(self):
        final_metrics = {}
        for key, values in self.metrics.items():
            if key == 'Per_Joint_MPJPE':
                final_metrics[key] = np.mean(values, axis=0)
            else:
                final_metrics[key] = np.mean(values)
        return final_metrics

    def get_file_wise_metrics(self):
        if not self.file_mappings:
            return None

        file_metrics = {}
        all_preds = np.concatenate(self.predictions, axis=0)
        all_gts = np.concatenate(self.groundtruths, axis=0)
        for i, mapping in enumerate(self.file_mappings):
            file_idx = mapping['file_index']
            if file_idx not in file_metrics:
                file_metrics[file_idx] = {'predictions': [], 'groundtruths': [], 'sample_indices': []}
            if i < len(all_preds):
                file_metrics[file_idx]['predictions'].append(all_preds[i])
                file_metrics[file_idx]['groundtruths'].append(all_gts[i])
                file_metrics[file_idx]['sample_indices'].append(mapping['sample_index'])
        file_results = {}
        for file_idx, data in file_metrics.items():
            pred_arr = np.array(data['predictions'])
            gt_arr = np.array(data['groundtruths'])
            metrics = self.calculate_pose_metrics(pred_arr, gt_arr)
            file_results[file_idx] = {'metrics': metrics, 'sample_count': len(data['predictions'])}
        return file_results

    def save_detailed_results(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        final_metrics = self.get_final_metrics()
        file_wise = self.get_file_wise_metrics()
        results = {
            'final_metrics': final_metrics,
            'batch_metrics': self.metrics,
            'file_wise_metrics': file_wise,
            'file_mappings': self.file_mappings
        }
        save_file = os.path.join(save_path, "metrics.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(results, f)
        all_preds = np.concatenate(self.predictions, axis=0)
        all_gts = np.concatenate(self.groundtruths, axis=0)
        np.save(os.path.join(save_path, "predictions.npy"), all_preds)
        np.save(os.path.join(save_path, "groundtruths.npy"), all_gts)

        if self.denormalized_predictions:
            all_denorm_preds = np.concatenate(self.denormalized_predictions, axis=0)
            np.save(os.path.join(save_path, "denormalized_predictions.npy"), all_denorm_preds)
        return save_file

################################################################################
# 可视化函数：展示单个样本的点云、预测关节和真实关节，并生成 HTML 文件供交互查看
################################################################################


def visualize_pose(point_cloud, pred_joints, gt_joints,  saved_seg_preds, title="Pose Visualization", save_path="./sample_visualization.html", draw_connections=True):
    """
    参数：
      point_cloud: (N, 3) 的 numpy 数组，点云数据
      pred_joints: (num_joints, 3) 的 numpy 数组，预测的关节坐标
      gt_joints: (num_joints, 3) 的 numpy 数组，真实 GT 关节坐标
      saved_seg_preds: (N,) 的 numpy 数组，点云的分割预测标签
      title: 图标题
      save_path: 保存的 HTML 文件路径
      draw_connections: 是否绘制每个关节处的误差连线（预测 vs GT）
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("请安装 plotly 库，例如通过 pip install plotly")
        return

    # 绘制点云：如果提供分割预测标签，则根据标签设置颜色，否则使用灰色
    if saved_seg_preds is not None:
        trace_pc = go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=saved_seg_preds,   # 用预测的分割标签作为颜色映射值
                colorscale='Viridis',    # 可以根据需要选择其他颜色映射，如 'Jet'
                opacity=0.5,
                colorbar=dict(title="分割预测标签")  # 添加颜色条显示标签数值
            ),
            name='点云'
        )
    else:
        trace_pc = go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='grey',
                opacity=0.5
            ),
            name='点云'
        )

    # 绘制预测关节：红色圆点
    trace_pred = go.Scatter3d(
        x=pred_joints[:, 0],
        y=pred_joints[:, 1],
        z=pred_joints[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color='red'
        ),
        name='预测关节'
    )

    # 绘制真实关节：蓝色菱形
    trace_gt = go.Scatter3d(
        x=gt_joints[:, 0],
        y=gt_joints[:, 1],
        z=gt_joints[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            symbol='diamond'
        ),
        name='真实关节'
    )

    data = [trace_pc, trace_pred, trace_gt]

    # 如果需要绘制每个关节的连线（预测值和 GT 之间的误差向量）
    if draw_connections:
        # 利用 None 分隔线段
        line_x, line_y, line_z = [], [], []
        num_joints = min(pred_joints.shape[0], gt_joints.shape[0])
        for i in range(num_joints):
            line_x.extend([pred_joints[i, 0], gt_joints[i, 0], None])
            line_y.extend([pred_joints[i, 1], gt_joints[i, 1], None])
            line_z.extend([pred_joints[i, 2], gt_joints[i, 2], None])
        trace_lines = go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='误差连线'
        )
        data.append(trace_lines)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        legend=dict(
            x=0,
            y=1
        )
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(save_path)
    print(f"可视化 HTML 文件已保存到: {save_path}")
    


################################################################################
# 参数解析函数
################################################################################
def get_arguments():
    parser = argparse.ArgumentParser(description="无参位姿估计评估")
    parser.add_argument("--npoints", type=int, default=1024, help="输入点云数量")
    parser.add_argument("--stages", type=int, default=4, help="网络阶段数")
    parser.add_argument("--dim", type=int, default=63, help="Embedding维度")
    parser.add_argument("--k", type=int, default=20, help="k近邻数")
    parser.add_argument("--de_k", type=int, default=10, help="Downsample时邻居数")
    parser.add_argument("--beta", type=float, default=0.5, help="beta 参数")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha 参数")
    parser.add_argument("--model_path", type=str, default="", help="预训练模型路径")
    parser.add_argument("--gt_dir", type=str, default="/data/coding/upload-data/data/Point-NN-main/data/hard_pose/output_test_txt_cloud",
                        help="Ground Truth 数据所在目录")
    parser.add_argument("--gt_pattern", type=str, default="", help="Ground Truth 数据文件匹配模式")
    parser.add_argument("--max_samples_per_file", type=int, default=None, help="每个文件最大采样数")
    parser.add_argument("--split", type=str, default="test", help="数据集 split (train/val/test)")
    parser.add_argument("--batch_size", type=int, default=32, help="测试批大小")
    # 新增参数：指定用于可视化的样本索引，默认 -1 表示选择第一个匹配的样本
    parser.add_argument("--vis_sample", type=int, default=-1, help="用于可视化的样本索引（匹配样本中的索引）")
    parser.add_argument("--save_results", type=str, default="./results", help="结果保存的目录")
    args = parser.parse_args()
    return args

################################################################################
# 主流程：加载模型、数据集与评估
################################################################################
@torch.no_grad()
def main():
    args = get_arguments()
    print("==> 参数：", args)

    print("==> 构造无参位姿估计网络...")
    model = NonParamPoseNet(input_points=args.npoints,
                            num_stages=args.stages,
                            embed_dim=args.dim,
                            k_neighbors=args.k,
                            de_neighbors=args.de_k,
                            num_parts=28)
    model = model.cuda()
    model.eval()

    if args.model_path and os.path.exists(args.model_path):
        print(f"==> 加载预训练模型: {args.model_path}")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    print("==> 提取多个 Ground Truth 文件的数据...")
    gt_source = args.gt_pattern if args.gt_pattern else args.gt_dir
    gt_extractor = MultiGroundTruthExtractor(gt_source)
    if not gt_extractor.extract_all_groundtruth(args.max_samples_per_file):
        print("Ground Truth 数据提取失败！")
        return

    gt_matrix = gt_extractor.get_groundtruth_matrix()  
    if gt_matrix is None:
        print("没有提取到任何 GT 数据！")
        return
    print(f"Ground Truth 矩阵形状: {gt_matrix.shape}")

    test_dataset = HardPoseMultiSubjectDataset(npoints=args.npoints,
                                                 split=args.split,
                                                 normalize=True,
                                                 fixed_seed=True,
                                                 return_normalization_params=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             drop_last=False)

    evaluator = PoseEvaluator()
    total_samples = 0
    matched_samples = 0
    unmatched_samples = 0
    debug_samples = []

    # 用于保存所有匹配到的样本，用于可视化
    valid_samples = []  # 每个元素为 (point_cloud, pred_joints, gt_joints)

    for batch_idx, batch in enumerate(tqdm(test_loader)):
        pts, subject_label, seg_labels, normals, filenames, centers, scales = batch
        pts = pts.float().cuda()
        x = pts.permute(0, 2, 1)

        try:
            seg_preds, poses_pred = model(x)
        except RuntimeError as e:
            print(f"模型推理失败 (batch {batch_idx}): {e}")
            continue

        if centers is not None and scales is not None:
            centers = centers.cuda()
            scales = scales.cuda()
            denorm_poses = poses_pred * scales.unsqueeze(-1).unsqueeze(-1) + centers.unsqueeze(1)
            evaluator.denormalized_predictions.append(denorm_poses.detach().cpu().numpy())

        batch_gt = []
        valid_mask = []
        for i, filename in enumerate(filenames):
            sample_idx = get_sample_index(filename)
            if sample_idx is not None and sample_idx < gt_matrix.shape[0]:
                gt_joints = gt_matrix[sample_idx].clone()
                # 【修改部分】：检查 ground truth 的维度
                # 如果 gt_joints 仅包含 3 个数（即 shape 为 (3,)），则扩展为与预测关节数量一致
                if (not isinstance(gt_joints, torch.Tensor)):
                    gt_joints = torch.tensor(gt_joints)
                if gt_joints.ndim == 1 or gt_joints.shape[0] == 3:
                    num_joints = poses_pred.shape[1]  # 预测的关节数，比如 28
                    gt_joints = gt_joints.view(1, 3).repeat(num_joints, 1)
                # 如果 centers 和 scales 存在，对 GT 做归一化处理
                if centers is not None and scales is not None:
                    center_i = centers[i].cpu().numpy()
                    scale_i = scales[i].cpu().numpy()
                    gt_joints = (gt_joints - center_i) / scale_i
                batch_gt.append(gt_joints)
                valid_mask.append(i)
                matched_samples += 1
                valid_samples.append((pts[i].detach().cpu().numpy(),
                                      poses_pred[i].detach().cpu().numpy(),
                                      seg_preds[i].detach().cpu().numpy(),
                                      gt_joints.detach().cpu().numpy()))
            else:
                unmatched_samples += 1
                if len(debug_samples) < 5:
                    debug_samples.append(filename)

        if batch_gt:
            batch_gt = torch.stack(batch_gt)
            if batch_gt.device != poses_pred.device:
                batch_gt = batch_gt.to(poses_pred.device)
            valid_poses = poses_pred[valid_mask]
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"\n调试信息 (batch {batch_idx}):")
                print(f"  示例文件名: {filenames[0]}")
                sample_diff = torch.norm(valid_poses[0] - batch_gt[0]).item()
                print(f"  预测值范围: [{valid_poses[0].min():.4f}, {valid_poses[0].max():.4f}]")
                print(f"  GT 值范围: [{batch_gt[0].min():.4f}, {batch_gt[0].max():.4f}]")
                print(f"  样本差异: {sample_diff:.4f}")
            evaluator.update_metrics(valid_poses, batch_gt)
        total_samples += len(filenames)

    print(f"\n=== 数据匹配统计 ===")
    print(f"总样本数: {total_samples}")
    print(f"成功匹配: {matched_samples}")
    print(f"未匹配: {unmatched_samples}")
    print(f"匹配率: {matched_samples/total_samples*100:.2f}%")
    if debug_samples:
        print(f"未匹配样本示例: {debug_samples}")
    if matched_samples == 0:
        print("❌ 没有匹配到任何样本！请检查文件名格式及 GT 文件。")
        return None

    final_metrics = evaluator.get_final_metrics()
    print("\n=== 总体评估结果 ===")
    print(f"MPJPE: {final_metrics['MPJPE']:.4f} m")
    print(f"RMSE: {final_metrics['RMSE']:.4f} m")
    print(f"MAE: {final_metrics['MAE']:.4f} m")
    print(f"相关系数: {final_metrics.get('Correlation', 0.0):.4f}")
    print("\nPCK 指标:")
    for key, value in final_metrics.items():
        if key.startswith('PCK_'):
            print(f"{key}: {value:.2f}%")

    save_file = evaluator.save_detailed_results(args.save_results)
    print(f"\n详细结果已保存至: {save_file}")

    # 根据参数选择用于可视化的样本
    if valid_samples:
        vis_index = args.vis_sample if 0 <= args.vis_sample < len(valid_samples) else 0
        saved_point_cloud, saved_pred_joints,saved_seg_preds , saved_gt_joints = valid_samples[vis_index]
        print(f"\n=== 正在生成样本 {vis_index} 的可视化 HTML 文件 ===")
        vis_save_path = os.path.join(args.save_results, "sample_visualization.html")
        visualize_pose(saved_point_cloud, saved_pred_joints, saved_gt_joints,saved_seg_preds,
                       title="预测 vs GT关节可视化", save_path=vis_save_path, draw_connections=True)

    else:
        print("未找到可视化样本。")

    return final_metrics

if __name__ == '__main__':
    results = main()
    if results:
        print("评估完成！")
    else:
        print("评估失败，请检查数据匹配问题。")
