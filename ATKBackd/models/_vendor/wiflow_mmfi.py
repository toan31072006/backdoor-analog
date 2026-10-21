# SPDX-License-Identifier: Apache-2.0
# WiFlow authors: Yi Dao, Lankai Zhang, Hao Liu, Haiwei Zhang, Wenbo Wang.
# Source repository: DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling
# Commit: 8ddb9a01772c368fc591f69e8414044b49d817e1
# Original file: cross_dataset_test/WiFlow/wiflow.py
# MODIFIED 2026-10-22: extracted model/loss definitions and their minimal imports;
# removed unrelated training, dataset, plotting imports and import-time effects.
# Architecture, initialization, forward, dropout and source loss are unchanged.
# The complete license is in licenses/WiFlow-APACHE-2.0.txt.
"""Pinned author MM-Fi 3D definitions. No training/data/plotting at import."""
import math

import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class InnerGroupedTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, attention_type='none'):
        super(InnerGroupedTemporalBlock, self).__init__()

        self.groups = 18

        self.conv1_group = nn.Conv1d(n_inputs, n_inputs, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=self.groups, bias=False)
        self.chomp1 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn1_group = nn.BatchNorm1d(n_inputs)
        self.relu1_group = nn.SiLU(inplace=True)

        self.conv1_pw = nn.Conv1d(n_inputs, n_outputs, 1, bias=False)
        self.bn1_pw = nn.BatchNorm1d(n_outputs)
        self.relu1_pw = nn.SiLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2_group = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                     stride=1, padding=padding, dilation=dilation,
                                     groups=self.groups, bias=False)
        self.chomp2 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn2_group = nn.BatchNorm1d(n_outputs)
        self.relu2_group = nn.SiLU(inplace=True)

        self.conv2_pw = nn.Conv1d(n_outputs, n_outputs, 1, bias=False)
        self.bn2_pw = nn.BatchNorm1d(n_outputs)
        self.relu2_pw = nn.SiLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        # 残差对齐
        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, 1, bias=False),
            nn.BatchNorm1d(n_outputs)
        ) if n_inputs != n_outputs else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)

        # 第一层前向传播：极其平滑的流水线
        out = self.conv1_group(x)
        out = self.chomp1(out)
        out = self.bn1_group(out)  # 第一道防波堤
        out = self.relu1_group(out)
        out = self.conv1_pw(out)
        out = self.bn1_pw(out)  # 第二道防波堤
        out = self.relu1_pw(out)
        out = self.dropout1(out)

        # 第二层前向传播
        out = self.conv2_group(out)
        out = self.chomp2(out)
        out = self.bn2_group(out)
        out = self.relu2_group(out)
        out = self.conv2_pw(out)
        out = self.bn2_pw(out)
        out = self.relu2_pw(out)
        out = self.dropout2(out)

        return F.silu(out + res)

class TemporalBlock(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, attention_type='none'):
        super(TemporalBlock, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                InnerGroupedTemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, dropout=dropout, attention_type=attention_type
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(ConvBlock1, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.final_activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out = out + identity
        out = self.final_activation(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 2), bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.final_activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out = out + identity
        out = self.final_activation(out)
        return out


# 1x1卷积用于改变通道数
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 定义一个一维卷积层，用于进行qkv变换
class qkv_transform(nn.Conv1d):
    """定义qkv变换的Conv1d"""

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
        super(qkv_transform, self).__init__(in_planes, out_planes, kernel_size, stride, padding, bias=bias)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.stride = stride
        self.bias = bias
        self.width = width

        # qkv变换层 - 使用3倍的输出通道
        self.qkv_transform = qkv_transform(in_planes, out_planes * 3, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 3)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes)

        # 如果步长大于1，添加池化层
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 初始化qkv变换权重
        nn.init.normal_(self.qkv_transform.weight.data, 0, math.sqrt(1. / self.in_planes))

    def forward(self, x):
        B, C, H, W = x.shape

        # 根据宽度或高度的设置调整张量维度顺序
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # 进行qkv变换
        qkv = self.bn_qkv(self.qkv_transform(x))  # [N*W, out_planes*3, H]

        # 将qkv分解为q、k、v
        qkv = qkv.reshape(N * W, 3, self.out_planes, H)  # [N*W, 3, out_planes, H]
        qkv = qkv.permute(1, 0, 2, 3)  # [3, N*W, out_planes, H]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [N*W, out_planes, H]

        # 重塑为分组形式
        q = q.reshape(N * W, self.groups, self.group_planes, H)
        k = k.reshape(N * W, self.groups, self.group_planes, H)
        v = v.reshape(N * W, self.groups, self.group_planes, H)

        # 直接计算注意力得分
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        # 归一化
        qk = self.bn_similarity(qk)
        # 应用softmax
        similarity = F.softmax(qk, dim=-1)
        # 加权求和得到v的注意力输出
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        # 整理输出维度
        sv = sv.reshape(N * W, self.out_planes, H)
        out = self.bn_output(sv)
        out = out.view(N, W, self.out_planes, H)

        # 恢复原始维度顺序
        if self.width:
            out = out.permute(0, 2, 1, 3)
        else:
            out = out.permute(0, 2, 3, 1)

        # 如果步长大于1，进行池化
        if self.stride > 1:
            out = self.pooling(out)

        return out

class DualAxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=2, stride=1, bias=False):
        super(DualAxialAttention, self).__init__()

        # 水平和垂直方向的轴向注意力
        self.height_axis = AxialAttention(
            in_planes,
            out_planes,
            groups,
            stride=stride,
            bias=bias,
            width=False
        )

        self.width_axis = AxialAttention(
            out_planes,
            out_planes,
            groups,
            stride=stride,
            bias=bias,
            width=True
        )

    def forward(self, x):

        x = self.width_axis(x)

        x = self.height_axis(x)

        return x

class CSIPoseEstimationModel(nn.Module):
    def __init__(self, dropout=0.3):
        super(CSIPoseEstimationModel, self).__init__()

        self.tcn = TemporalBlock(
            num_inputs=342,
            num_channels=[342, 306, 288],
            kernel_size=3,
            dropout=dropout,
            attention_type='none'
        )

        self.tcn_proj = nn.Sequential(
            nn.Conv1d(288, 272, kernel_size=1, bias=False),
            nn.BatchNorm1d(272),
            nn.SiLU(inplace=True)
        )

        self.up = ConvBlock1(1, 8)

        self.residual_blocks = nn.ModuleList()
        in_channels = 8
        out_channels_list = [8, 16, 32, 64]

        for out_channels in out_channels_list:
            self.residual_blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.att = DualAxialAttention(in_planes=64, out_planes=64, groups=8)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )

        # self.avg_pool = nn.AdaptiveMaxPool2d((17, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 此时的输入 x 形状: [batch_size, 3, 114, 10]
        batch_size = x.size(0)
        time_steps = x.size(3)  # 10

        # ========================================== #
        #  [B, 3, 114, 10] -> [B, 342, 10]
        # ========================================== #
        x = x.reshape(batch_size, -1, time_steps)

        # ===== encoder =====
        x_ant = self.tcn(x)  # [batch_size, 288, 10】

        x_ant = self.tcn_proj(x_ant)  # [B, 272, 10]

        x_ant = x_ant.transpose(1, 2) # [batch_size, 10, 272]
        x_ant = x_ant.unsqueeze(1)  # [batch_size, 1, 10, 272]

        x_ant = self.up(x_ant)

        for block in self.residual_blocks:
            x_ant = block(x_ant)   # [batch_size, 64, 10, 17]

        x = x_ant.permute(0, 1, 3, 2) # 最终: [batch_size, 64, 17, 10]

        x = self.att(x)

        # ===== decoder =====
        x = x[..., -1:]

        x = self.final_conv(x)  # [B, 3, 17, 1]
        x = x.squeeze(-1)  # [B, 3, 17]
        x = x.transpose(1, 2)  # [batch_size, 17, 3]

        # 返回关键点坐标 [batch_size, 17, 3]
        return x

class SimplePoseLoss(nn.Module):
    """简化的姿态损失函数 - 只包含位置和骨骼长度损失"""
    def __init__(self,
                 position_weight=1.0,
                 bone_length_weight=0.2,
                 loss_type='smooth_l1'):  # 'mse', 'l1', 'smooth_l1'
        super().__init__()

        self.position_weight = position_weight
        self.bone_length_weight = bone_length_weight
        self.loss_type = loss_type

        self.bone_connections = [
            (0, 7), (7, 8), (8, 9), (9, 10),
            # 脊柱与头部: Bot Torso -> Center Torso -> Upper Torso -> Neck Base -> Center Head
            (0, 1), (1, 2), (2, 3),  # 左腿: Bot Torso -> L.Hip -> L.Knee -> L.Foot
            (0, 4), (4, 5), (5, 6),  # 右腿: Bot Torso -> R.Hip -> R.Knee -> R.Foot
            (9, 14), (14, 15), (15, 16),  # 左臂: Neck Base -> L.Shoulder -> L.Elbow -> L.Hand
            (9, 11), (11, 12), (12, 13)  # 右臂: Neck Base -> R.Shoulder -> R.Elbow -> R.Hand
        ]

    def compute_bone_lengths(self, keypoints):
        """计算骨骼长度"""
        bone_lengths = []
        for start_idx, end_idx in self.bone_connections:
            bone_vec = keypoints[..., end_idx, :] - keypoints[..., start_idx, :]
            bone_length = torch.sqrt(torch.sum(bone_vec ** 2, dim=-1) + 1e-8)
            bone_lengths.append(bone_length)
        return torch.stack(bone_lengths, dim=-1)

    def forward(self, pred, target):
        """
        pred: [batch_size, 17, 3] - 预测的关键点
        target: [batch_size, 17, 3] - 真实的关键点
        """
        batch_size = pred.shape[0]

        # 1. 位置损失 - 使用不同的损失函数
        if self.loss_type == 'mse':
            position_loss = F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            position_loss = F.l1_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            position_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 2. 骨骼长度约束损失
        try:
            pred_bone_lengths = self.compute_bone_lengths(pred)
            target_bone_lengths = self.compute_bone_lengths(target)

            if self.loss_type == 'mse':
                bone_loss = F.mse_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'l1':
                bone_loss = F.l1_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'smooth_l1':
                bone_loss = F.smooth_l1_loss(pred_bone_lengths, target_bone_lengths, beta=0.05)

        except Exception as e:
            print(f"骨骼长度计算错误: {e}")
            bone_loss = torch.tensor(0.0, device=pred.device)

        # 损失字典
        loss_dict = {
            'position': position_loss.item(),
            'bone': bone_loss.item()
        }

        # 总损失
        total_loss = (self.position_weight * position_loss +
                      self.bone_length_weight * bone_loss)

        return total_loss, loss_dict

# ============================== #
# 评估函数
# ============================== #
