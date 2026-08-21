"""
多尺度特征提取主干网络

包含多个不同卷积核大小的分支
"""

import torch
import torch.nn as nn
from typing import List


class ConvBranch(nn.Module):
    """单个卷积分支"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super(ConvBranch, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MultiScaleBackbone(nn.Module):
    """多尺度特征提取骨干网络"""
    
    def __init__(self, in_channels: int = 1, num_branches: int = 4):
        super(MultiScaleBackbone, self).__init__()
        
        # 分支1: 3x3卷积
        self.branch1 = nn.Sequential(
            ConvBranch(in_channels, 64, kernel_size=3),
            ConvBranch(64, 64, kernel_size=3)
        )
        
        # 分支2: 5x5卷积
        self.branch2 = nn.Sequential(
            ConvBranch(in_channels, 128, kernel_size=5),
            ConvBranch(128, 128, kernel_size=5)
        )
        
        # 分支3: 7x7卷积
        self.branch3 = nn.Sequential(
            ConvBranch(in_channels, 256, kernel_size=7),
            ConvBranch(256, 256, kernel_size=7)
        )
        
        # 分支4: 空洞卷积
        self.branch4 = nn.Sequential(
            ConvBranch(in_channels, 128, kernel_size=3, dilation=2),
            ConvBranch(128, 128, kernel_size=3, dilation=4)
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回多尺度特征"""
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        return [f1, f2, f3, f4]
