"""
完整的多尺度注意力融合网络 (MS-AFNet)

"""

import torch
import torch.nn as nn
from .backbone import MultiScaleBackbone
from .attention import CBAM


class MSAFNet(nn.Module):
    """多尺度注意力融合诊断网络"""
    
    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super(MSAFNet, self).__init__()
        
        # 多尺度特征提取
        self.backbone = MultiScaleBackbone(in_channels=in_channels)
        
        # 注意力机制
        self.attention = CBAM(in_channels=576, reduction_ratio=16, kernel_size=7)
        
        # 融合层
        self.fusion_conv = nn.Conv1d(576, 256, kernel_size=1)
        self.fusion_bn = nn.BatchNorm1d(256)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # 分类头
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 多尺度特征提取
        features = self.backbone(x)  # 获取4个分支的特征
        
        # 特征拼接
        x = torch.cat(features, dim=1)  # (B, 576, L/16)
        
        # 注意力机制
        x = self.attention(x)
        
        # 融合
        x = self.fusion_conv(x)
        x = self.fusion_bn(x)
        x = self.fusion_relu(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)  # (B, 256, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # 分类
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
