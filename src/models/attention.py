"""
注意力机制模块

实现SE-Block和CBAM两种注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力机制 (SE-Block)"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        mid_channels = max(1, in_channels // reduction_ratio)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (B, C, L)
        batch_size, channels, length = x.size()
        
        # 平均池化
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc_layers(avg_out).view(batch_size, channels, 1)
        
        # 最大池化
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc_layers(max_out).view(batch_size, channels, 1)
        
        # 融合
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (B, C, L)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """CBAM注意力模块 (通道 + 空间)"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
