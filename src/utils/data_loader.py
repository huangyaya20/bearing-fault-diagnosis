"""
数据加载与预处理模块

负责CWRU轴承数据的加载、预处理和增强
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class BearingDataset(Dataset):
    """轴承故障诊断数据集"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 augmentation: bool = False, 
                 augmentation_params: dict = None):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.augmentation = augmentation
        self.augmentation_params = augmentation_params or {}
        
        logger.info(f"Dataset loaded: {data.shape[0]} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal_data = self.data[idx:idx+1]
        label = self.labels[idx]
        
        if self.augmentation:
            signal_data = self._augment(signal_data)
        
        return signal_data, label
    
    def _augment(self, signal_data: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        if 'gaussian_noise_std' in self.augmentation_params:
            noise = torch.randn_like(signal_data) * self.augmentation_params['gaussian_noise_std']
            signal_data = signal_data + noise
        
        if 'amplitude_scale' in self.augmentation_params:
            scale_range = self.augmentation_params['amplitude_scale']
            scale = np.random.uniform(scale_range[0], scale_range[1])
            signal_data = signal_data * scale
        
        return signal_data


class BearingDataLoader:
    """轴承数据加载器"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.train_config = config['train']
        self.signal_length = self.data_config['signal_length']
        self.sample_rate = self.data_config['sample_rate']
        os.makedirs(self.data_config['processed_data_dir'], exist_ok=True)
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成演示数据"""
        num_samples_per_class = 100
        num_classes = 4
        data_list = []
        labels_list = []
        
        for class_idx in range(num_classes):
            for _ in range(num_samples_per_class):
                t = np.linspace(0, 1, self.signal_length)
                if class_idx == 0:
                    signal_data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(self.signal_length)
                elif class_idx == 1:
                    signal_data = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(self.signal_length)
                elif class_idx == 2:
                    signal_data = np.sin(2 * np.pi * 15 * t) + 0.3 * np.abs(np.sin(2 * np.pi * 30 * t)) + 0.1 * np.random.randn(self.signal_length)
                else:
                    signal_data = np.sin(2 * np.pi * 25 * t) + 0.2 * np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(self.signal_length)
                data_list.append(signal_data)
                labels_list.append(class_idx)
        
        data = np.array(data_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)
        logger.info(f"Synthetic data generated: {data.shape}")
        return data, labels
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """预处理信号数据"""
        logger.info("Preprocessing data...")
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        return data_normalized.astype(np.float32)
    
    def split_data(self, data: np.ndarray, labels: np.ndarray):
        """分割数据"""
        train_ratio = self.data_config['train_ratio']
        val_ratio = self.data_config['val_ratio']
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        indices = np.random.permutation(n_samples)
        data_shuffled = data[indices]
        labels_shuffled = labels[indices]
        
        X_train = data_shuffled[:n_train]
        X_val = data_shuffled[n_train:n_train+n_val]
        X_test = data_shuffled[n_train+n_val:]
        
        y_train = labels_shuffled[:n_train]
        y_val = labels_shuffled[n_train:n_train+n_val]
        y_test = labels_shuffled[n_train+n_val:]
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_dataloaders(self, batch_size: int = None, num_workers: int = None):
        """获取PyTorch数据加载器"""
        batch_size = batch_size or self.train_config['batch_size']
        num_workers = num_workers or self.train_config.get('num_workers', 0)
        
        data, labels = self._generate_synthetic_data()
        data = self.preprocess(data)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(data, labels)
        
        aug_params = self.data_config.get('augmentation', {})
        train_dataset = BearingDataset(X_train, y_train, augmentation=True, augmentation_params=aug_params)
        val_dataset = BearingDataset(X_val, y_val, augmentation=False)
        test_dataset = BearingDataset(X_test, y_test, augmentation=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        logger.info(f"DataLoaders created with batch_size={batch_size}")
        return train_loader, val_loader, test_loader
