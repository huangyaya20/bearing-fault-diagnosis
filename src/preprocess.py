"""
数据预处理模块
负责从原始.mat文件加载、处理轴承故障诊断数据
"""
import os
import numpy as np
from scipy.io import loadmat
from .utils import DataUtils


class BearingDataPreprocessor:
    """滚动轴承数据预处理器"""
    
    # 故障类型映射表
    FAULT_TYPES = {
        0: '正常',
        1: '内圈故障0.007"',
        2: '内圈故障0.014"',
        3: '内圈故障0.021"',
        4: '外圈故障0.007"',
        5: '外圈故障0.014"',
        6: '外圈故障0.021"',
        7: '滚动体故障0.007"',
        8: '滚动体故障0.014"',
        9: '滚动体故障0.021"'
    }
    
    def __init__(self, raw_data_dir, processed_data_dir, 
                 window_size=1024, step_size=256, normalize=True):
        """初始化预处理器
        
        Args:
            raw_data_dir: 原始数据目录
            processed_data_dir: 处理后数据保存目录
            window_size: 滑动窗口大小
            step_size: 步长
            normalize: 是否归一化
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        
        os.makedirs(processed_data_dir, exist_ok=True)
    
    def load_single_file(self, file_path, label):
        """加载单个.mat文件并提取信号
        
        Args:
            file_path: 文件路径
            label: 故障标签
            
        Returns:
            tuple: (信号数组, 标签)
        """
        try:
            mat_data = DataUtils.load_mat_file(file_path)
            signal = DataUtils.extract_signal(mat_data)
            
            if self.normalize:
                signal = DataUtils.normalize_signal(signal, method='standard')
            
            return signal, label
        except Exception as e:
            print(f"✗ 加载失败 {file_path}: {str(e)}")
            return None, None
    
    def preprocess_file(self, file_path, label):
        """预处理单个文件：分割成窗口并保存
        
        Args:
            file_path: 输入文件路径
            label: 故障标签
            
        Returns:
            np.ndarray: 处理后的数据 (num_windows, window_size)
        """
        signal, _ = self.load_single_file(file_path, label)
        if signal is None:
            return None
        
        # 创建滑动窗口
        windows = DataUtils.create_sliding_windows(
            signal, self.window_size, self.step_size
        )
        
        return windows
    
    def preprocess_all(self, file_list):
        """批量预处理所有文件
        
        Args:
            file_list: 文件列表 [(file_path, label), ...]
            
        Returns:
            tuple: (X_data, y_labels)
        """
        X_data = []
        y_labels = []
        
        for file_path, label in file_list:
            print(f"处理: {os.path.basename(file_path)} -> {self.FAULT_TYPES[label]}")
            
            windows = self.preprocess_file(file_path, label)
            if windows is not None:
                X_data.append(windows)
                # 为每个窗口创建标签
                y_labels.extend([label] * len(windows))
        
        if not X_data:
            raise ValueError("没有成功处理任何文件")
        
        # 合并所有数据
        X_data = np.vstack(X_data).astype(np.float32)
        y_labels = np.array(y_labels, dtype=np.int64)
        
        print(f"\n✓ 数据处理完成")
        print(f"  总样本数: {len(X_data)}")
        print(f"  样本形状: {X_data.shape}")
        print(f"  标签形状: {y_labels.shape}")
        
        return X_data, y_labels
    
    def save_processed_data(self, X_data, y_labels, prefix='bearing'):
        """保存处理后的数据
        
        Args:
            X_data: 样本数据
            y_labels: 标签数据
            prefix: 文件名前缀
        """
        X_path = os.path.join(self.processed_data_dir, f'{prefix}_X.npy')
        y_path = os.path.join(self.processed_data_dir, f'{prefix}_y.npy')
        
        DataUtils.save_processed_data(X_data, X_path)
        DataUtils.save_processed_data(y_labels, y_path)
        
        print(f"✓ 数据已保存:")
        print(f"  X: {X_path}")
        print(f"  y: {y_path}")
    
    def load_processed_data(self, prefix='bearing'):
        """加载处理后的数据
        
        Args:
            prefix: 文件名前缀
            
        Returns:
            tuple: (X_data, y_labels)
        """
        X_path = os.path.join(self.processed_data_dir, f'{prefix}_X.npy')
        y_path = os.path.join(self.processed_data_dir, f'{prefix}_y.npy')
        
        X_data = DataUtils.load_processed_data(X_path)
        y_labels = DataUtils.load_processed_data(y_path)
        
        print(f"✓ 数据已加载:")
        print(f"  X: {X_data.shape}")
        print(f"  y: {y_labels.shape}")
        
        return X_data, y_labels
