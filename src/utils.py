"""
工具函数模块
包含数据处理、可视化和模型管理的通用工具
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import torch


class DataUtils:
    """数据处理工具类"""
    
    @staticmethod
    def load_mat_file(file_path):
        """加载.mat文件
        
        Args:
            file_path: .mat文件路径
            
        Returns:
            dict: 加载的数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            mat_data = loadmat(file_path)
            # 移除MATLAB特殊键
            mat_data = {k: v for k, v in mat_data.items() 
                       if not k.startswith('__')}
            return mat_data
        except Exception as e:
            raise RuntimeError(f"加载失败 {file_path}: {str(e)}")
    
    @staticmethod
    def extract_signal(data_dict, key='X'):
        """从字典中提取信号
        
        Args:
            data_dict: 数据字典
            key: 信号键值
            
        Returns:
            np.ndarray: 信号数组
        """
        if key not in data_dict:
            valid_keys = [k for k in data_dict.keys() 
                         if isinstance(data_dict[k], np.ndarray) 
                         and data_dict[k].dtype in [np.float32, np.float64]]
            if not valid_keys:
                raise ValueError(f"找不到合适的数据键")
            key = valid_keys[0]
        
        signal = data_dict[key]
        if signal.ndim > 1:
            signal = signal.flatten()
        return signal.astype(np.float32)
    
    @staticmethod
    def normalize_signal(signal, method='standard'):
        """信号归一化
        
        Args:
            signal: 输入信号
            method: 'standard' 或 'minmax'
            
        Returns:
            np.ndarray: 归一化后的信号
        """
        signal = np.asarray(signal, dtype=np.float32)
        
        if method == 'standard':
            mean = signal.mean()
            std = signal.std() + 1e-8
            return (signal - mean) / std
        elif method == 'minmax':
            min_val = signal.min()
            max_val = signal.max()
            return (signal - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"未知的归一化方法: {method}")
    
    @staticmethod
    def create_sliding_windows(signal, window_size, step_size):
        """创建滑动窗口
        
        Args:
            signal: 输入信号
            window_size: 窗口大小
            step_size: 步长
            
        Returns:
            np.ndarray: 形状为 (num_samples, window_size) 的数组
        """
        windows = []
        for i in range(0, len(signal) - window_size + 1, step_size):
            windows.append(signal[i:i + window_size])
        
        return np.array(windows, dtype=np.float32)
    
    @staticmethod
    def save_processed_data(data, save_path):
        """保存处理后的数据
        
        Args:
            data: 数据数组
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data)
        print(f"✓ 数据已保存: {save_path}")
    
    @staticmethod
    def load_processed_data(load_path):
        """加载处理后的数据
        
        Args:
            load_path: 加载路径
            
        Returns:
            np.ndarray: 加载的数据
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"文件不存在: {load_path}")
        return np.load(load_path)


class VisualizationUtils:
    """可视化工具类"""
    
    @staticmethod
    def plot_signal(signal, title="信号", save_path=None):
        """绘制信号波形
        
        Args:
            signal: 输入信号
            title: 标题
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 4))
        plt.plot(signal, linewidth=0.5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('样本点', fontsize=12)
        plt.ylabel('幅值', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 信号图已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path=None):
        """绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.colorbar(im)
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)
        
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 混淆矩阵已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """绘制训练历史曲线
        
        Args:
            history: 包含loss和accuracy的字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss曲线
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('模型损失', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy曲线
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_title('模型精度', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy (%)', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 训练曲线已保存: {save_path}")
        plt.close()


class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def save_model(model, save_path):
        """保存模型权重
        
        Args:
            model: PyTorch模型
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✓ 模型已保存: {save_path}")
    
    @staticmethod
    def load_model(model, load_path, device='cpu'):
        """加载模型权重
        
        Args:
            model: PyTorch模型
            load_path: 加载路径
            device: 设备
            
        Returns:
            model: 加载权重后的模型
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        model.load_state_dict(torch.load(load_path, map_location=device))
        return model
    
    @staticmethod
    def count_parameters(model):
        """计算模型参数数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            int: 参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_device():
        """获取计算设备
        
        Returns:
            torch.device: CUDA或CPU设备
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ 使用GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print(f"✓ 使用CPU设备")
        return device
