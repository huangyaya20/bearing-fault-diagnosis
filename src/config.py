"""
配置管理模块

加载、管理和验证项目配置参数
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """配置类 - 管理所有超参数"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """初始化配置"""
        self._config = config_dict
        self._validate()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()
    
    def _validate(self):
        """验证配置的合法性"""
        required_keys = ['data', 'model', 'train', 'output']
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required config key: {key}")
        
        data_config = self._config['data']
        if data_config['train_ratio'] + data_config['val_ratio'] + data_config['test_ratio'] != 1.0:
            raise ValueError("Data ratios must sum to 1.0")
        
        logger.info("Configuration validation passed")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"Config({len(self._config)} items)"


def load_config(config_path: str = 'config.yaml') -> Config:
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Config对象
    """
    if not os.path.exists(config_path):
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Config loaded from {config_path}")
        return Config(config_dict)
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config: {e}")
    except Exception as e:
        raise Exception(f"Error loading config: {e}")


def create_directories(config: Config):
    """根据配置创建必要的目录"""
    output_config = config['output']
    
    directories = [
        output_config['save_dir'],
        output_config['models_dir'],
        output_config['logs_dir'],
        output_config['plots_dir'],
        config['data']['data_dir'],
        config['data']['raw_data_dir'],
        config['data']['processed_data_dir'],
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory created: {directory}")
