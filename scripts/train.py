"""
训练脚本

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import torch
import logging
from pathlib import Path
from src.config import load_config, create_directories
from src.models.network import MSAFNet
from src.utils.data_loader import BearingDataLoader
from src.engine.trainer import Trainer
from src.utils.metrics import plot_training_history

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 加载配置
    config = load_config('config.yaml')
    create_directories(config)
    
    logger.info("Starting training...")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # 数据加载
    logger.info("Loading data...")
    data_loader = BearingDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    
    # 构建模型
    logger.info("Building model...")
    model = MSAFNet(num_classes=config['model']['num_classes'])
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 构建训练器
    trainer = Trainer(model, config)
    
    # 训练
    history = trainer.train(train_loader, val_loader)
    
    # 保存训练历史
    plots_dir = Path(config['output']['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, save_path=str(plots_dir / 'training_history.png'))
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to {trainer.output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
