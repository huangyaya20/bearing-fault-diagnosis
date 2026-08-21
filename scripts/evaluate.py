"""
评估脚本 - 测试模型性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import torch
import logging
from pathlib import Path
from src.config import load_config
from src.models.network import MSAFNet
from src.utils.data_loader import BearingDataLoader
from src.engine.evaluator import Evaluator
from src.utils.metrics import plot_confusion_matrix, calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 加载配置
    config = load_config('config.yaml')
    
    logger.info("Starting evaluation...")
    
    # 数据加载
    logger.info("Loading data...")
    data_loader = BearingDataLoader(config)
    _, _, test_loader = data_loader.get_dataloaders()
    
    # 加载模型
    logger.info("Loading model...")
    model = MSAFNet(num_classes=config['model']['num_classes'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(config['output']['models_dir']) / 'best_model.pth'
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {checkpoint_path}")
    else:
        logger.warning("No checkpoint found, using untrained model")
    
    model = model.to(device)
    
    # 评估
    logger.info("Evaluating model...")
    evaluator = Evaluator(model, device=device)
    results = evaluator.evaluate(test_loader)
    
    # 计算指标
    metrics = calculate_metrics(results['labels'], results['predictions'])
    logger.info(f"\nMetrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # 保存混淆矩阵
    plots_dir = Path(config['output']['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / 'confusion_matrix.png'
    plot_confusion_matrix(results['labels'], results['predictions'], save_path=str(cm_path))
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
