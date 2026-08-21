#!/usr/bin/env python3
"""
单样本测试脚本 - 用于演示模型推理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from src.config import load_config
from src.models.network import MSAFNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_single(signal, model, device):
    """
    对单个信号进行预测
    
    Args:
        signal: 输入信号
        model: 训练好的模型
        device: 计算设备
    
    Returns:
        预测类别和概率
    """
    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, axis=0)
    
    if signal.shape[0] != 1:
        signal = signal[np.newaxis, :]
    
    signal_tensor = torch.from_numpy(signal).float().to(device)
    
    with torch.no_grad():
        output = model(signal_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='单样本故障诊断测试')
    parser.add_argument('--model', type=str, default='results/models/best_model.pth',
                       help='模型路径')
    parser.add_argument('--signal', type=str, help='信号文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    logger.info("Loading model...")
    model = MSAFNet(num_classes=config['model']['num_classes'])
    
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {args.model}")
    else:
        logger.warning(f"Model file not found: {args.model}")
    
    model = model.to(device)
    model.eval()
    
    # 生成或加载信号
    if args.signal and os.path.exists(args.signal):
        signal = np.load(args.signal)
        logger.info(f"Signal loaded from {args.signal}")
    else:
        logger.info("Generating demo signal...")
        signal_length = config['data']['signal_length']
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(signal_length)
    
    # 预测
    logger.info("Making prediction...")
    pred_class, probabilities = predict_single(signal, model, device)
    
    class_names = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault']
    logger.info(f"Predicted class: {class_names[pred_class]}")
    logger.info(f"Class probabilities:")
    for i, prob in enumerate(probabilities):
        logger.info(f"  {class_names[i]}: {prob:.4f}")


if __name__ == '__main__':
    main()
