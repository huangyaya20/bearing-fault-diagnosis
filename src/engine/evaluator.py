"""
评估与测试模块
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(self, data_loader):
        """评估模型"""
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in data_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(signals)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(all_labels, all_preds)}")
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'confusion_matrix': cm
        }
