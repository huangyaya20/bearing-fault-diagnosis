"""
模型训练引擎
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """训练器"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu')
        self.model = self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = self._build_optimizer()
        
        # 学习率调度
        self.scheduler = self._build_scheduler() if config['train']['scheduler']['enabled'] else None
        
        # 早停机制
        self.early_stopping_patience = config['train']['early_stopping']['patience']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 输出目录
        self.output_dir = Path(config['output']['models_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        lr = self.config['train']['learning_rate']
        weight_decay = self.config['train']['weight_decay']
        
        if self.config['train']['optimizer'] == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, 
                           momentum=self.config['train']['momentum'])
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        if self.config['train']['scheduler']['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['train']['num_epochs'],
                eta_min=self.config['train']['scheduler']['min_lr']
            )
        return None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for signals, labels in tqdm(train_loader, desc="Training"):
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels in tqdm(val_loader, desc="Validating"):
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """完整训练流程"""
        logger.info(f"Training on device: {self.device}")
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.config['train']['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['train']['num_epochs']}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 学习率调整
            if self.scheduler:
                self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                logger.info(f"Model improved. Saving checkpoint.")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("Training completed!")
        return history
    
    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
