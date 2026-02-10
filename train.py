#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

import os
import yaml 
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# Import custom modules
from model import create_model, ConvNeXtV2
from dataset import get_dataloaders

class Trainer:
    """
    Custom trainer for ConvNeXt-V2 model using YAML configuration.
    """
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device from hardware config
        self.device = torch.device(config['hardware'].get('device', 'cuda'))
        self.model.to(self.device)
        
        # Setup optimizer & loss
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training from hardware/training config
        use_amp = config['training'].get('use_amp', False) or config['hardware'].get('mixed_precision', False)
        self.scaler = GradScaler() if use_amp else None
        
        # Setup TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            log_dir = Path(config['paths']['log_dir']) / f"tensorboard_{int(time.time())}"
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
            print(f"View TensorBoard with: tensorboard --logdir={log_dir}")
        else:
            self.writer = None
            print("TensorBoard logging disabled (tensorboard not installed)")
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # Setup logging
        self._setup_logging()
        # os.makedirs(config['paths']['output_dir'], exist_ok=True)
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        train_cfg = self.config['training']
        # Defaulting to AdamW as per your model needs
        return optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg['learning_rate']),
            weight_decay=float(train_cfg['weight_decay'])
        )
    
    def _create_criterion(self) -> nn.Module:
        class_cfg = self.config['classes']
        if class_cfg.get('class_weights'):
            weights = torch.tensor(class_cfg['class_weights'], device=self.device)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        # Implementation of Cosine Annealing as a default high-performance choice
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['training']['epochs']
        )
    
    def _setup_logging(self):
        log_dir = self.config['paths']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                # Optional: Gradient Clipping
                if self.config['training'].get('gradient_clip'):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'Loss': f'{running_loss/total:.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
            
        # Log to TensorBoard
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        if self.writer:
            self.writer.add_scalar('Train/Loss', epoch_loss, self.current_epoch)
            self.writer.add_scalar('Train/Accuracy', epoch_acc, self.current_epoch)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
            
        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Log to TensorBoard
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        if self.writer:
            self.writer.add_scalar('Val/Loss', val_loss, self.current_epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, self.current_epoch)
        
        return {'loss': val_loss, 'accuracy': val_acc}

    def train(self):
        epochs = self.config['training']['epochs']
        for epoch in range(epochs):
            self.current_epoch = epoch
            train_m = self.train_epoch()
            val_m = self.validate()
            if self.scheduler: self.scheduler.step()
            
            self.logger.info(f"Epoch {epoch+1} | Val Acc: {val_m['accuracy']:.2f}% | Val Loss: {val_m['loss']:.4f}")
            
            if val_m['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_m['accuracy']
                torch.save(self.model.state_dict(), os.path.join(self.config['paths']['checkpoint_dir'], 'best_model.pth'))
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
            print(f"TensorBoard logs saved to: {self.writer.log_dir}")
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def close(self):
        """Close TensorBoard writer manually if needed."""
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['classes']['num_classes'],
        in_chans=config['model']['in_chans'],
        drop_path_rate=config['model']['drop_path_rate'],
        use_coordconv=config['model']['use_coordconv']
    )

    tao_path = config['paths']['tao_weights']
    if config['model']['pretrained']:
        if os.path.exists(tao_path):
            print(f"Loading TAO weights from {tao_path}")
            tao_dict = torch.load(tao_path, map_location='cpu')
            # Access the actual state_dict from TAO checkpoint
            if 'state_dict' in tao_dict:
                tao_weights = tao_dict['state_dict']
            else:
                tao_weights = tao_dict
            loaded, missing = model.load_tao_weights(tao_weights)
            print(f"Loaded {loaded} weights from TAO checkpoint")
            print(f"Missing keys: {missing}")
    
    if config['training'].get('freeze_backbone'):
        for name, param in model.named_parameters():
            if 'head' not in name: param.requires_grad = False

    # 3. Data Loaders
    train_loader, val_loader = get_dataloaders(config)

    # 4. Run Training
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == '__main__':
    main()