#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import numpy as np

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
        # Implementation of Cosine Annealing with warmup
        train_cfg = self.config['training']
        
        # Create cosine annealing scheduler with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=train_cfg['epochs'],
            eta_min=train_cfg.get('min_lr', 0.0)
        )
        
        # Apply warmup if specified
        if train_cfg.get('warmup_epochs', 0) > 0:
            from torch.optim.lr_scheduler import _LRScheduler
            
            class WarmupCosineAnnealingLR(_LRScheduler):
                def __init__(self, optimizer, T_max, eta_min, warmup_epochs):
                    self.optimizer = optimizer
                    self.T_max = T_max
                    self.eta_min = eta_min
                    self.warmup_epochs = warmup_epochs
                    self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                    self.step_count = 0
                    
                def get_lr(self):
                    if self.step_count < self.warmup_epochs:
                        # Linear warmup
                        return self.base_lrs[0] * (self.step_count + 1) / self.warmup_epochs
                    else:
                        # Cosine annealing
                        return scheduler.get_lr()
                
                def step(self):
                    self.step_count += 1
                    return self.optimizer.step()
                
                def get_last_lr(self):
                    return [self.get_lr()]
            
            return WarmupCosineAnnealingLR(self.optimizer, train_cfg['epochs'], 
                                       train_cfg.get('min_lr', 0.0), 
                                       train_cfg['warmup_epochs'])
        
        return scheduler
    
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
            
            # Apply MixUp or CutMix
            aug_cfg = self.config['training']['augmentation']
            use_mixup = aug_cfg.get('mixup', {}).get('enabled', False)
            use_cutmix = aug_cfg.get('cutmix', {}).get('enabled', False)
            
            # Random toggle when both are enabled to use both augmentations
            if use_mixup and use_cutmix:
                if np.random.rand() < 0.5:
                    data, targets_a, targets_b, lam = self.apply_mixup(data, targets)
                    is_mixed = True
                else:
                    data, targets_a, targets_b, lam = self.apply_cutmix(data, targets)
                    is_mixed = True
            elif use_mixup:
                data, targets_a, targets_b, lam = self.apply_mixup(data, targets)
                is_mixed = True
            elif use_cutmix:
                data, targets_a, targets_b, lam = self.apply_cutmix(data, targets)
                is_mixed = True
            else:
                targets_a, targets_b, lam = targets, None, None
                is_mixed = False
            
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(data)
                    if is_mixed:
                        loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    else:
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
                if is_mixed:
                    loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            
            # For mixed samples, use the first target for accuracy calculation
            if is_mixed:
                _, predicted = outputs.max(1)
                total += targets_a.size(0)
                correct += predicted.eq(targets_a).sum().item()
            else:
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
        eval_interval = self.config['logging'].get('eval_interval', 1)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            train_m = self.train_epoch()
            
            # Validate only at specified intervals
            if (epoch + 1) % eval_interval == 0:
                val_m = self.validate()
                if self.scheduler: self.scheduler.step()
                
                self.logger.info(f"Epoch {epoch+1} | Val Acc: {val_m['accuracy']:.2f}% | Val Loss: {val_m['loss']:.4f}")
                
                # Save checkpoint at intervals
                save_interval = self.config['logging'].get('save_interval', 5)
                if (epoch + 1) % save_interval == 0:
                    checkpoint_path = os.path.join(self.config['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
                    torch.save(self.model.state_dict(), checkpoint_path)
                    self.logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Save best model
                if val_m['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_m['accuracy']
                    torch.save(self.model.state_dict(), os.path.join(self.config['paths']['checkpoint_dir'], 'best_model.pth'))
            else:
                # Still log training progress even without validation
                self.logger.info(f"Epoch {epoch+1} | Training only (eval_interval: {eval_interval})")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
            print(f"TensorBoard logs saved to: {self.writer.log_dir}")
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def close(self):
        """Close TensorBoard writer manually if needed."""
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()
    
    def apply_mixup(self, data, targets):
        """Apply MixUp augmentation to batch."""
        aug_cfg = self.config['training']['augmentation']
        mixup_cfg = aug_cfg.get('mixup', {})
        
        if not mixup_cfg.get('enabled', False):
            return data, targets, None
        
        alpha = mixup_cfg.get('alpha', 0.2)
        beta = mixup_cfg.get('beta', 0.2)
        
        # Generate mixing parameter
        lam = np.random.beta(alpha, beta)
        lam = np.clip(lam, 0.7, 0.95)  # 70-95% original, 5-30% second
        
        # Random shuffle indices
        index = torch.randperm(data.size(0)).to(self.device)
        
        # Mix data
        mixed_data = lam * data + (1 - lam) * data[index]
        
        # Create targets for mixed samples
        targets_a, targets_b = targets, targets[index]
        
        return mixed_data, targets_a, targets_b, lam
    
    def apply_cutmix(self, data, targets):
        """Apply CutMix augmentation to batch."""
        aug_cfg = self.config['training']['augmentation']
        cutmix_cfg = aug_cfg.get('cutmix', {})
        
        if not cutmix_cfg.get('enabled', False):
            return data, targets, None
        
        alpha = cutmix_cfg.get('alpha', 1.0)
        
        # Generate mixing parameter
        lam = np.random.beta(alpha, alpha)
        
        # Random shuffle indices
        index = torch.randperm(data.size(0)).to(self.device)
        
        # Generate random bounding box
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        
        # Apply CutMix
        data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        
        # Create targets for mixed samples
        targets_a, targets_b = targets, targets[index]
        
        return data, targets_a, targets_b, lam
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Loss function for MixUp/CutMix."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['classes']['num_classes'],
        in_chans=config['model']['in_chans'],
        drop_path_rate=config['model']['drop_path_rate']
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