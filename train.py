#!/usr/bin/env python3
from cv2 import data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import numpy as np

import os
import yaml 
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
        self.current_epoch = 0
        self.best_val_acc = 0.0
        
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
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        train_cfg = self.config['training']
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
        train_cfg = self.config['training']
        warmup_epochs = int(train_cfg.get('warmup_epochs', 0))
        total_epochs = int(train_cfg['epochs'])
        
        if warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=float(train_cfg.get('min_lr', 0.0))
            )
            return optim.lr_scheduler.SequentialLR(
                self.optimizer, 
                schedulers=[warmup_sched, cosine_sched], 
                milestones=[warmup_epochs]
            )
        
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=float(train_cfg.get('min_lr', 0.0))
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

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        last_path = os.path.join(self.config['paths']['checkpoint_dir'], 'last_checkpoint.pth')
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.config['paths']['checkpoint_dir'], 'best_model.pth')
            torch.save(state, best_path)
            
        save_interval = self.config['logging'].get('save_interval', 5)
        if (epoch + 1) % save_interval == 0:
            epoch_path = os.path.join(self.config['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(state, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found at: {checkpoint_path}, starting from scratch.")
            return

        self.logger.info(f"Resuming training from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.current_epoch = checkpoint['epoch'] + 1 
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.logger.info(f"Success! Resuming from Epoch {self.current_epoch}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        use_coordconv = self.config.get('coordconv', {}).get('enabled', False)
        
        for batch in pbar:
            if use_coordconv:
                data, targets, coords = batch
                data, targets, coords = data.to(self.device), targets.to(self.device), coords.to(self.device)
            else:
                data, targets = batch
                data, targets = data.to(self.device), targets.to(self.device)
                coords = None
            
            self.optimizer.zero_grad()
            
            aug_cfg = self.config['training']['augmentation']
            use_mixup = aug_cfg.get('mixup', {}).get('enabled', False)
            use_cutmix = aug_cfg.get('cutmix', {}).get('enabled', False)
            
            if use_mixup and use_cutmix:
                if np.random.rand() < 0.5:
                    data, targets_a, targets_b, lam, mixed_coords = self.apply_mixup(data, targets, coords)
                    is_mixed = True
                else:
                    data, targets_a, targets_b, lam, mixed_coords = self.apply_cutmix(data, targets, coords)
                    is_mixed = True
            elif use_mixup:
                data, targets_a, targets_b, lam, mixed_coords = self.apply_mixup(data, targets, coords)
                is_mixed = True
            elif use_cutmix:
                data, targets_a, targets_b, lam, mixed_coords = self.apply_cutmix(data, targets, coords)
                is_mixed = True
            else:
                targets_a, targets_b, lam = targets, None, None
                mixed_coords = coords
                is_mixed = False
            
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(data, coords=mixed_coords)
                    loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam) if is_mixed else self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                if self.config['training'].get('gradient_clip'):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data, coords=mixed_coords)
                loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam) if is_mixed else self.criterion(outputs, targets)
                loss.backward()
                if self.config['training'].get('gradient_clip'):
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.optimizer.step()
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_size = targets_a.size(0) if is_mixed else targets.size(0)
            total += batch_size
            correct += predicted.eq(targets_a if is_mixed else targets).sum().item()
            
            # SAFE PROGRESS BAR (Prevents div by zero if first batch fails)
            if total > 0:
                pbar.set_postfix({'Loss': f'{running_loss/total:.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
            
        # SAFE EPOCH METRICS
        epoch_loss = running_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        epoch_acc = 100. * correct / total if total > 0 else 0.0
        
        if self.writer:
            self.writer.add_scalar('Train/Loss', epoch_loss, self.current_epoch)
            self.writer.add_scalar('Train/Accuracy', epoch_acc, self.current_epoch)
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/Learning_Rate', current_lr, self.current_epoch)
            
        return {'loss': epoch_loss, 'accuracy': epoch_acc}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        tp, fp, tn, fn = 0, 0, 0, 0
        use_coordconv = self.config.get('coordconv', {}).get('enabled', False)
        
        with torch.no_grad():
            for batch in self.val_loader:
                if use_coordconv:
                    data, targets, coords = batch
                    coords = coords.to(self.device)
                else:
                    data, targets = batch
                    coords = None
                
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data, coords=coords)
                loss = self.criterion(outputs, targets)

                if torch.isnan(loss):
                    print("Detected NaN loss!")
                    print(f"Logits: {outputs}") 
                    print(f"Targets: {targets}")
                    torch.save({'data': data, 'targets': targets}, 'nan_batch.pt')
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                for i in range(targets.size(0)):
                    actual, pred = targets[i].item(), predicted[i].item()
                    if actual == 0 and pred == 0: tp += 1
                    elif actual == 1 and pred == 0: fp += 1
                    elif actual == 1 and pred == 1: tn += 1
                    elif actual == 0 and pred == 1: fn += 1
        
        # SAFE METRIC CALCULATIONS
        val_loss = running_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        val_acc = 100. * correct / total if total > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if self.writer:
            self.writer.add_scalar('Val/Loss', val_loss, self.current_epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, self.current_epoch)
            self.writer.add_scalar('Val/Recall_NG', recall, self.current_epoch)
            self.writer.add_scalar('Val/Precision_NG', precision, self.current_epoch)
            self.writer.add_scalar('Val/F1_NG', f1_score, self.current_epoch)
        
        return {
            'loss': val_loss, 'accuracy': val_acc,
            'recall_ng': recall, 'precision_ng': precision, 'f1_ng': f1_score
        }

    def train(self):
        epochs = self.config['training']['epochs']
        eval_interval = self.config['logging'].get('eval_interval', 1)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            train_m = self.train_epoch()
            
            if self.scheduler: 
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.info(f"Epoch {epoch+1}: LR updated to {current_lr:.6f}")
            
            if (epoch + 1) % eval_interval == 0:
                val_m = self.validate()
                self.logger.info(f"Epoch {epoch+1} | Val Acc: {val_m['accuracy']:.2f}% | Val Loss: {val_m['loss']:.4f} Recall: {val_m['recall_ng']:.4f}")
                
                is_best = val_m['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_m['accuracy']
                
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                self.logger.info(f"Epoch {epoch+1} | Training only")
        
        if self.writer: self.writer.close()
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def close(self):
        if hasattr(self, 'writer') and self.writer: self.writer.close()
    
    def apply_mixup(self, data, targets, coords=None):
        aug_cfg = self.config['training']['augmentation']
        mixup_cfg = aug_cfg.get('mixup', {})
        if not mixup_cfg.get('enabled', False): return data, targets, None
        
        alpha, beta = mixup_cfg.get('alpha', 0.2), mixup_cfg.get('beta', 0.2)
        lam = np.clip(np.random.beta(alpha, beta), 0.7, 0.95)
        index = torch.randperm(data.size(0)).to(self.device)
        
        mixed_data = lam * data + (1 - lam) * data[index]
        mixed_coords = lam * coords + (1 - lam) * coords[index] if coords is not None else None
        return mixed_data, targets, targets[index], lam, mixed_coords
    
    def apply_cutmix(self, data, targets, coords=None):
        aug_cfg = self.config['training']['augmentation']
        cutmix_cfg = aug_cfg.get('cutmix', {})
        if not cutmix_cfg.get('enabled', False): return data, targets, None
        
        lam = np.random.beta(cutmix_cfg.get('alpha', 1.0), cutmix_cfg.get('alpha', 1.0))
        index = torch.randperm(data.size(0)).to(self.device)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        
        data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        mixed_coords = lam * coords + (1 - lam) * coords[index] if coords is not None else None
        return data, targets, targets[index], lam, mixed_coords
    
    def rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        return np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H), \
               np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    model = create_model(
        model_name=config['model']['name'], num_classes=config['classes']['num_classes'],
        in_chans=config['model']['in_chans'], drop_path_rate=config['model']['drop_path_rate'],
        use_coordconv=config.get('coordconv', {}).get('enabled', False)
    )
    
    if config['model']['pretrained']:
        tao_path = Path(config['paths']['tao_weights']).expanduser().resolve()
        if tao_path.exists():
            try:
                tao_dict = torch.load(tao_path, map_location='cpu')
                tao_weights = tao_dict['state_dict'] if isinstance(tao_dict, dict) and 'state_dict' in tao_dict else tao_dict
                model.load_tao_weights(tao_weights)
            except Exception as e: print(f"Load error: {e}")

    if config['training'].get('freeze_backbone'):
        for name, param in model.named_parameters():
            if 'head' not in name: param.requires_grad = False

    train_loader, val_loader = get_dataloaders(config)
    trainer = Trainer(model, train_loader, val_loader, config)
    should_resume = config['training'].get('resume', False)
    resume_path = config['training'].get('resume_path', "")

    if should_resume:
        if not resume_path:
            resume_path = os.path.join(config['paths']['checkpoint_dir'], 'last_checkpoint.pth')
        
        trainer.load_checkpoint(resume_path)

    trainer.train()

if __name__ == '__main__':
    main()