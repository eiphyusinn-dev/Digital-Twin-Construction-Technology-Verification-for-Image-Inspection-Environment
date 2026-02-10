import torch
import numpy as np
import cv2
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from utils.preprocessing import FDATransform,HistogramNormalization,BackgroundMasking

class ClientCustomDataset(Dataset):
    """
    Supports ImageFolder format with custom preprocessing transforms.
    """
    def __init__(self, 
                 split_path: Path,
                 root_dir: Path,
                 transform: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None):
        self.split_path = Path(split_path)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Determine class names
        self.class_names = self._prepare_class_names(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        self._load_samples()

    def _prepare_class_names(self, class_names_from_cfg: Optional[List[str]]) -> List[str]:
        # 1. Use config names if provided
        if class_names_from_cfg and len(class_names_from_cfg) > 0:
            return class_names_from_cfg
            
        classes_txt = self.root_dir / "classes.txt"
        if classes_txt.exists():
            with open(classes_txt, 'r') as f:
                names = [line.strip() for line in f if line.strip()]
                print(f'Names from classes.txt: {names}')
                if names: return sorted(names)

        
        # 3. Fallback: Auto-detect from folders in the split path
        print(f"Auto-detecting classes from directory: {self.split_path}")
        return sorted([d.name for d in self.split_path.iterdir() if d.is_dir()])

    def _load_samples(self):
        """Load image samples from split directory structure."""
        if not self.split_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.split_path}")
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.split_path / class_name
            if not class_dir.exists():
                continue
            
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Loaded {len(self.samples)} samples for path: {self.split_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, label

def create_training_transforms(config: Dict) -> A.Compose:
    """
    Builds the pipeline based on the YAML config. 
    Only includes augmentations with values > 0.
    """
    transforms_list = []
    
    # --- 1. Preprocessing (Conditional on training flags) ---
    train_cfg = config.get('training', {})
    fda_cfg = config.get('fda', {})
    
    if train_cfg.get('use_fda') and fda_cfg.get('reference_dir'):
        # Use first beta value if provided
        beta = fda_cfg.get('beta_range', [0.05])[0]
        transforms_list.append(FDATransform(fda_cfg['reference_dir'], beta_limit=beta, always_apply=True))
    
    if train_cfg.get('use_hist_norm'):
        transforms_list.append(HistogramNormalization(p=1.0))
    
    if train_cfg.get('use_bg_masking'):
        transforms_list.append(BackgroundMasking(p=1.0))
    
    # --- 2. Geometric & Color Augmentations---
    aug_cfg = config['training'].get('augmentation', {})
    input_size = config['dataset'].get('input_size', 224)

    # Always Resize first
    transforms_list.append(A.Resize(input_size, input_size))

    # Check Horizontal Flip
    if aug_cfg.get('horizontal_flip', 0) > 0:
        transforms_list.append(A.HorizontalFlip(p=aug_cfg['horizontal_flip']))

    # Check Vertical Flip
    if aug_cfg.get('vertical_flip', 0) > 0:
        transforms_list.append(A.VerticalFlip(p=aug_cfg['vertical_flip']))

    # Check Rotation
    if aug_cfg.get('rotation', 0) > 0:
        transforms_list.append(A.Rotate(limit=15, p=aug_cfg['rotation']))

    # Check Brightness/Contrast
    if aug_cfg.get('brightness', 0) > 0 or aug_cfg.get('contrast', 0) > 0:
        transforms_list.append(A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get('brightness', 0.2),
            contrast_limit=aug_cfg.get('contrast', 0.2),
            p=aug_cfg.get('brightness', 0.2)
        ))

    # Check Saturation/Hue
    if aug_cfg.get('saturation', 0) > 0 or aug_cfg.get('hue', 0) > 0:
        transforms_list.append(A.HueSaturationValue(
            hue_shift_limit=int(aug_cfg.get('hue', 0.1) * 180),
            sat_shift_limit=int(aug_cfg.get('saturation', 0.2) * 255),
            p=max(aug_cfg.get('saturation', 0.2), aug_cfg.get('hue', 0.1))
        ))

    # Check Gaussian Blur
    if aug_cfg.get('gaussian_blur', 0) > 0:
        transforms_list.append(A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg['gaussian_blur']))

    # Check Gaussian Noise
    if aug_cfg.get('gaussian_noise', 0) > 0:
        transforms_list.append(A.GaussNoise(var_limit=(10.0, 50.0), p=aug_cfg['gaussian_noise']))

    # --- 3. Final Required Steps ---
    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)

def create_validation_transforms(config: Dict) -> A.Compose:
    """
    Standard validation pipeline.
    """
    transforms_list = []
    train_cfg = config['training']
    input_size = config['dataset']['input_size']
    

    if train_cfg.get('use_hist_norm'):
        transforms_list.append(HistogramNormalization(p=1.0))
    if train_cfg.get('use_bg_masking'):
        transforms_list.append(BackgroundMasking())

    transforms_list.extend([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Main entry point. Correctly joins root_dir with split_dir.
    """
    ds_cfg = config['dataset']
    root_path = Path(ds_cfg['root_dir'])
    
    train_transform = create_training_transforms(config)
    val_transform = create_validation_transforms(config)
    
    # Correctly join paths: root_dir / train_dir
    train_dataset = ClientCustomDataset(
        split_path=root_path / ds_cfg['train_dir'],
        root_dir=root_path,
        transform=train_transform,
        class_names=config['classes']['names']
    )
    
    val_dataset = ClientCustomDataset(
        split_path=root_path / ds_cfg['val_dir'],
        root_dir=root_path,
        transform=val_transform,
        class_names=config['classes']['names']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=ds_cfg['batch_size'],
        shuffle=True,
        num_workers=ds_cfg['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=ds_cfg['batch_size'],
        shuffle=False,
        num_workers=ds_cfg['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader