import torch
import numpy as np
import cv2
import os
import yaml
import re
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
                 config: Dict,
                 transform: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None):
        self.split_path = Path(split_path)
        self.root_dir = Path(root_dir)
        self.config = config
        self.transform = transform
        self.samples = []
        
        # Cache config sections for performance
        self.bg_masking_cfg = config.get('background_masking', {})
        self.fda_cfg = config.get('fda', {})
        self.train_cfg = config.get('training', {})
        
        # Load CoordConv settings
        self.coordconv_cfg = config.get('coordconv', {})
        self.use_coordconv = self.coordconv_cfg.get('enabled', False)
        if self.use_coordconv:
            self.grid_rows = self.coordconv_cfg.get('grid_rows', 1)
            self.grid_cols = self.coordconv_cfg.get('grid_cols', 1)
        
        # Initialize transforms once for performance
        self.bg_mask_cg = None
        self.bg_mask_real = None
        self.fda_transform = None
        
        # Initialize background masking transforms
        if self.train_cfg.get('use_bg_masking', False):
            from utils.preprocessing import BackgroundMasking
            if self.bg_masking_cfg.get('cg_mask_json'):
                self.bg_mask_cg = BackgroundMasking(
                    json_path=self.bg_masking_cfg['cg_mask_json'],
                    always_apply=True
                )
            if self.bg_masking_cfg.get('real_mask_json'):
                self.bg_mask_real = BackgroundMasking(
                    json_path=self.bg_masking_cfg['real_mask_json'],
                    always_apply=True
                )
        
        # Initialize FDA transform
        if self.train_cfg.get('use_fda', False) and self.fda_cfg.get('reference_dir'):
            from utils.preprocessing import FDATransform
            beta_range = self.fda_cfg.get('beta_range', [0.05, 0.05])
            beta_init = beta_range[0] if isinstance(beta_range, list) else beta_range
            self.fda_transform = FDATransform(
                self.fda_cfg['reference_dir'],
                beta_limit=beta_init,
                always_apply=True
            )
            self.beta_range = beta_range
        
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

    def _parse_patch_coords(self, filepath: str) -> Tuple[float, float]:
        """
        Extract patch grid coordinates from filename and normalize to 0-1.
        """
        filename = Path(filepath).stem  # Filename without extension
        match = re.search(r'_r(\d+)_c(\d+)_', filename)

        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            # Normalize to 0-1 (avoid division by zero when grid_rows/cols is 1)
            row_norm = row / max(self.grid_rows - 1, 1)
            col_norm = col / max(self.grid_cols - 1, 1)
            
            # Safety clipping to prevent overflow if grid size changes
            row_norm = np.clip(row_norm, 0.0, 1.0)
            col_norm = np.clip(col_norm, 0.0, 1.0)

            return (row_norm, col_norm)
        else:
            # Default to center when coordinates not found in filename
            return (0.5, 0.5)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Conditional Background Masking based on image type
        filename = Path(img_path).name
        
        if self.train_cfg.get('use_bg_masking', False):
            if filename.startswith('cg_') and self.bg_mask_cg:
                image = self.bg_mask_cg(image=image)['image']
            elif filename.startswith('real_') and self.bg_mask_real:
                image = self.bg_mask_real(image=image)['image']
        
        # Conditional FDA: only apply to images matching filename_filter prefix
        filename_filter = self.fda_cfg.get('filename_filter', 'cg_')
        if filename.startswith(filename_filter) and self.train_cfg.get('use_fda', False) and self.fda_transform:
            # Update beta dynamically if range is provided
            if isinstance(self.beta_range, list) and len(self.beta_range) == 2:
                self.fda_transform.beta = np.random.uniform(self.beta_range[0], self.beta_range[1])
            image = self.fda_transform(image=image)['image']
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # CoordConv: Return coordinate tensor as additional output
        if self.use_coordconv:
            row_norm, col_norm = self._parse_patch_coords(img_path)
            coords = torch.tensor([row_norm, col_norm], dtype=torch.float32)
            return image, label, coords

        return image, label

def create_training_transforms(config: Dict) -> A.Compose:
    """
    Builds the pipeline based on the YAML config. 
    Only includes augmentations with values > 0.
    """
    transforms_list = []
    
    # --- 1. Preprocessing on Original Size (before Resize) ---
    train_cfg = config.get('training', {})
    fda_cfg = config.get('fda', {})
    
    # Background masking is handled in __getitem__ - not here
    # FDA is handled in __getitem__ - not here
    
    if train_cfg.get('use_hist_norm'):
        transforms_list.append(HistogramNormalization(p=1.0))
    
    # --- 2. Resize AFTER preprocessing ---
    input_size = config['dataset'].get('input_size', 224)
    transforms_list.append(A.Resize(input_size, input_size))
    
    # --- 3. Geometric & Color Augmentations ---
    aug_cfg = config['training'].get('augmentation', {})
    
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
        transforms_list.append(A.GaussNoise(var_range=(10.0, 50.0), p=aug_cfg['gaussian_noise']))
    
    # Advanced Augmentations
    # Color Jitter
    if aug_cfg.get('color_jitter', {}).get('enabled', False):
        color_cfg = aug_cfg['color_jitter']
        transforms_list.append(A.ColorJitter(
            brightness=color_cfg.get('brightness', 0.2),
            contrast=color_cfg.get('contrast', 0.2),
            saturation=color_cfg.get('saturation', 0.2),
            hue=color_cfg.get('hue', 0.1),
            p=1.0
        ))
    
    # RandAugment (using A.SomeOf for dynamic per-image selection)
    if aug_cfg.get('randaugment', {}).get('enabled', False):
        rand_cfg = aug_cfg['randaugment']
        n = rand_cfg.get('n', 2)
        m = rand_cfg.get('m', 9)
        
        # Create a list of augmentations to choose from (applied dynamically per image)
        augmentations = [
            A.RandomBrightnessContrast(brightness_limit=min(0.15, m/20.0), contrast_limit=min(0.15, m/20.0), p=1.0),
            A.RandomGamma(gamma_limit=(max(70, 100-m*1), min(130, 100+m*1)), p=1.0),
            A.GaussianBlur(blur_limit=(1, max(2, m//5)), p=1.0),
            A.GaussNoise(std_range=(0.03, 0.03+m*0.01), p=1.0),
            A.RandomRotate90(p=1.0),
        ]
        
        # Use A.SomeOf to randomly select n augmentations per image dynamically
        transforms_list.append(A.SomeOf(augmentations, n=n, p=1.0))
    
    # Apply Normalization AFTER augmentations
    transforms_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # --- 3. Final Required Steps ---
    transforms_list.extend([
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)

def create_validation_transforms(config: Dict) -> A.Compose:
    """
    Standard validation pipeline.
    """
    transforms_list = []
    train_cfg = config.get('training', {})
    input_size = config['dataset']['input_size']
    
    # Background masking is handled in __getitem__ - not here
    if train_cfg.get('use_hist_norm'):
        transforms_list.append(HistogramNormalization(p=1.0))

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
        config=config,
        transform=train_transform,
        class_names=config['classes']['names']
    )
    
    val_dataset = ClientCustomDataset(
        split_path=root_path / ds_cfg['val_dir'],
        root_dir=root_path,
        config=config,
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