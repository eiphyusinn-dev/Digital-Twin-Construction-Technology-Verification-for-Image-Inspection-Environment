#!/usr/bin/env python3
"""
DataLoader Visualization Script - Sequentially matching dataset.py logic
"""

import os
import sys
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    ClientCustomDataset, 
    create_training_transforms
)

from utils.preprocessing import FDATransform, HistogramNormalization, BackgroundMasking

import albumentations as A

class DataLoaderVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.root_dir = Path(config['dataset']['root_dir'])
        self.train_dir = Path(config['dataset']['train_dir'])
        self.output_dir = Path(config['paths']['visualization_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_size = config['dataset'].get('input_size', 224)

    def visualize_full_sequence(self, num_images: int):
        print(f"Generating {num_images} sequential visualizations...")
        
        # Initialize dataset without transform to get raw images
        dataset = ClientCustomDataset(
            root_dir=str(self.root_dir),
            split_path= self.root_dir / self.train_dir,
            config=self.config,
            transform=None, 
            class_names=self.config['classes']['names']
        )
        
        if len(dataset) == 0:
            print("No images found in training directory.")
            return
            
        indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
        
        train_cfg = self.config['training']
        aug_cfg = self.config['training']['augmentation'] 
        fda_cfg = self.config.get('fda', {})
        bg_masking_cfg = self.config.get('background_masking', {})

        for i, img_idx in enumerate(indices):
            img_path, label_idx = dataset.samples[img_idx]
            class_name = dataset.class_names[label_idx]
            
            # Load image (original size)
            raw_img = cv2.imread(img_path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            current_img = raw_img.copy()  # Work on original size
            
            # History list for plotting steps - use Lanczos for better quality
            def resize_vis(img):
                return cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LANCZOS4)
            
            history = [('Original', resize_vis(current_img).copy())]

            # history = [('Original', current_img.copy())]


            # Background Masking (on original size) - choose mask based on image type
            if train_cfg.get('use_bg_masking'):
                filename = Path(img_path).name
                orig_h, orig_w = current_img.shape[:2]
                
                if filename.startswith('cg_') and bg_masking_cfg.get('cg_mask_json'):
                    bg_mask = BackgroundMasking(json_path=bg_masking_cfg['cg_mask_json'], 
                                                original_size=(orig_w, orig_h), always_apply=True)
                    current_img = bg_mask(image=current_img)['image']
                    history.append(('+ Mask (CG)', resize_vis(current_img).copy()))
                elif filename.startswith('real_') and bg_masking_cfg.get('real_mask_json'):
                    bg_mask = BackgroundMasking(json_path=bg_masking_cfg['real_mask_json'], 
                                                original_size=(orig_w, orig_h), always_apply=True)
                    current_img = bg_mask(image=current_img)['image']
                    history.append(('+ Mask (Real)', resize_vis(current_img).copy()))
       
            # FDA (on original size, only for images matching filename_filter prefix)
            filename = Path(img_path).name
            filename_filter = fda_cfg.get('filename_filter', 'cg_')  # Default to 'cg_' if not set
            if filename.startswith(filename_filter) and train_cfg.get('use_fda', False) and fda_cfg.get('reference_dir'):
                # Random beta sampling from range
                beta_range = fda_cfg.get('beta_range', [0.05, 0.05])
                if isinstance(beta_range, list) and len(beta_range) == 2:
                    beta = np.random.uniform(beta_range[0], beta_range[1])
                else:
                    beta = beta_range[0] if isinstance(beta_range, list) else beta_range
                    
                fda_tf = FDATransform(fda_cfg['reference_dir'], beta_limit=beta, always_apply=True)
                current_img = fda_tf(image=current_img)['image']
                history.append(('+ FDA', resize_vis(current_img).copy()))

            # Histogram Norm (GHE) (on original size)
            if train_cfg.get('use_hist_norm'):
                current_img = HistogramNormalization(p=1.0)(image=current_img)['image']
                history.append(('+ GHE', resize_vis(current_img).copy()))
            
            # NOW Resize after all preprocessing
            current_img = cv2.resize(current_img, (self.input_size, self.input_size))

            # --- PART 2: TRAINING AUGMENTATIONS (One-by-One) ---
            def apply_and_log(img, transform, label):
                result = transform(image=img)['image']
                # Check if the pixels actually changed
                if not np.array_equal(img, result):
                    history.append((label, result.copy()))
                    return result
                return img

            if aug_cfg.get('horizontal_flip', 0) > 0:
                current_img = apply_and_log(
                    current_img, 
                    A.HorizontalFlip(p=aug_cfg['horizontal_flip']), 
                    '+ H-Flip'
                )

            if aug_cfg.get('rotation', 0) > 0:
                current_img = apply_and_log(
                    current_img, 
                    A.Rotate(limit=50, p=aug_cfg['rotation']), 
                    '+ Rotate'
                )

            if aug_cfg.get('brightness', 0) > 0:
                current_img = apply_and_log(
                    current_img,
                    A.RandomBrightnessContrast(
                        brightness_limit=aug_cfg['brightness'], 
                        contrast_limit=aug_cfg.get('contrast', 0), 
                        p=aug_cfg['brightness']
                    ),
                    '+ Bright'
                )

            if aug_cfg.get('gaussian_blur', 0) > 0:
                current_img = apply_and_log(
                    current_img,
                    A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg['gaussian_blur']),
                    '+ Blur'
                )
            if aug_cfg.get('gaussian_noise', 0) > 0:
                current_img = apply_and_log(
                    current_img,
                    A.GaussNoise(var_range=(10.0, 50.0), p=1.0),
                    '+ Noise'
                )
            
            # Note: MixUp and CutMix are batch-level augmentations and cannot be visualized 
            # in single image mode. They are applied during training in batches.
            # See config.yaml to enable them:
            # mixup.enabled: true
            # cutmix.enabled: true
            
            # Color Jitter
            if aug_cfg.get('color_jitter', {}).get('enabled', False):
                color_cfg = aug_cfg['color_jitter']
                current_img = apply_and_log(
                    current_img,
                    A.ColorJitter(
                        brightness=color_cfg.get('brightness', 0.2),
                        contrast=color_cfg.get('contrast', 0.2),
                        saturation=color_cfg.get('saturation', 0.2),
                        hue=color_cfg.get('hue', 0.1),
                        p=1.0
                    ),
                    '+ ColorJitter'
                )
            
            # RandAugment (using direct augmentations)
            if aug_cfg.get('randaugment', {}).get('enabled', False):
                rand_cfg = aug_cfg['randaugment']
                n = rand_cfg.get('n', 2)
                m = rand_cfg.get('m', 9)
                
                # Apply n random augmentations directly
                augmentations = [
                    (A.RandomBrightnessContrast(brightness_limit=min(0.15, m/20.0), contrast_limit=min(0.15, m/20.0), p=1.0), "Brightness"),
                    (A.RandomGamma(gamma_limit=(max(70, 100-m*1), min(130, 100+m*1)), p=1.0), "Gamma"),
                    (A.GaussianBlur(blur_limit=(1, max(2, m//5)), p=1.0), "Blur"),
                    (A.GaussNoise(std_range=(0.03, 0.03+m*0.01), p=1.0), "Noise"),
                    (A.RandomRotate90(p=1.0), "Rotate90"),
                    # (A.RandomGridShuffle(grid=(2, 2), p=1.0), "GridShuffle")  # Commented out
                ]
                
                # Randomly select n augmentations
                selected = np.random.choice(len(augmentations), min(n, len(augmentations)), replace=False)
                
                for idx in selected:
                    aug, name = augmentations[idx]
                    current_img = aug(image=current_img)['image']
                
                history.append(('RandAugment', current_img.copy()))
                
            # --- SIMULATE MixUp and CutMix for visualization ---
            aug_cfg = self.config['training']['augmentation']
            
            # Simulate MixUp
            if aug_cfg.get('mixup', {}).get('enabled', False):
                mixup_cfg = aug_cfg['mixup']
                alpha = mixup_cfg.get('alpha', 0.2)
                
                # Get a second random image for realistic MixUp
                if len(dataset.samples) > 1:
                    second_idx = np.random.choice([i for i in range(len(dataset.samples)) if i != img_idx])
                    second_img_path, _ = dataset.samples[second_idx]
                    second_img = cv2.imread(second_img_path)
                    second_img = cv2.cvtColor(second_img, cv2.COLOR_BGR2RGB)
                    second_img = cv2.resize(second_img, (current_img.shape[1], current_img.shape[0]))
                    
                    # Create realistic MixUp
                    lam = np.random.beta(alpha, alpha)
                    lam = np.clip(lam, 0.7, 0.95)  # 70-95% original, 5-30% second
                    # Ensure minimum mixing effect - avoid lambda too close to 0 or 1
                    mixed_img = (lam * current_img + (1 - lam) * second_img).astype(np.uint8)
                    
                    
                    history.append(('MixUp Sim\n(λ={:.2f})'.format(lam), mixed_img.copy()))
                else:
                    # Fallback: use same image
                    lam = np.random.beta(alpha, alpha)
                    lam = np.clip(lam, 0.7, 0.95)  # 70-95% original, 5-30% second
                    mixed_img = (lam * current_img + (1 - lam) * current_img).astype(np.uint8)
                    history.append(('MixUp Sim\n(λ={:.2f})'.format(lam), mixed_img.copy()))
            
            # Simulate CutMix
            if aug_cfg.get('cutmix', {}).get('enabled', False):
                cutmix_cfg = aug_cfg['cutmix']
                alpha = cutmix_cfg.get('alpha', 1.0)
                
                # Get a second random image for realistic CutMix
                if len(dataset.samples) > 1:
                    second_idx = np.random.choice([i for i in range(len(dataset.samples)) if i != img_idx])
                    second_img_path, _ = dataset.samples[second_idx]
                    second_img = cv2.imread(second_img_path)
                    second_img = cv2.cvtColor(second_img, cv2.COLOR_BGR2RGB)
                    second_img = cv2.resize(second_img, (current_img.shape[1], current_img.shape[0]))
                    
                    # Create realistic CutMix
                    lam = np.random.beta(alpha, alpha)
                    lam = np.clip(lam, 0.7, 0.95)  # 70-95% original, 5-30% second
                    h, w = current_img.shape[:2]
                    
                    # Generate random bounding box
                    cut_rat = np.sqrt(1. - lam)
                    cut_w = int(w * cut_rat)
                    cut_h = int(h * cut_rat)
                    
                    # Random position
                    cx = np.random.randint(w)
                    cy = np.random.randint(h)
                    
                    bbx1 = np.clip(cx - cut_w // 2, 0, w)
                    bby1 = np.clip(cy - cut_h // 2, 0, h)
                    bbx2 = np.clip(cx + cut_w // 2, 0, w)
                    bby2 = np.clip(cy + cut_h // 2, 0, h)
                    
                    # Apply CutMix with second image
                    cutmix_img = current_img.copy()
                    cutmix_img[bbx1:bbx2, bby1:bby2] = second_img[bbx1:bbx2, bby1:bby2]
                    history.append(('CutMix Sim\n(λ={:.2f})'.format(lam), cutmix_img.copy()))
                else:
                    # Fallback: use same image
                    lam = np.random.beta(alpha, alpha)
                    lam = np.clip(lam, 0.7, 0.95)  # 70-95% original, 5-30% second
                    h, w = current_img.shape[:2]
                    
                    # Generate random bounding box
                    cut_rat = np.sqrt(1. - lam)
                    cut_w = int(w * cut_rat)
                    cut_h = int(h * cut_rat)
                    
                    # Random position
                    cx = np.random.randint(w)
                    cy = np.random.randint(h)
                    
                    bbx1 = np.clip(cx - cut_w // 2, 0, w)
                    bby1 = np.clip(cy - cut_h // 2, 0, h)
                    bbx2 = np.clip(cx + cut_w // 2, 0, w)
                    bby2 = np.clip(cy + cut_h // 2, 0, h)
                    
                    # Apply CutMix (using same image for simulation)
                    cutmix_img = current_img.copy()
                    cutmix_img[bbx1:bbx2, bby1:bby2] = current_img[bbx1:bbx2, bby1:bby2]
                    history.append(('CutMix Sim\n(λ={:.2f})'.format(lam), cutmix_img.copy()))

            if aug_cfg.get('vertical_flip', 0) > 0:
                current_img = apply_and_log(
                    current_img, 
                    A.VerticalFlip(p=aug_cfg['vertical_flip']), 
                    '+ V-Flip'
                )
            if aug_cfg.get('saturation', 0) > 0 or aug_cfg.get('hue', 0) > 0:
                current_img = apply_and_log(
                    current_img,
                    A.HueSaturationValue(
                        hue_shift_limit=int(aug_cfg.get('hue', 0.1) * 180),
                        sat_shift_limit=int(aug_cfg.get('saturation', 0.2) * 255),
                        p=max(aug_cfg.get('saturation', 0.2), aug_cfg.get('hue', 0.1))
                    ),
                    '+ Hue/Sat'
                )
            # --- FINAL PLOTTING ---
            # Add real model input to history
            # Apply the complete transform pipeline to get the actual model input
            transform = create_training_transforms(self.config)
            transformed = transform(image=current_img)
            model_input = transformed['image']  # This is the tensor that goes to model
            
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            denormalized = model_input * std + mean
            denormalized = torch.clamp(denormalized, 0, 1)
            
            # Convert to numpy for plotting
            model_input_np = denormalized.permute(1, 2, 0).cpu().numpy()
            history.append(('Model Input\n(Normalized)', model_input_np))
            
            num_steps = len(history)
            fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 5))
            
            # Ensure axes is iterable if only 1 step
            if num_steps == 1: axes = [axes]

            for idx, (label, img_state) in enumerate(history):
                axes[idx].imshow(img_state)
                axes[idx].set_title(label, fontsize=12, fontweight='bold')
                axes[idx].axis('off')

            plt.suptitle(f"Sample {i+1}: {class_name} | {Path(img_path).name}", fontsize=14)
            
            file_stem = Path(img_path).stem
            save_path = self.output_dir / f"seq_{i+1}_{file_stem}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated: {save_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='Number of input images to visualize')
    args = parser.parse_args()

    # Load YAML
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    visualizer = DataLoaderVisualizer(config)
    visualizer.visualize_full_sequence(num_images=args.n)

if __name__ == '__main__':
    main()