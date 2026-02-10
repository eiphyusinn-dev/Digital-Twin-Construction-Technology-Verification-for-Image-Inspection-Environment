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
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    ClientCustomDataset, 
    FDATransform, 
    HistogramNormalization, 
    BackgroundMasking
)
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

        for i, img_idx in enumerate(indices):
            img_path, label_idx = dataset.samples[img_idx]
            class_name = dataset.class_names[label_idx]
            
            # Load and initial resize
            raw_img = cv2.imread(img_path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            current_img = cv2.resize(raw_img, (self.input_size, self.input_size))
            
            # History list for plotting steps
            history = [('Original', current_img.copy())]

       
            # FDA
            if train_cfg.get('use_fda') and fda_cfg.get('reference_dir'):
                beta = fda_cfg.get('beta_range', [0.05])[0]
                fda_tf = FDATransform(fda_cfg['reference_dir'], beta_limit=beta, always_apply=True)
                current_img = fda_tf(image=current_img)['image']
                history.append(('+ FDA', current_img.copy()))

            # Histogram Norm (GHE)
            if train_cfg.get('use_hist_norm'):
                current_img = HistogramNormalization(p=1.0)(image=current_img)['image']
                history.append(('+ GHE', current_img.copy()))

            # Background Masking
            if train_cfg.get('use_bg_masking'):
                current_img = BackgroundMasking(p=1.0)(image=current_img)['image']
                history.append(('+ BG Mask', current_img.copy()))

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
                    A.GaussNoise(var_limit=(10.0, 50.0), p=aug_cfg['gaussian_noise']),
                    '+ Noise'
                )
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