#!/usr/bin/env python3
"""
Fixed Inference Pipeline with Background Masking
===============================================

Simple inference script that applies background masking during inference
to match training pipeline distribution.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from tqdm import tqdm

# Import custom modules
from model import create_model, ConvNeXtV2
from dataset import create_validation_transforms


class SimpleInferenceEngine:
    """
    Simple inference engine with background masking.
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Dict,
                 device: Optional[str] = None,
                 threshold: float = 0.5):
        self.config = config
        self.threshold = threshold
        print(f'Inference threshold: {self.threshold}')
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup background masking
        self.use_bg_masking = config.get('training', {}).get('use_bg_masking', False)
        self.bg_masking_cfg = config.get('background_masking', {})
        
        # Initialize background masks
        self.bg_masks = {}
        if self.use_bg_masking:
            self._initialize_background_masks()
        
        # Setup transforms (without background masking - handled separately)
        temp_config = {
            'training': {
                'use_hist_norm': config.get('use_hist_norm', False),
                'use_bg_masking': False  # Background masking handled separately
            },
            'dataset': {
                'input_size': config.get('input_size', 224)
            }
        }
        self.transform = create_validation_transforms(temp_config)
        
        # Class names
        self.class_names = ['OK', 'NG']  # Default for binary classification
        
        print(f"Inference engine initialized on {self.device}")
        print(f"Background masking: {'Enabled' if self.use_bg_masking else 'Disabled'}")
    
    def _initialize_background_masks(self):
        """Initialize background masking transforms."""
        try:
            from utils.preprocessing import BackgroundMasking
            
            if self.bg_masking_cfg.get('cg_mask_json'):
                self.bg_masks['cg'] = BackgroundMasking(
                    json_path=self.bg_masking_cfg['cg_mask_json'],
                    always_apply=True
                )
                print(f"Loaded CG background mask: {self.bg_masking_cfg['cg_mask_json']}")
            
            if self.bg_masking_cfg.get('real_mask_json'):
                self.bg_masks['real'] = BackgroundMasking(
                    json_path=self.bg_masking_cfg['real_mask_json'],
                    always_apply=True
                )
                print(f"Loaded Real background mask: {self.bg_masking_cfg['real_mask_json']}")
                
        except Exception as e:
            print(f"Warning: Failed to initialize background masks: {e}")
            self.use_bg_masking = False
    
    def _apply_background_masking(self, image: np.ndarray, image_path: str) -> np.ndarray:
        """Apply appropriate background mask based on image filename."""
        if not self.use_bg_masking:
            return image
        
        filename = os.path.basename(image_path)
        
        try:
            if filename.startswith('cg_') and 'cg' in self.bg_masks:
                return self.bg_masks['cg'](image=image)['image']
            elif filename.startswith('real_') and 'real' in self.bg_masks:
                return self.bg_masks['real'](image=image)['image']
            else:
                return image
        except Exception as e:
            print(f"Warning: Background masking failed for {filename}: {e}")
            return image
    
    def _load_model(self, model_path: str) -> ConvNeXtV2:
        """Load trained model."""
        model = create_model(
            model_name=self.config.get('model_name', 'convnextv2_large'),
            num_classes=self.config.get('num_classes', 2),
            in_chans=self.config.get('in_chans', 3),
            drop_path_rate=self.config.get('drop_path_rate', 0.1)
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """Preprocess image with background masking and transforms."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None: 
                raise ValueError(f"Could not load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_path = image
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            image_path = "unknown_image"
        
        # Apply background masking FIRST
        img = self._apply_background_masking(img, image_path)
        
        # Apply validation transforms
        transformed = self.transform(image=img)
        return transformed['image'].unsqueeze(0)
    
    def predict_single(self, image: Union[str, np.ndarray]) -> Dict:
        """Predict single image with background masking."""
        input_tensor = self.preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        idx = predicted_idx.item()
        conf = confidence.item()
        probs = probabilities.cpu().numpy()[0]
        
        result = {
            'predicted_class': self.class_names[idx],
            'class_index': idx,
            'confidence': float(conf),
            'class_probabilities': {self.class_names[i]: float(p) for i, p in enumerate(probs)},
            'inference_time': inference_time
        }
        if isinstance(image, str):
            result['annotated_image'] = self.create_classification_image(image, result)
        
        return result
    
    def create_classification_image(self, image: Union[str, np.ndarray], prediction: Dict) -> str:
        img = cv2.imread(image) if isinstance(image, str) else image.copy()
        pred_class = prediction['predicted_class']
        confidence = prediction['confidence']
        idx = prediction['class_index']
        
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        color = colors[idx % len(colors)]
        
        text = f"{pred_class}: {confidence:.2f}"
        overlay = img.copy()
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        annotated_dir = "annotated_images"
        os.makedirs(annotated_dir, exist_ok=True)
        fname = f"annotated_{os.path.basename(image)}" if isinstance(image, str) else "annotated_img.jpg"
        save_path = os.path.join(annotated_dir, fname)
        cv2.imwrite(save_path, img)
        return save_path
    
    def preview_image(self, image_path: str) -> str:
        # Load original image
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Could not load image: {image_path}")
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Apply background masking
        masked_img = self._apply_background_masking(orig_img, image_path)
        
        # Create side-by-side comparison
        h, w = orig_img.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Original on left
        comparison[:, :w] = orig_img
        
        # Masked on right
        comparison[:, w:] = masked_img
        

        # Original label
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Masked label
        cv2.putText(comparison, "Masked", (w+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save comparison
        comparison_dir = "Inference_previews"
        os.makedirs(comparison_dir, exist_ok=True)
        fname = f"comparison_{os.path.basename(image_path)}"
        save_path = os.path.join(comparison_dir, fname)
        
        # Convert RGB to BGR for OpenCV
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, comparison_bgr)
        
        return save_path


def main():
    """Main inference function."""
 
    parser = argparse.ArgumentParser(description='Fixed inference with background masking')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Path to image or directory')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--threshold', type=float, help='Classification threshold')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--save_preview', action='store_true', help='Save preprocessed preview images')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize inference engine
    engine = SimpleInferenceEngine(
        model_path=args.model,
        config=config,
        device=device,
        threshold=args.threshold or config['inference']['threshold']
    )
    
    # Process images
    image_path = Path(args.input)
    
    if image_path.is_file():
        # Single image
        print(f"Processing single image: {image_path}")
        
        result = engine.predict_single(str(image_path))
        if args.save_preview:
            preview_path = engine.preview_image(str(image_path))
        
        print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
       
        
    elif image_path.is_dir():
        # Directory of images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_path.glob(ext))
        
        print(f"Processing {len(image_files)} images in directory: {image_path}")
        
        results = []
        for img_file in tqdm(image_files, desc="Processing images"):
            result = engine.predict_single(str(img_file))
            if args.save_preview:
                preview_path = engine.preview_image(str(img_file))
            results.append({
                'image': str(img_file),
                'prediction': result['predicted_class'],
                'confidence': result['confidence']
            })
        
        # Print summary
        ok_count = sum(1 for r in results if r['prediction'] == 'OK')
        ng_count = sum(1 for r in results if r['prediction'] == 'NG')
        
        print(f"\n=== INFERENCE SUMMARY ===")
        print(f"Total images: {len(results)}")
        print(f"OK predictions: {ok_count} ({ok_count/len(results)*100:.1f}%)")
        print(f"NG predictions: {ng_count} ({ng_count/len(results)*100:.1f}%)")
        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")

    else:
        print(f"Error: {image_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
