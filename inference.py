#!/usr/bin/env python3
"""
Inference Pipeline for ConvNeXt-V2 Model
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
import logging

# Import custom modules
from model import create_model, ConvNeXtV2
from dataset import create_validation_transforms, ClientCustomDataset


class InferenceEngine:
    """
    Inference engine for ConvNeXt-V2 model.
    """
    
    def __init__(self, 
                 model_path: str,
                 model_config: Dict,
                 device: Optional[str] = None,
                 threshold: float = 0.5):
        self.model_config = model_config
        self.threshold = threshold
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        temp_config = {
            'training': {
                'use_hist_norm': model_config.get('use_hist_norm', False),
                'use_bg_masking': model_config.get('use_bg_masking', False)
            },
            'dataset': {
                'input_size': model_config.get('input_size', 224)
            }
        }
        self.transform = create_validation_transforms(temp_config)
        
        # Class names logic: Config -> classes.txt -> Fallback
        self.class_names = self._prepare_class_names(model_config.get('class_names', []))
        
        print(f"Inference engine initialized on {self.device}")
        print(f"Model loaded with classes: {self.class_names}")
    
    def _prepare_class_names(self, class_names_from_cfg: Optional[List[str]]) -> List[str]:
        # 1. Use config names if provided
        if class_names_from_cfg and len(class_names_from_cfg) > 0:
            return class_names_from_cfg
            
        # 2. Try to read from classes.txt in dataset root
        classes_txt = Path("dataset/classes.txt")
        if classes_txt.exists():
            with open(classes_txt, 'r') as f:
                # Assuming classes.txt can be space-separated or line-separated
                content = f.read().strip()
                names = content.split() if ' ' in content else content.splitlines()
                names = [n.strip() for n in names if n.strip()]
                if names: 
                    print(f"Loaded classes from {classes_txt}: {names}")
                    return names
        
        # 3. Fallback
        return ['cat', 'dog']
    
    def _load_model(self, model_path: str) -> ConvNeXtV2:
        model = create_model(
            model_name=self.model_config.get('model_name', 'convnextv2_large'),
            num_classes=self.model_config.get('num_classes', 2),
            in_chans=self.model_config.get('in_chans', 3),
            drop_path_rate=self.model_config.get('drop_path_rate', 0.1),
            use_coordconv=self.model_config.get('use_coordconv', False)
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None: raise ValueError(f"Could not load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        transformed = self.transform(image=img)
        return transformed['image'].unsqueeze(0)
    
    def predict_single(self, image: Union[str, np.ndarray]) -> Dict:
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
        
        # Color mapping: Cycle through a set of colors for different classes
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        color = colors[idx % len(colors)]
        
        text = f"{pred_class}: {confidence:.2f}"
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), color, -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        annotated_dir = "annotated_images"
        os.makedirs(annotated_dir, exist_ok=True)
        fname = f"annotated_{os.path.basename(image)}" if isinstance(image, str) else "annotated_patch.jpg"
        save_path = os.path.join(annotated_dir, fname)
        cv2.imwrite(save_path, img)
        return save_path

    def sliding_window_inference(self, image: Union[str, np.ndarray], patch_size: int = 224, 
                               overlap: float = 0.5, aggregation_method: str = 'max') -> Dict:
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        h, w = img.shape[:2]
        stride = int(patch_size * (1 - overlap))
        patch_predictions = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img[y:y+patch_size, x:x+patch_size]
                patch_predictions.append(self.predict_single(patch))
        
        if aggregation_method == 'max':
            # Aggregates by the highest confidence across any class
            best_patch = max(patch_predictions, key=lambda x: x['confidence'])
            final_result = best_patch.copy()
        else:
            # Average probabilities
            avg_probs = {c: np.mean([p['class_probabilities'][c] for p in patch_predictions]) for c in self.class_names}
            max_class = max(avg_probs, key=avg_probs.get)
            final_result = {
                'predicted_class': max_class,
                'confidence': avg_probs[max_class],
                'class_probabilities': avg_probs
            }
        
        final_result['num_patches'] = len(patch_predictions)
        return final_result

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        return [self.predict_single(p) for p in tqdm(image_paths, desc="Processing batch")]

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt-V2 Inference')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--device', type=str)
    parser.add_argument('--sliding-window', action='store_true')
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--aggregation', type=str, default='max', choices=['max', 'avg'])
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = {
        'model_name': config['model']['name'],
        'num_classes': config['classes']['num_classes'],
        'in_chans': config['model']['in_chans'],
        'input_size': config['dataset']['input_size'],
        'drop_path_rate': config['model']['drop_path_rate'],
        'use_coordconv': config['model']['use_coordconv'],
        'use_hist_norm': config['training']['use_hist_norm'],
        'use_bg_masking': config['training']['use_bg_masking'],
        'class_names': config['classes'].get('names', [])
    }
    
    engine = InferenceEngine(
        model_path=args.model_path,
        model_config=model_config,
        device=args.device or config['inference']['device'],
        threshold=args.threshold or config['inference']['threshold']
    )
    
    if os.path.isfile(args.input):
        results = [engine.sliding_window_inference(args.input, args.patch_size, args.overlap, args.aggregation)] if args.sliding_window else [engine.predict_single(args.input)]
    else:
        paths = [str(p) for p in Path(args.input).glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        results = engine.predict_batch(paths)
    
    if args.output:
        with open(args.output, 'w') as f: json.dump(results, f, indent=2)

    # Print Distribution Summary
    valid = [r for r in results if 'predicted_class' in r]
    counts = {c: sum(1 for r in valid if r['predicted_class'] == c) for c in engine.class_names}
    print("\nInference Summary:")
    for c, count in counts.items():
        print(f"  {c}: {count} ({100*count/len(valid):.1f}%)" if len(valid)>0 else f"  {c}: 0")

if __name__ == '__main__':
    main()