#!/usr/bin/env python3
"""
Simplified Inference Pipeline for ConvNeXt-V2 Model (Direct Classification)
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
        
        # Class names logic
        self.class_names = self._prepare_class_names(model_config.get('class_names', []))
        
        print(f"Inference engine initialized on {self.device}")
        print(f"Model loaded with classes: {self.class_names}")
    
    def _prepare_class_names(self, class_names_from_cfg: Optional[List[str]]) -> List[str]:
        if class_names_from_cfg and len(class_names_from_cfg) > 0:
            return class_names_from_cfg
            
        classes_txt = Path("dataset/classes.txt")
        if classes_txt.exists():
            with open(classes_txt, 'r') as f:
                content = f.read().strip()
                names = content.split() if ' ' in content else content.splitlines()
                names = [n.strip() for n in names if n.strip()]
                if names: return names
        
        return ['cat', 'dog']
    
    def _load_model(self, model_path: str) -> ConvNeXtV2:
        model = create_model(
            model_name=self.model_config.get('model_name', 'convnextv2_large'),
            num_classes=self.model_config.get('num_classes', 2),
            in_chans=self.model_config.get('in_chans', 3),
            drop_path_rate=self.model_config.get('drop_path_rate', 0.1)
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
        
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        color = colors[idx % len(colors)]
        
        text = f"{pred_class}: {confidence:.2f}"
        overlay = img.copy()
        # cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), color, -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        annotated_dir = "annotated_images"
        os.makedirs(annotated_dir, exist_ok=True)
        fname = f"annotated_{os.path.basename(image)}" if isinstance(image, str) else "annotated_img.jpg"
        save_path = os.path.join(annotated_dir, fname)
        cv2.imwrite(save_path, img)
        return save_path

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        return [self.predict_single(p) for p in tqdm(image_paths, desc="Processing batch")]

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt-V2 Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory for batch processing')
    parser.add_argument('--output', type=str, help='Path to save inference results (JSON file)')
    parser.add_argument('--threshold', type=float, help='Classification threshold (overrides config)')
    parser.add_argument('--device', type=str, help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = {
        'model_name': config['model']['name'],
        'num_classes': config['classes']['num_classes'],
        'in_chans': config['model']['in_chans'],
        'input_size': config['dataset']['input_size'],
        'drop_path_rate': config['model']['drop_path_rate'],
        'use_hist_norm': config['training']['use_hist_norm'],
        'use_bg_masking': config['training']['use_bg_masking'],
        'class_names': config['classes'].get('names', [])
    }
    
    engine = InferenceEngine(
        model_path=args.model_path,
        model_config=model_config,
        device=args.device or config['hardware']['device'],
        threshold=args.threshold or config['inference']['threshold']
    )
    
    if os.path.isfile(args.input):
        results = [engine.predict_single(args.input)]
    else:
        paths = [str(p) for p in Path(args.input).glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        results = engine.predict_batch(paths)
    
    if args.output:
        with open(args.output, 'w') as f: json.dump(results, f, indent=2)

    valid = [r for r in results if 'predicted_class' in r]
    counts = {c: sum(1 for r in valid if r['predicted_class'] == c) for c in engine.class_names}
    print("\nInference Summary:")
    for c, count in counts.items():
        print(f"  {c}: {count} ({100*count/len(valid):.1f}%)" if len(valid)>0 else f"  {c}: 0")

if __name__ == '__main__':
    main()