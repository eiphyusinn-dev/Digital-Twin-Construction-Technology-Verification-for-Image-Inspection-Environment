import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

# Import custom modules
from model import create_model, ConvNeXtV2
from dataset import create_validation_transforms

class InferenceEngine:
    def __init__(self, 
                 model_path: str,
                 model_config: Dict,
                 device: Optional[str] = None,
                 threshold: float = 0.5,
                 patch_size: int = 224,
                 stride: int = 112):
        self.model_config = model_config
        self.threshold = threshold
        self.patch_size = patch_size
        self.stride = stride
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Standard transforms for the 224x224 patches
        temp_config = {
            'training': {
                'use_hist_norm': model_config.get('use_hist_norm', False)
            },
            'dataset': {'input_size': patch_size}
        }
        self.transform = create_validation_transforms(temp_config)
        self.class_names = model_config.get('class_names', ['NG', 'OK'])
    
    def _load_model(self, model_path: str) -> ConvNeXtV2:
        model = create_model(
            model_name=self.model_config.get('model_name', 'convnextv2_large'),
            num_classes=self.model_config.get('num_classes', 2),
            in_chans=self.model_config.get('in_chans', 3)
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model

    def get_patches(self, img: np.ndarray) -> List[Dict]:
        """Splits image into patches and keeps track of coordinates."""
        h, w = img.shape[:2]
        patches = []
        
        y_coords = list(range(0, h - self.patch_size + 1, self.stride))
        if y_coords[-1] != h - self.patch_size: y_coords.append(h - self.patch_size)
        x_coords = list(range(0, w - self.patch_size + 1, self.stride))
        if x_coords[-1] != w - self.patch_size: x_coords.append(w - self.patch_size)

        for y in y_coords:
            for x in x_coords:
                patch = img[y:y+self.patch_size, x:x+self.patch_size]
                # Filter out empty/black patches (optional)
                if np.count_nonzero(patch) < (self.patch_size * self.patch_size * 0.1):
                    continue
                patches.append({'img': patch, 'x': x, 'y': y})
        return patches

    def predict_full_image(self, image_path: str) -> Dict:
        """Slices image, predicts on patches, and aggregates results."""
        original_img = cv2.imread(image_path)
        if original_img is None: raise ValueError(f"Could not load {image_path}")
        
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        patches = self.get_patches(img_rgb)
        
        ng_patches = []
        max_ng_confidence = 0.0
        
        start_time = time.time()
        
        # Batch process patches for speed
        batch_size = 16
        for i in range(0, len(patches), batch_size):
            batch_data = patches[i:i+batch_size]
            batch_tensors = []
            for p in batch_data:
                transformed = self.transform(image=p['img'])['image']
                batch_tensors.append(transformed)
            
            input_batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_batch)
                probs = F.softmax(outputs, dim=1)
                
            for j, prob in enumerate(probs):
                # Assuming index 1 is 'NG' and index 0 is 'OK'
                # Adjust based on your dataset.class_to_idx
                ng_conf = prob[0].item() 
                
                if ng_conf >= self.threshold:
                    patch_info = batch_data[j]
                    patch_info['confidence'] = ng_conf
                    ng_patches.append(patch_info)
                    max_ng_confidence = max(max_ng_confidence, ng_conf)

        inference_time = time.time() - start_time
        overall_label = "NG" if len(ng_patches) > 0 else "OK"
        
        # Create visualization
        vis_path = self.create_visual_result(original_img, ng_patches, overall_label, image_path)

        return {
            'filename': os.path.basename(image_path),
            'overall_prediction': overall_label,
            'num_ng_patches': len(ng_patches),
            'max_confidence': max_ng_confidence if overall_label == "NG" else (1 - max_ng_confidence),
            'inference_time': inference_time,
            'annotated_image': vis_path
        }

    def create_visual_result(self, img, ng_patches, label, original_path):
        vis_img = img.copy()
        overlay = img.copy()
        
        # Highlight all NG patches found
        for patch in ng_patches:
            x, y = patch['x'], patch['y']
            cv2.rectangle(overlay, (x, y), (x + self.patch_size, y + self.patch_size), (0, 0, 255), -1)
            cv2.rectangle(vis_img, (x, y), (x + self.patch_size, y + self.patch_size), (0, 255, 255), 2)

        # Apply transparency
        cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
        
        # Draw Summary Header
        color = (0, 0, 255) if label == "NG" else (0, 255, 0)
        cv2.rectangle(vis_img, (0, 0), (img.shape[1], 50), color, -1)
        cv2.putText(vis_img, f"RESULT: {label} | NG Patches: {len(ng_patches)}", 
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        output_dir = Path("inference_results")
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / f"res_{os.path.basename(original_path)}"
        cv2.imwrite(str(save_path), vis_img)
        return str(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    engine = InferenceEngine(
        model_path=args.model_path,
        model_config={
            'model_name': config['model']['name'],
            'num_classes': config['classes']['num_classes'],
            'in_chans': config['model']['in_chans'],
            'class_names': config['classes'].get('names', ['NG', 'OK'])
        },
        threshold=config['inference'].get('threshold', 0.5),
        stride=112 # Overlap patches for better coverage
    )

    # Process
    input_path = Path(args.input)
    if input_path.is_file():
        files = [str(input_path)]
    else:
        files = [str(p) for p in input_path.glob('*') if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    all_results = []
    for f in tqdm(files):
        res = engine.predict_full_image(f)
        all_results.append(res)
        print(f"File: {res['filename']} -> {res['overall_prediction']} ({res['num_ng_patches']} defects)")

    with open("results.json", "w") as jf:
        json.dump(all_results, jf, indent=4)

if __name__ == '__main__':
    main()