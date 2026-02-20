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
from utils.preprocessing import BackgroundMasking

class InferenceEngine:
    def __init__(self, 
                 model_path: str,
                 model_config: Dict,
                 full_config: Dict,
                 device: Optional[str] = None,
                 threshold: float = 0.5,
                 patch_size: int = 224,
                 stride: int = 112):
        self.model_config = model_config
        self.full_config = full_config
        self.threshold = threshold
        print(f'Inference threshold: {self.threshold}')
        self.patch_size = patch_size
        self.stride = stride
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Background masking setup
        self.use_bg_masking = full_config.get('training', {}).get('use_bg_masking', False)
        self.bg_mask_cfg = full_config.get('background_masking', {})
        self.bg_masks = {}
        if self.use_bg_masking:
            self._initialize_background_masks()
        
        # Standard transforms (same as original)
        temp_config = {
            'training': {'use_hist_norm': model_config.get('use_hist_norm', False)},
            'dataset': {'input_size': patch_size}
        }
        self.transform = create_validation_transforms(temp_config)
        self.class_names = model_config.get('class_names', ['NG', 'OK'])
        
        print(f"Background masking: {'Enabled' if self.use_bg_masking else 'Disabled'}")
        print(f"Class names: {self.class_names}")
    
    def _initialize_background_masks(self):
        """Initialize masking objects based on config."""
        try:
            if self.bg_mask_cfg.get('cg_mask_json'):
                self.bg_masks['cg'] = BackgroundMasking(
                    json_path=self.bg_mask_cfg['cg_mask_json'], always_apply=True
                )
            if self.bg_mask_cfg.get('real_mask_json'):
                self.bg_masks['real'] = BackgroundMasking(
                    json_path=self.bg_mask_cfg['real_mask_json'], always_apply=True
                )
            print(f"Background masks loaded: {list(self.bg_masks.keys())}")
        except Exception as e:
            print(f"Failed to load background masks: {e}. Masking disabled.")
            self.use_bg_masking = False

    def _apply_background_masking(self, image: np.ndarray, filename: str) -> np.ndarray:
        """Apply masking based on file prefix."""
        if not self.use_bg_masking:
            return image
        try:
            if filename.startswith('cg_') and 'cg' in self.bg_masks:
                return self.bg_masks['cg'](image=image)['image']
            elif filename.startswith('real_') and 'real' in self.bg_masks:
                return self.bg_masks['real'](image=image)['image']
        except Exception as e:
            print(f"Masking error for {filename}: {e}")
        return image

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
                # Filter out empty/black patches (same threshold as original)
                if np.count_nonzero(patch) < (self.patch_size * self.patch_size * 0.1):
                    continue
                patches.append({'img': patch, 'x': x, 'y': y})
        return patches

    def predict_full_image(self, image_path: str, save_preview: bool = True) -> Dict:
        """Main prediction method - matches original logic."""
        filename = os.path.basename(image_path)
        original_img_bgr = cv2.imread(image_path)
        if original_img_bgr is None: 
            raise ValueError(f"Could not load {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply background masking if enabled
        if self.use_bg_masking:
            masked_rgb = self._apply_background_masking(img_rgb.copy(), filename)
            
            # Save preview for debugging
            if save_preview:
                self._save_mask_preview(img_rgb, masked_rgb, filename)
            
            # Use masked image for patching
            patches = self.get_patches(masked_rgb)
            print(f"Using masked image: {len(patches)} patches found")
        else:
            patches = self.get_patches(img_rgb)
            print(f"Using original image: {len(patches)} patches found")
        
        ng_patches = []
        max_ng_confidence = 0.0
        start_time = time.time()
        
        # Batch process patches (same as original)
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
                ng_conf = prob[0].item()  # Probability of 'NG'
                
                if ng_conf >= self.threshold:
                    patch_info = batch_data[j]
                    patch_info['confidence'] = ng_conf
                    ng_patches.append(patch_info)
                    max_ng_confidence = max(max_ng_confidence, ng_conf)

        inference_time = time.time() - start_time
        overall_label = "NG" if len(ng_patches) > 0 else "OK"
        
        # Create visualization
        vis_path = self._create_visual_result(original_img_bgr, ng_patches, overall_label, image_path)

        return {
            'filename': filename,
            'overall_prediction': overall_label,
            'num_ng_patches': len(ng_patches),
            'max_confidence': max_ng_confidence if overall_label == "NG" else (1 - max_ng_confidence),
            'inference_time': inference_time,
            'annotated_image': vis_path,
            'background_masking': self.use_bg_masking
        }

    def _save_mask_preview(self, original_rgb: np.ndarray, masked_rgb: np.ndarray, filename: str):
        """Save side-by-side comparison for debugging."""
        prev_dir = Path("inference_previews")
        prev_dir.mkdir(exist_ok=True)
        
        # Convert back to BGR for OpenCV
        orig_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
        
        comparison = np.hstack((orig_bgr, mask_bgr))
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(comparison, "Masked", (original_rgb.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imwrite(str(prev_dir / f"mask_check_{filename}"), comparison)

    def _create_visual_result(self, img, ng_patches, label, original_path):
        """Create visualization - same as original."""
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

        # Save
        annotated_dir = "annotated_images"
        os.makedirs(annotated_dir, exist_ok=True)
        fname = f"annotated_{os.path.basename(original_path)}"
        save_path = os.path.join(annotated_dir, fname)
        cv2.imwrite(save_path, vis_img)
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Patch-based inference with optional background masking')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Path to image or directory')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--threshold', type=float, help='NG detection threshold')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--save_preview', action='store_true', help='Save masking comparison previews')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize inference engine
    engine = InferenceEngine(
        model_path=args.model,
        model_config=config,
        full_config=config,
        device=device,
        threshold=args.threshold or config['inference']['threshold']
    )
    
    # Process images
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        result = engine.predict_full_image(str(input_path), save_preview=args.save_preview)
        print(f"\n=== RESULT ===")
        print(f"File: {result['filename']}")
        print(f"Prediction: {result['overall_prediction']}")
        print(f"NG Patches: {result['num_ng_patches']}")
        print(f"Max Confidence: {result['max_confidence']:.3f}")
        print(f"Background Masking: {result['background_masking']}")
        print(f"Time: {result['inference_time']:.3f}s")
        
    elif input_path.is_dir():
        # Directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(input_path.glob(ext))
        
        results = []
        for img_file in tqdm(image_files, desc="Processing images"):
            result = engine.predict_full_image(str(img_file), save_preview=args.save_preview)
            results.append(result)
            print(f"File: {result['filename']} -> {result['overall_prediction']} ({result['num_ng_patches']} defects)")
        
        # Summary
        ok_count = sum(1 for r in results if r['overall_prediction'] == 'OK')
        ng_count = sum(1 for r in results if r['overall_prediction'] == 'NG')
        
        print(f"\n=== SUMMARY ===")
        print(f"Total images: {len(results)}")
        print(f"OK predictions: {ok_count} ({ok_count/len(results)*100:.1f}%)")
        print(f"NG predictions: {ng_count} ({ng_count/len(results)*100:.1f}%)")
        print(f"Background masking: {engine.use_bg_masking}")
        print(f"Average NG patches per image: {np.mean([r['num_ng_patches'] for r in results]):.1f}")


if __name__ == "__main__":
    main()
