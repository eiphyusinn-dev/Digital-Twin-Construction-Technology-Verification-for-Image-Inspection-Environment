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
import csv
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
                 stride: int = 112,
                 save_patch_probs: bool = False):
        self.model_config = model_config
        self.full_config = full_config
        self.threshold = threshold
        self.save_patch_probs = save_patch_probs
        print(f'Inference threshold: {self.threshold}')
        self.patch_size = patch_size
        self.stride = stride
        
        # --- CoordConv Setup ---
        self.coordconv_cfg = full_config.get('coordconv', {})
        self.use_coordconv = self.coordconv_cfg.get('enabled', False)
        self.grid_rows = self.coordconv_cfg.get('grid_rows', 1)
        self.grid_cols = self.coordconv_cfg.get('grid_cols', 1)
        print(f"CoordConv: {'Enabled' if self.use_coordconv else 'Disabled'}")
        # -----------------------

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
            # if filename.startswith('cg_') and 'cg' in self.bg_masks:
            if 'cg' in self.bg_masks:
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
            in_chans=self.model_config.get('in_chans', 3),
            use_coordconv=self.use_coordconv  # Pass CoordConv flag to model
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model
    
    def get_patches(self, img: np.ndarray) -> List[Dict]:
            """
            Splits image into patches and keeps track of raw pixel coordinates
            and grid indices for CoordConv normalization.
            """
            h, w = img.shape[:2]
            patches = []

            # Calculate coordinate grids
            y_coords = list(range(0, h - self.patch_size + 1, self.stride))
            if not y_coords or y_coords[-1] != h - self.patch_size:
                y_coords.append(h - self.patch_size)

            x_coords = list(range(0, w - self.patch_size + 1, self.stride))
            if not x_coords or x_coords[-1] != w - self.patch_size:
                x_coords.append(w - self.patch_size)

            # Store grid dimensions for CoordConv normalization
            self._grid_rows = len(y_coords)
            self._grid_cols = len(x_coords)

            for row_idx, y in enumerate(y_coords):
                for col_idx, x in enumerate(x_coords):
                    patch = img[y:y+self.patch_size, x:x+self.patch_size]

                    # Filter out empty/black patches
                    if np.count_nonzero(patch) < (self.patch_size * self.patch_size * 0.1):
                        continue

                    patches.append({
                        'img': patch,
                        'x': x,
                        'y': y,
                        'row_idx': row_idx,
                        'col_idx': col_idx
                    })
            return patches

    def predict_full_image(self, image_path: str, annotated_dir: str, save_preview: bool = True) -> Dict:
        """
        Main prediction method updated for Pixel-Based Coordinate Normalization.
        """
        filename = os.path.basename(image_path)
        original_img_bgr = cv2.imread(image_path)
        if original_img_bgr is None: 
            raise ValueError(f"Could not load {image_path}")
        
        img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # 1. Apply background masking if enabled
        if self.use_bg_masking:
            masked_rgb = self._apply_background_masking(img_rgb.copy(), filename)
            if save_preview:
                self._save_mask_preview(img_rgb, masked_rgb, filename)
            patches = self.get_patches(masked_rgb)
        else:
            patches = self.get_patches(img_rgb)
            
        print(f'Total Patches to process: {len(patches)}')
        
        # 2. Pre-compute normalization denominators (Grid-index-based, matching training)
        max_row_idx = max(self._grid_rows - 1, 1)
        max_col_idx = max(self._grid_cols - 1, 1)
        
        ng_patches = []
        all_patches_data = []  # Store all patch data for CSV export
        max_ng_confidence = 0.0
        start_time = time.time()
        
        # 3. Batch process patches
        batch_size = 16
        for i in range(0, len(patches), batch_size):
            batch_data = patches[i:i+batch_size]
            batch_tensors = []
            batch_coords = []
            
            for p in batch_data:
                # Transform image patch
                transformed = self.transform(image=p['img'])['image']
                batch_tensors.append(transformed)
                
                # Normalize coordinates using grid indices (same as training)
                if self.use_coordconv:
                    row_norm = p['row_idx'] / max_row_idx
                    col_norm = p['col_idx'] / max_col_idx
                    batch_coords.append([row_norm, col_norm])
            
            input_batch = torch.stack(batch_tensors).to(self.device)
            
            # 4. Prepare coordinate tensor
            coords = None
            if self.use_coordconv and batch_coords:
                coords = torch.tensor(batch_coords, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Forward pass with optional coords
                outputs = self.model(input_batch, coords=coords)
                probs = F.softmax(outputs, dim=1)
                
            # 5. Process results
            for j, prob in enumerate(probs):
                ng_conf = prob[0].item()  # NG is class 0
                
                # Store all patch data
                patch_data = {
                    'filename': filename,
                    'row_idx': batch_data[j]['row_idx'],
                    'col_idx': batch_data[j]['col_idx'],
                    'x': batch_data[j]['x'],
                    'y': batch_data[j]['y'],
                    'ng_probability': ng_conf
                }
                all_patches_data.append(patch_data)
                
                if ng_conf >= self.threshold:
                    patch_info = batch_data[j].copy()
                    patch_info['confidence'] = ng_conf
                    ng_patches.append(patch_info)
                    max_ng_confidence = max(max_ng_confidence, ng_conf)

        inference_time = time.time() - start_time
        overall_label = "NG" if len(ng_patches) > 0 else "OK"
        
        # 6. Create visualization
        vis_path = self._create_visual_result(original_img_bgr, ng_patches, overall_label, image_path, annotated_dir)
        
        # 7. Save patch probabilities to CSV if enabled
        csv_path = None
        if self.save_patch_probs:
            csv_path = self._save_patch_probabilities(all_patches_data, annotated_dir)

        return {
            'filename': filename,
            'overall_prediction': overall_label,
            'num_ng_patches': len(ng_patches),
            'max_confidence': max_ng_confidence if overall_label == "NG" else (1 - max_ng_confidence),
            'inference_time': inference_time,
            'annotated_image': vis_path,
            'patch_probabilities_csv': csv_path,
            'background_masking': self.use_bg_masking,
            'coordconv_enabled': self.use_coordconv
        }
    
    def _save_mask_preview(self, original_rgb: np.ndarray, masked_rgb: np.ndarray, filename: str):
        """Save side-by-side comparison for debugging."""
        prev_dir = Path("inference_previews")
        prev_dir.mkdir(exist_ok=True)
        
        orig_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
        
        comparison = np.hstack((orig_bgr, mask_bgr))
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(comparison, "Masked", (original_rgb.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imwrite(str(prev_dir / f"mask_check_{filename}"), comparison)

    def _create_visual_result(self, img, ng_patches, label, original_path, annotated_dir):
        """Create visualization."""
        vis_img = img.copy()
        overlay = img.copy()
        
        for patch in ng_patches:
            x, y = patch['x'], patch['y']
            cv2.rectangle(overlay, (x, y), (x + self.patch_size, y + self.patch_size), (0, 0, 255), -1)
            cv2.rectangle(vis_img, (x, y), (x + self.patch_size, y + self.patch_size), (0, 255, 255), 2)

        cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
        
        color = (0, 0, 255) if label == "NG" else (0, 255, 0)
        cv2.rectangle(vis_img, (0, 0), (img.shape[1], 50), color, -1)
        cv2.putText(vis_img, f"RESULT: {label} | NG Patches: {len(ng_patches)}", 
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        os.makedirs(annotated_dir, exist_ok=True)
        fname = f"annotated_{os.path.basename(original_path)}"
        save_path = os.path.join(annotated_dir, fname)
        cv2.imwrite(save_path, vis_img)
        return save_path
    
    def _save_patch_probabilities(self, all_patches_data: List[Dict], annotated_dir: str) -> str:
        """Save all patch probabilities to CSV file for evaluation."""
        os.makedirs(f'{annotated_dir}/patch_prob_csv', exist_ok=True)
        
        # Generate CSV filename based on the first patch's filename
        base_filename = all_patches_data[0]['filename'] if all_patches_data else 'unknown'
        csv_filename = f"patch_probs_{Path(base_filename).stem}.csv"
        csv_path = os.path.join(annotated_dir,'patch_prob_csv', csv_filename)
        
        # Write CSV with headers
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'row_idx', 'col_idx', 'x', 'y', 'ng_probability']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_patches_data)
        
        print(f"Saved patch probabilities to {csv_path}")
        return csv_path


def main():
    parser = argparse.ArgumentParser(description='Patch-based inference with CoordConv and Background Masking')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Path to image or directory')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--threshold', type=float, help='NG detection threshold')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--save_preview', action='store_true', help='Save masking comparison previews')
    parser.add_argument('--save_patch_probs', action='store_true', help='Save all patch probabilities to CSV')
    parser.add_argument('--annotated_dir', default='annotated_dir')
    
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    engine = InferenceEngine(
        model_path=args.model,
        model_config=config,
        full_config=config,
        device=device,
        threshold=args.threshold or config['inference']['threshold'],
        save_patch_probs=args.save_patch_probs
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = engine.predict_full_image(str(input_path), save_preview=args.save_preview, annotated_dir=args.annotated_dir)
        print(f"\n=== RESULT ===")
        print(f"File: {result['filename']}")
        print(f"Prediction: {result['overall_prediction']}")
        print(f"NG Patches: {result['num_ng_patches']}")
        print(f"Max Confidence: {result['max_confidence']:.3f}")
        print(f"CoordConv: {result['coordconv_enabled']}")
        print(f"Time: {result['inference_time']:.3f}s")
        
    elif input_path.is_dir():
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(input_path.glob(ext))
        
        results = []
        for img_file in tqdm(image_files, desc="Processing images"):
            result = engine.predict_full_image(str(img_file), save_preview=args.save_preview, annotated_dir=args.annotated_dir)
            results.append(result)
            print(f"File: {result['filename']} -> {result['overall_prediction']} ({result['num_ng_patches']} defects)")
        
        ok_count = sum(1 for r in results if r['overall_prediction'] == 'OK')
        ng_count = sum(1 for r in results if r['overall_prediction'] == 'NG')
        
        print(f"\n=== SUMMARY ===")
        print(f"Total images: {len(results)}")
        print(f"OK predictions: {ok_count} ({ok_count/len(results)*100:.1f}%)")
        print(f"NG predictions: {ng_count} ({ng_count/len(results)*100:.1f}%)")
        print(f"CoordConv Enabled: {engine.use_coordconv}")
        print(f"Average NG patches per image: {np.mean([r['num_ng_patches'] for r in results]):.1f}")

if __name__ == "__main__":
    main()