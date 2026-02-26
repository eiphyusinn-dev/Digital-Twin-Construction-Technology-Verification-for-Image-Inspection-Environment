import os
import json
import cv2
import numpy as np
import glob
from pathlib import Path
import random
import sys
from typing import List, Optional

# Add parent directory to path to find utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing modules
from utils.preprocessing import HistogramNormalization, FDATransform


class BackgroundMasking:
    def __init__(self, json_path: str = None, original_size: tuple = None):
        self.json_path = Path(json_path)
        self.original_size = original_size
        self.mask_data = self._load_json()

    def _load_json(self):
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return None

    def get_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.mask_data is None: 
            mask.fill(255)
            return mask
        
        orig_w, orig_h = self.original_size if self.original_size else (w, h)
        scale_x, scale_y = w / orig_w, h / orig_h
        
        for shape in self.mask_data.get('shapes', []):
            if shape['label'] == 'target_object':
                pts = np.array(shape['points'], dtype=np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
        return mask


class CompletePatchProcessor:
    """
    Complete patch processing pipeline:
    1. Histogram normalization (whole image)
    2. Background masking
    3. Split patches
    4. FDA on each patch (using real image samples)
    5. Save processed patches
    """
    
    def __init__(self, 
                 patch_size: int = 224,
                 stride: int = 112,
                 fda_reference_dir: str = None,
                 fda_beta: float = 0.1,
                 use_fda: bool = True):
        
        self.patch_size = patch_size
        self.stride = stride
        self.use_fda = use_fda
        
        # Initialize histogram normalization
        self.hist_norm = HistogramNormalization(always_apply=True)
        
        # Initialize FDA if enabled
        self.fda_transform = None
        if self.use_fda and fda_reference_dir:
            self.fda_transform = FDATransform(
                reference_images_path=fda_reference_dir,  # Fixed parameter name
                beta_limit=fda_beta,
                always_apply=True
            )
            print(f"FDA initialized with beta={fda_beta}, reference_path={fda_reference_dir}")
        else:
            print("FDA disabled")
    
    def create_visualization(self, processed_data: dict, output_root: str):
        """Create visualization showing patch grid and labels."""
        base_name = processed_data['base_name']
        img_masked = processed_data['img_masked']
        defect_mask = processed_data['defect_mask']
        patches_info = processed_data['patches_info']
        
        # Create visualization
        vis_img = img_masked.copy()
        overlay = vis_img.copy()
        ALPHA = 0.25
        
        for patch_info in patches_info:
            x, y = patch_info['coords']
            label = patch_info['label']
            r, c = patch_info['grid_pos']
            
            if label == "NG":
                # Red semi-transparent fill
                cv2.rectangle(overlay, (x, y), (x + self.patch_size, y + self.patch_size), (0, 0, 255), -1)
                # Yellow border
                cv2.rectangle(vis_img, (x, y), (x + self.patch_size, y + self.patch_size), (0, 255, 255), 1)
                # Label
                cv2.putText(vis_img, f"NG_{r}_{c}", (x + 5, y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.addWeighted(overlay, ALPHA, vis_img, 1 - ALPHA, 0, vis_img)
        
        # Highlight defect areas
        if defect_mask is not None:
            vis_img[defect_mask > 0] = [255, 255, 255]
        
        # Save visualization
        vis_dir = os.path.join(output_root, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        vis_save_path = os.path.join(vis_dir, f"{base_name}_complete_pipeline.jpg")
        cv2.imwrite(vis_save_path, vis_img)
        
        return vis_save_path
    
    def process_single_image(self, 
                           img_path: str, 
                           seg_path: str, 
                           mask_json_path: str) -> dict:
        """Process a single image through the complete pipeline."""
        
        base_name = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            return {"error": f"Could not load image: {img_path}"}
        
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Apply Histogram Normalization to whole image
        print(f"Applying histogram normalization to {base_name}")
        try:
            hist_normalized = self.hist_norm(image=img_rgb)['image']
        except Exception as e:
            print(f"Histogram normalization failed: {e}")
            hist_normalized = img_rgb
                    
        # Step 2: Apply Background Masking
        masker = BackgroundMasking(json_path=mask_json_path, original_size=(w, h))
        full_mask = masker.get_mask(h, w)
        img_masked = cv2.bitwise_and(hist_normalized, hist_normalized, mask=full_mask)
        
        # Step 3: Load and isolate Defect Mask
        idx = base_name.split('_')[-1]
        seg_full_path = os.path.join(seg_path, f"instance_segmentation_{idx}.png")
        
        defect_mask = None
        if os.path.exists(seg_full_path):
            seg_map = cv2.imread(seg_full_path)
            DEFECT_BGR = np.array([61, 61, 204])  # Your defect color in BGR
            defect_mask = cv2.inRange(seg_map, DEFECT_BGR, DEFECT_BGR)
            defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=full_mask)
        else:
            print(f"Warning: Seg map not found at {seg_full_path}")
        
        # Step 4: Split into patches and apply FDA
        patches_info = []
        
        # Calculate patch coordinates
        y_coords = list(range(0, h - self.patch_size + 1, self.stride))
        if y_coords[-1] != h - self.patch_size: 
            y_coords.append(h - self.patch_size)
        x_coords = list(range(0, w - self.patch_size + 1, self.stride))
        if x_coords[-1] != w - self.patch_size: 
            x_coords.append(w - self.patch_size)
        
        count_ng = 0
        
        for r, y in enumerate(y_coords):
            for c, x in enumerate(x_coords):
                # Extract patch
                patch_data = img_masked[y:y+self.patch_size, x:x+self.patch_size]
                
                # Skip nearly empty patches
                if np.count_nonzero(patch_data) < 10: 
                    continue
                
                # Step 5: Apply FDA to patch if enabled
                processed_patch = patch_data.copy()
                if self.use_fda and self.fda_transform:
                    try:
                        # Apply FDA using the single reference image from FDATransform
                        fda_result = self.fda_transform(image=patch_data)
                        processed_patch = fda_result['image']
                        
                    except Exception as e:
                        print(f"FDA failed for patch {r}_{c}: {e}")
                        # Use original patch if FDA fails
                        processed_patch = patch_data
                
                # Determine label based on defect mask
                label = "OK"
                if defect_mask is not None:
                    patch_defect_pixels = defect_mask[y:y+self.patch_size, x:x+self.patch_size]
                    num_defect_pixels = np.count_nonzero(patch_defect_pixels)
                    label = "NG" if num_defect_pixels >= 10 else "OK"
                
                if label == "NG":
                    count_ng += 1
                
                # Save patch info
                patches_info.append({
                    'patch': processed_patch,
                    'coords': (x, y),
                    'grid_pos': (r, c),
                    'label': label,
                    'filename': f"{base_name}_r{r}_c{c}_{label}.png"
                })
        
        return {
            'base_name': base_name,
            'img_masked': img_masked,
            'defect_mask': defect_mask,
            'patches_info': patches_info,
            'count_ng': count_ng,
            'patch_coords': list(zip(x_coords, y_coords))
        }
    
    def save_patches(self, processed_data: dict, output_root: str):
        """Save all patches to appropriate folders."""
        base_name = processed_data['base_name']
        patches_info = processed_data['patches_info']
        
        # Create output directories
        for label in ["OK", "NG"]:
            os.makedirs(os.path.join(output_root, label), exist_ok=True)
        
        # Save each patch
        for patch_info in patches_info:
            patch = patch_info['patch']
            filename = patch_info['filename']
            label = patch_info['label']
            
            # Convert RGB to BGR for OpenCV saving
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(output_root, label, filename)
            cv2.imwrite(save_path, patch_bgr)
        
        print(f"Saved {len(patches_info)} patches for {base_name}")
        return len(patches_info)


def run_complete_patch_pipeline(img_dir: str, 
                              seg_dir: str, 
                              mask_json_path: str,
                              output_root: str,
                              real_reference_dir: str = None,
                              patch_size: int = 224,
                              stride: int = 112,
                              fda_beta: float = 0.1,
                              use_fda: bool = True):
    """
    Run complete patch processing pipeline:
    1. Histogram normalization
    2. Background masking  
    3. Patch splitting
    4. FDA on patches
    5. Save patches
    """
    
    print("=" * 60)
    print("COMPLETE PATCH PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input directory: {img_dir}")
    print(f"Output directory: {output_root}")
    print(f"Patch size: {patch_size}, Stride: {stride}")
    print(f"Histogram normalization: Enabled")
    print(f"Background masking: Enabled")
    print(f"FDA: {'Enabled' if use_fda else 'Disabled'}")
    if use_fda:
        print(f"FDA beta: {fda_beta}")
        print(f"Real reference directory: {real_reference_dir}")
    print("=" * 60)
    
    # Initialize processor
    processor = CompletePatchProcessor(
        patch_size=patch_size,
        stride=stride,
        fda_reference_dir=real_reference_dir,
        fda_beta=fda_beta,
        use_fda=use_fda
    )
    
    # Get all image files
    img_files = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"Found {len(img_files)} images to process")
    
    # Process each image
    total_patches = 0
    total_ng_patches = 0
    
    for i, img_path in enumerate(img_files, 1):
        print(f"\n[{i}/{len(img_files)}] Processing: {Path(img_path).name}")
        
        # Process image
        processed_data = processor.process_single_image(
            img_path=img_path,
            seg_path=seg_dir,
            mask_json_path=mask_json_path
        )
        
        if 'error' in processed_data:
            print(f"Error: {processed_data['error']}")
            continue
        
        # Save patches
        patch_count = processor.save_patches(processed_data, output_root)
        total_patches += patch_count
        total_ng_patches += processed_data['count_ng']
        
        # Create visualization
        vis_path = processor.create_visualization(processed_data, output_root)
        
        print(f"  -> {patch_count} patches saved ({processed_data['count_ng']} NG)")
        print(f"  -> Visualization: {vis_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {len(img_files)}")
    print(f"Total patches created: {total_patches}")
    print(f"Total NG patches: {total_ng_patches}")
    print(f"Total OK patches: {total_patches - total_ng_patches}")
    print(f"NG ratio: {total_ng_patches/total_patches*100:.1f}%")
    print(f"Output directory: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    # Configuration - update these paths as needed
    run_complete_patch_pipeline(
        img_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/rgb",
        seg_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/instance_segmentation", 
        mask_json_path="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/mask/cg_mask_label.json",
        output_root="patches_dataset/patches_dataset_bgmask_histnorm_fda",
        real_reference_dir="real_image_cropped/5_7000_25.png", 
        patch_size=224,
        stride=112,
        fda_beta=0.001,
        use_fda=True 
    )
