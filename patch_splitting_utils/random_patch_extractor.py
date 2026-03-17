#!/usr/bin/env python3
"""
Random Patch Extraction for CG Images

This script implements random patch extraction for CG training data to prevent overfitting
from identical textures in Omniverse images. It generates OK and NG patches from separate
source images with configurable preprocessing and ratio control.
"""

import os
import json
import cv2
import numpy as np
import glob
from pathlib import Path
import sys
import random
from typing import List, Tuple, Dict, Optional
import yaml
import time

# Add parent directory to path to find utils module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(script_dir)
sys.path.append(parent_dir)

# Import preprocessing modules
try:
    from utils.fda_utils import FDA_source_to_target_np
    from utils.preprocessing import HistogramNormalization, FDATransform
except ImportError as e:
    print(f"Error importing utils: {e}")
    print(f"Script directory: {script_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Available paths: {sys.path}")
    sys.exit(1)


class BackgroundMasking:
    """Background masking utility for random patch extraction."""
    
    def __init__(self, json_path: str = None):
        self.json_path = Path(json_path) if json_path else None
        self.mask_data = self._load_json()
    
    def _load_json(self):
        if self.json_path and self.json_path.exists():
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_mask(self, h: int, w: int) -> np.ndarray:
        """Get background mask for specified dimensions."""
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.mask_data is None: 
            mask.fill(255)
            return mask
        
        # Get original size from JSON if available
        if 'imageWidth' in self.mask_data and 'imageHeight' in self.mask_data:
            orig_w, orig_h = self.mask_data['imageWidth'], self.mask_data['imageHeight']
            scale_x, scale_y = w / orig_w, h / orig_h
        else:
            scale_x, scale_y = 1.0, 1.0
        
        for shape in self.mask_data.get('shapes', []):
            if shape.get('label') == 'target_object':
                pts = np.array(shape['points'], dtype=np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
        return mask


class DefectColorConfig:
    """Manages defect color configurations for CG and Real images."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cg_defect_colors = self._load_defect_colors('cg')
        self.real_defect_colors = self._load_defect_colors('real')
        print(f"Loaded CG defect colors: {self.cg_defect_colors}")
        print(f"Loaded Real defect colors: {self.real_defect_colors}")
    
    def _load_defect_colors(self, image_type: str) -> List[np.ndarray]:
        """Load defect colors from config for specified image type."""
        colors = []
        defect_colors_config = self.config.get('defect_colors', {})
        image_config = defect_colors_config.get(image_type, {})
        color_configs = image_config.get('colors', [])
        
        for color_config in color_configs:
            if color_config.get('enabled', True):
                bgr = color_config.get('bgr', [61, 61, 204])
                colors.append(np.array(bgr, dtype=np.uint8))
        
        return colors
    
    def get_defect_mask(self, seg_image: np.ndarray, image_type: str = 'cg') -> np.ndarray:
        """Generate defect mask for semantic segmentation image."""
        if seg_image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        
        defect_mask = np.zeros(seg_image.shape[:2], dtype=np.uint8)
        
        colors = self.cg_defect_colors if image_type == 'cg' else self.real_defect_colors
        
        for color in colors:
            # Fast exact color matching (no tolerance needed for exact matches)
            mask = cv2.inRange(seg_image, color, color)
            
            if np.any(mask):
                print(f"Found {np.count_nonzero(mask)} pixels for color {color}")
            defect_mask = cv2.bitwise_or(defect_mask, mask)
        
        return defect_mask

class RandomPatchExtractor:
    """Main class for random patch extraction from CG images."""
    
    def __init__(self, config: dict):
        self.config = config
        self.patch_size = config['patch_size']
        
        # Load full config for defect colors
        full_config = self._load_full_config()
        self.defect_color_config = DefectColorConfig(full_config)
    
        # Initialize preprocessing components
        self._init_preprocessing()
        
        # Set random seed for reproducibility
        self.random_seed = config.get('random_seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Test mode - limit number of images for testing
        self.test_mode = config.get('test_mode', False)
        self.test_image_limit = config.get('test_image_limit', 10)
        
        print(f"RandomPatchExtractor initialized with seed={self.random_seed}")
        print(f"Test mode: {'ON' if self.test_mode else 'OFF'} (limit: {self.test_image_limit} images)")
    
    def _load_full_config(self) -> dict:
        """Load full configuration from config.yaml."""
        config_path = 'config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _init_preprocessing(self):
        """Initialize preprocessing components based on config."""
        preprocessing_config = self.config.get('preprocessing', {})
        
        print("\nInitializing preprocessing components...")
        
        # Background masking
        mask_json_path = self.config.get('mask_json_path')
        self.bg_masking = BackgroundMasking(json_path=mask_json_path) if mask_json_path else None
        print(f"Background masking: {'Enabled' if self.bg_masking else 'Disabled'}")
        
        # Histogram normalization
        hist_norm_config = preprocessing_config.get('hist_norm', {})
        self.hist_norm = None
        self.hist_norm_mode = hist_norm_config.get('mode', 'crop_first')
        
        if hist_norm_config.get('enabled', False):
            try:
                self.hist_norm = HistogramNormalization(always_apply=True)
                print(f"Histogram normalization: Enabled ({self.hist_norm_mode})")
            except Exception as e:
                print(f"Warning: Could not initialize histogram normalization: {e}")
                self.hist_norm = None
        else:
            print("Histogram normalization: Disabled")
        
        # FDA
        fda_config = preprocessing_config.get('fda', {})
        self.fda_enabled = False
        self.fda_mode = fda_config.get('mode', 'crop_first')
        self.fda_beta = fda_config.get('beta_limit', 0.001)
        self.fda_reference_fullsize = None

        if fda_config.get('enabled', False):
            reference_path = fda_config.get('reference_path')
            if reference_path and os.path.exists(reference_path):
                try:
                    ref_img = cv2.imread(reference_path)
                    self.fda_reference_fullsize = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    self.fda_enabled = True
                    print(f"FDA: Enabled ({self.fda_mode}) with reference: {reference_path} "
                          f"({ref_img.shape[1]}x{ref_img.shape[0]})")
                except Exception as e:
                    print(f"Warning: Could not load FDA reference: {e}")
            else:
                print(f"FDA enabled but reference path not found: {reference_path}, disabling FDA")
        else:
            print("FDA: Disabled")
    
    def _apply_fda(self, src_image: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
        """Apply FDA: transfer low-frequency style from ref_image to src_image.

        Both images must be RGB uint8 with the same dimensions.
        """
        src = src_image.astype(np.float32).transpose(2, 0, 1) / 255.0
        trg = ref_image.astype(np.float32).transpose(2, 0, 1) / 255.0
        result = FDA_source_to_target_np(src, trg, L=self.fda_beta)
        return np.clip(result.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

    def _apply_preprocessing_crop_first(self, image: np.ndarray, image_id: str = "debug") -> Tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing in crop_first mode: crop to product -> normalize -> restore to black background."""
        processed_image = image.copy()
        h, w = image.shape[:2]
        
        # Get background mask first
        bg_mask = None
        if self.bg_masking:
            bg_mask = self.bg_masking.get_mask(h, w)
        
        # STEP 1: Crop to product area (remove black background)
        if bg_mask is not None:
            # Find bounding box of product area
            coords = cv2.findNonZero(bg_mask)
            if coords is not None:
                x, y, w_crop, h_crop = cv2.boundingRect(coords)
                
                # Crop to product area only
                cropped_image = processed_image[y:y+h_crop, x:x+w_crop]
                cropped_mask = bg_mask[y:y+h_crop, x:x+w_crop]
                
                print(f"    Cropped to product area: {w_crop}x{h_crop} (from {w}x{h})")
            else:
                # Fallback: use whole image if no product found
                cropped_image = processed_image
                cropped_mask = bg_mask
                print("    Warning: No product area found, using whole image")
                x, y, w_crop, h_crop = 0, 0, w, h
        else:
            # No background mask, use whole image
            cropped_image = processed_image
            cropped_mask = None
            print("    No background mask, using whole image")
            x, y, w_crop, h_crop = 0, 0, w, h
        
        # STEP 2: Apply FDA with coordinate-matched Real reference
        if self.fda_enabled and self.fda_mode == 'crop_first' and self.fda_reference_fullsize is not None:
            try:
                # Apply same mask + crop to Real reference image
                ref_h, ref_w = self.fda_reference_fullsize.shape[:2]
                if self.bg_masking:
                    ref_mask = self.bg_masking.get_mask(ref_h, ref_w)
                    ref_masked = cv2.bitwise_and(
                        self.fda_reference_fullsize, self.fda_reference_fullsize, mask=ref_mask)
                    ref_cropped = ref_masked[y:y+h_crop, x:x+w_crop]
                else:
                    ref_cropped = self.fda_reference_fullsize[y:y+h_crop, x:x+w_crop]

                cropped_image = self._apply_fda(cropped_image, ref_cropped)
                print("    Applied FDA (crop-matched) to cropped product area")
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        # STEP 3: Apply histogram normalization to cropped product area only
        if self.hist_norm and self.hist_norm_mode == 'crop_first':
            try:
                cropped_image = self.hist_norm(image=cropped_image)['image']
                print("    Applied histogram normalization to cropped product area")
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")
        
        # STEP 4: Restore normalized product area back to original black background
        if bg_mask is not None:
            # Create final image with black background
            final_image = np.zeros_like(processed_image)
            
            # Put processed product area back in original position
            final_image[y:y+h_crop, x:x+w_crop] = cropped_image
            
            # Apply background mask to ensure clean edges
            final_mask = np.zeros_like(bg_mask)
            final_mask[y:y+h_crop, x:x+w_crop] = cropped_mask if cropped_mask is not None else bg_mask[y:y+h_crop, x:x+w_crop]
            final_image = cv2.bitwise_and(final_image, final_image, mask=final_mask)
            
            print("    Restored processed product area to black background")
            return final_image, bg_mask
        else:
            # No cropping was done, return processed image as-is
            return processed_image, bg_mask
    
    def _apply_preprocessing_patch_first(self, patch: np.ndarray, x: int = 0, y: int = 0) -> np.ndarray:
        """Apply preprocessing in patch_first mode: preprocess individual patch.

        Args:
            patch: Source CG patch (RGB, uint8)
            x, y: Patch coordinates in the original full-size image (for FDA coordinate matching)
        """
        processed_patch = patch.copy()

        # Apply FDA with coordinate-matched Real reference patch
        if self.fda_enabled and self.fda_mode == 'patch_first' and self.fda_reference_fullsize is not None:
            try:
                # Extract Real patch at same coordinates
                ref_patch = self.fda_reference_fullsize[y:y+self.patch_size, x:x+self.patch_size]
                processed_patch = self._apply_fda(processed_patch, ref_patch)
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        # Apply histogram normalization
        if self.hist_norm and self.hist_norm_mode == 'patch_first':
            try:
                processed_patch = self.hist_norm(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")

        return processed_patch
    
    def _extract_patch(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """Extract patch from image at given coordinates."""
        return image[y:y+self.patch_size, x:x+self.patch_size]
    
    def _calculate_work_area_ratio(self, mask: np.ndarray, x: int, y: int) -> float:
        """Calculate ratio of non-black pixels in patch region."""
        if mask is None:
            return 1.0
        
        patch_mask = mask[y:y+self.patch_size, x:x+self.patch_size]
        non_zero_pixels = np.count_nonzero(patch_mask)
        total_pixels = self.patch_size * self.patch_size
        return non_zero_pixels / total_pixels
    
    def generate_ok_patches(self, target_count: int) -> List[Tuple[np.ndarray, str]]:
        """Generate OK patches from defect-free CG image."""
        ok_source_path = self.config.get('ok_source_image')
        if not ok_source_path or not os.path.exists(ok_source_path):
            raise FileNotFoundError(f"OK source image not found: {ok_source_path}")
        
        print(f"\nGenerating {target_count} OK patches from {ok_source_path}")
        
        # Load defect-free image
        image = cv2.imread(ok_source_path)
        if image is None:
            raise ValueError(f"Could not load OK source image: {ok_source_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Apply preprocessing based on mode
        if self.hist_norm_mode == 'crop_first' or self.fda_mode == 'crop_first':
            image_rgb, bg_mask = self._apply_preprocessing_crop_first(image_rgb, "ok_source")
        else:
            bg_mask = self.bg_masking.get_mask(h, w) if self.bg_masking else None
        
        min_work_area_ratio = self.config.get('min_work_area_ratio', 0.9)
        max_attempts = target_count * 20  # Prevent infinite loops
        
        ok_patches = []
        attempts = 0
        valid_coords = []
        
        # Pre-calculate valid coordinates to speed up processing
        if bg_mask is not None:
            for y in range(0, h - self.patch_size + 1, 10):  # Sample every 10 pixels
                for x in range(0, w - self.patch_size + 1, 10):
                    if self._calculate_work_area_ratio(bg_mask, x, y) >= min_work_area_ratio:
                        valid_coords.append((x, y))
        
        while len(ok_patches) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Select coordinates
            if valid_coords and len(valid_coords) > 0:
                # Use pre-calculated valid coordinates if available
                x, y = random.choice(valid_coords)
            else:
                # Random coordinates
                x = random.randint(0, max(0, w - self.patch_size))
                y = random.randint(0, max(0, h - self.patch_size))
                
                # Check work area ratio
                if bg_mask is not None:
                    work_area_ratio = self._calculate_work_area_ratio(bg_mask, x, y)
                    if work_area_ratio < min_work_area_ratio:
                        continue
            
            # Extract patch
            patch = self._extract_patch(image_rgb, x, y)
            
            # Apply preprocessing if patch_first mode
            if self.hist_norm_mode == 'patch_first' or self.fda_mode == 'patch_first':
                patch = self._apply_preprocessing_patch_first(patch, x, y)
            
            # Generate filename
            filename = f"cg_ok_rnd{len(ok_patches)+1:05d}_OK.png"
            ok_patches.append((patch, filename))
            
            if attempts % 100 == 0:
                print(f"    Generated {len(ok_patches)}/{target_count} OK patches...")
            
            if attempts >= max_attempts:
                print(f"    Warning: Reached max attempts ({max_attempts}) without meeting target")
                break
        
        print(f"    Generated {len(ok_patches)} OK patches after {attempts} attempts")
        return ok_patches
    
    def _find_defect_centroid(self, defect_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """Calculate centroid of defect pixels."""
        ys, xs = np.where(defect_mask > 0)
        if len(ys) == 0:
            return None
        
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        return (cx, cy)
    
    def _generate_ng_patches_for_image(self, image_path: str, seg_path: str, image_id: str) -> List[Tuple[np.ndarray, str]]:
        """Generate NG patches for a single defect image."""
        start_time = time.time()
        print(f"  Processing: {image_id}")
        
        # Load image (only once)
        load_start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Warning: Could not load image {image_path}")
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        print(f"    Image loading: {time.time() - load_start:.2f}s")
        
        # Load semantic segmentation (only once)
        seg_start = time.time()
        seg_image = cv2.imread(seg_path)
        if seg_image is None:
            print(f"  Warning: Could not load segmentation {seg_path}")
            return []
        print(f"    Segmentation loading: {time.time() - seg_start:.2f}s")
        
        # Generate defect mask (only once)
        mask_start = time.time()
        defect_mask = self.defect_color_config.get_defect_mask(seg_image, 'cg')
        total_defect_pixels = np.count_nonzero(defect_mask)
        print(f"    Defect mask generation: {time.time() - mask_start:.2f}s")
        
        if total_defect_pixels == 0:
            print(f"  Warning: No defects found in {seg_path}")
            return []
        
        print(f"  Total defect pixels: {total_defect_pixels}")
        # Find defect centroid (only once)
        centroid = self._find_defect_centroid(defect_mask)
        if centroid is None:
            print(f"  Warning: Could not find defect centroid")
            return []
        
        cx, cy = centroid
        print(f"  Defect centroid: ({cx}, {cy})")
        
        # Apply preprocessing based on mode
        bg_mask = None
        if self.hist_norm_mode == 'crop_first' or self.fda_mode == 'crop_first':
            print(f"  Applying crop_first preprocessing...")
            image_rgb, bg_mask = self._apply_preprocessing_crop_first(image_rgb, image_id)
            # Update defect mask to only include pixels within background mask
            if bg_mask is not None:
                defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=bg_mask)
        else:
            if self.bg_masking:
                bg_mask = self.bg_masking.get_mask(h, w)
        
        # Generate patches
        patch_start = time.time()
        patches_per_defect = self.config.get('patches_per_defect', 10)
        min_overlap_ratio = self.config.get('min_defect_overlap', 0.3)
        min_defect_pixels = self.config.get('min_defect_pixels_in_patch', 200)
        
        ng_patches = []
        attempts = 0
        max_attempts_per_patch = 50
        max_total_attempts = patches_per_defect * max_attempts_per_patch
        
        # Generate unique patches by tracking used coordinates
        used_coords = set()
        
        while len(ng_patches) < patches_per_defect and attempts < max_total_attempts:
            attempts += 1
            
            # Random offset around centroid
            dx = random.randint(-self.patch_size//2, self.patch_size//2)
            dy = random.randint(-self.patch_size//2, self.patch_size//2)
            
            # Calculate patch coordinates
            patch_x = cx - self.patch_size//2 + dx
            patch_y = cy - self.patch_size//2 + dy
            
            # Clamp to image boundaries
            patch_x = max(0, min(w - self.patch_size, patch_x))
            patch_y = max(0, min(h - self.patch_size, patch_y))
            
            # Avoid duplicate patches
            coord_key = (patch_x, patch_y)
            if coord_key in used_coords:
                continue
            used_coords.add(coord_key)
            
            # Count defect pixels in patch
            patch_defect_mask = defect_mask[patch_y:patch_y+self.patch_size, 
                                           patch_x:patch_x+self.patch_size]
            defect_pixels_in_patch = np.count_nonzero(patch_defect_mask)
            
            # Check NG conditions
            overlap_ratio = defect_pixels_in_patch / total_defect_pixels if total_defect_pixels > 0 else 0
            is_ng = (overlap_ratio >= min_overlap_ratio) and (defect_pixels_in_patch >= min_defect_pixels)
            
            if is_ng:
                # Extract patch
                patch = self._extract_patch(image_rgb, patch_x, patch_y)
                
                # Apply preprocessing if patch_first mode
                if self.hist_norm_mode == 'patch_first' or self.fda_mode == 'patch_first':
                    patch = self._apply_preprocessing_patch_first(patch, patch_x, patch_y)
                
                # Generate filename
                filename = f"cg_{image_id}_rnd{len(ng_patches)+1:02d}_NG.png"
                ng_patches.append((patch, filename))
                
                # print(f"    Generated NG patch {len(ng_patches)}: {filename} (defect pixels: {defect_pixels_in_patch}, ratio: {overlap_ratio:.2f})")
        
        if len(ng_patches) < patches_per_defect:
            print(f"  Warning: Only generated {len(ng_patches)}/{patches_per_defect} NG patches after {attempts} attempts")
        
        print(f"    Patch generation: {time.time() - patch_start:.2f}s")
        print(f"  Total processing time: {time.time() - start_time:.2f}s")
        
        return ng_patches
    
    def collect_ng_source_images(self) -> List[Tuple[str, str, str]]:
        """Collect all NG source images across multiple directories."""
        ng_source_dirs = self.config.get('ng_source_dirs', [])
        rgb_subdir = self.config.get('rgb_subdir', 'post_rgb.png')
        seg_subdir = self.config.get('seg_subdir', 'post_semantic_segmentation.png')
        
        source_images = []
        
        for source_dir in ng_source_dirs:
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory not found: {source_dir}")
                continue
            
            possible_rgb_dirs = [os.path.join(source_dir, rgb_subdir), os.path.join(source_dir, 'rgb'), source_dir]
            possible_seg_dirs = [os.path.join(source_dir, seg_subdir), os.path.join(source_dir, 'segmentation'), source_dir]
            
            rgb_dir = next((d for d in possible_rgb_dirs if os.path.exists(d)), None)
            seg_dir = next((d for d in possible_seg_dirs if os.path.exists(d)), None)
            
            if not rgb_dir or not seg_dir:
                print(f"Warning: Could not find RGB or segmentation directory in {source_dir}")
                continue
            
            rgb_files_set = set()
            for pattern in ["post_rgb_*.png", "*.png", "rgb_*.png"]:
                matched = glob.glob(os.path.join(rgb_dir, pattern))
                rgb_files_set.update(matched) 
            rgb_files = sorted(list(rgb_files_set))

            for rgb_path in rgb_files:
                filename = os.path.basename(rgb_path)
                
                # Extract image ID
                if 'post_rgb_' in filename:
                    image_id = filename.replace('post_rgb_', '').replace('.png', '')
                elif 'rgb_' in filename:
                    image_id = filename.replace('rgb_', '').replace('.png', '')
                else:
                    image_id = filename.replace('.png', '')
                
                # Find corresponding segmentation
                seg_path = None
                for seg_pattern in [
                    f"post_semantic_segmentation_{image_id}.png",
                    f"semantic_segmentation_{image_id}.png",
                    f"seg_{image_id}.png",
                    filename.replace('.png', '_seg.png'),
                    filename
                ]:
                    potential_path = os.path.join(seg_dir, seg_pattern)
                    if os.path.exists(potential_path):
                        seg_path = potential_path
                        break
                
                if seg_path:
                    source_images.append((rgb_path, seg_path, image_id))
                else:
                    print(f"Warning: Segmentation file not found for {rgb_path}")
                    
        print(f"Found {len(source_images)} unique NG source images")
        return source_images

    def split_train_val_images(self, source_images: List[Tuple[str, str, str]]) -> Tuple[List, List]:
        """Split source images into train and val sets."""
        train_ratio = self.config.get('train_val_split', 0.8)
        
        # Shuffle for random split
        shuffled_images = source_images.copy()
        random.shuffle(shuffled_images)
        
        split_idx = int(len(shuffled_images) * train_ratio)
        train_images = shuffled_images[:split_idx]
        val_images = shuffled_images[split_idx:]
        
        print(f"Split {len(source_images)} images: {len(train_images)} train, {len(val_images)} val")
        return train_images, val_images
    
    def generate_ng_patches(self, source_images: List[Tuple[str, str, str]], split_name: str) -> List[Tuple[np.ndarray, str]]:
        """Generate NG patches from source images."""
        ng_patches = []
        
        # Apply test mode limit if enabled
        if self.test_mode:
            source_images = source_images[:self.test_image_limit]
            print(f"Test mode: Processing only {len(source_images)} images for {split_name}")
        
        # Setup output directory for incremental saving
        output_dir = self.config.get('output_dir', 'patches_dataset/random_patches')
        save_dir = os.path.join(output_dir, split_name.lower(), 'NG')
        os.makedirs(save_dir, exist_ok=True)
        
        for i, (rgb_path, seg_path, image_id) in enumerate(source_images, 1):
            print(f"\n[{i}/{len(source_images)}] Processing {split_name} NG image: {image_id}")
            
            patches = self._generate_ng_patches_for_image(rgb_path, seg_path, image_id)
            ng_patches.extend(patches)
            
            print(f"  -> Generated {len(patches)} NG patches for this image")
           
            # Save patches incrementally
            for patch, filename in patches:
                patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, patch_bgr)
            
            print(f"  -> Saved {len(patches)} patches to {save_dir}")
        
        print(f"\nTotal {split_name} NG patches generated and saved: {len(ng_patches)}")
       
            
        return ng_patches
    
    def save_patches(self, patches: List[Tuple[np.ndarray, str]], output_dir: str, split: str, label: str):
        """Save patches to output directory."""
        if not patches:
            print(f"  No {label} patches to save for {split}")
            return
        
        save_dir = os.path.join(output_dir, split, label)
        os.makedirs(save_dir, exist_ok=True)
        
        for patch, filename in patches:
            # Convert RGB to BGR for OpenCV saving
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, patch_bgr)
        
        print(f"  Saved {len(patches)} {label} patches to {save_dir}")
    
    def run(self):
        """Main execution method."""
        print("=" * 60)
        print("RANDOM PATCH EXTRACTION PIPELINE")
        print("=" * 60)
        
        # Collect NG source images
        print("\nCollecting NG source images...")
        source_images = self.collect_ng_source_images()
        if not source_images:
            print("No NG source images found. Exiting.")
            return
        
        # Split train/val
        print("\nSplitting images into train/val sets...")
        train_images, val_images = self.split_train_val_images(source_images)

        # Generate NG patches
        print("\n" + "=" * 60)
        print("GENERATING NG PATCHES")
        print("=" * 60)
        train_ng_patches = self.generate_ng_patches(train_images, "TRAIN")
        val_ng_patches = self.generate_ng_patches(val_images, "VAL")
        
        # Calculate OK patch counts based on NG:OK ratio
        ng_ok_ratio = self.config.get('ng_ok_ratio', [6, 4])
        ng_ratio, ok_ratio = ng_ok_ratio
    
        train_ok_count = int(len(train_ng_patches) * ok_ratio / ng_ratio) if len(train_ng_patches) > 0 else 0
        val_ok_count = int(len(val_ng_patches) * ok_ratio / ng_ratio) if len(val_ng_patches) > 0 else 0
        
        # train_ok_count = int(100 * ok_ratio / ng_ratio) if 10 > 0 else 0
        # val_ok_count = int(50 * ok_ratio / ng_ratio) if 5 > 0 else 0
        

        print("\n" + "=" * 60)
        print("PATCH COUNT SUMMARY")
        print("=" * 60)
        print(f"NG:OK ratio target: {ng_ratio}:{ok_ratio}")
        # print(f"Train - NG: {len(train_ng_patches)}")
        # print(f"Val    - NG: {len(val_ng_patches)}")
        print(f"Target OK - Train: {train_ok_count}, Val: {val_ok_count}")
        
        # Generate OK patches
        print("\n" + "=" * 60)
        print("GENERATING OK PATCHES")
        print("=" * 60)
        train_ok_patches = []
        val_ok_patches = []
        
        if train_ok_count > 0:
            train_ok_patches = self.generate_ok_patches(train_ok_count)
        if val_ok_count > 0:
            val_ok_patches = self.generate_ok_patches(val_ok_count)
        
        # Setup output directories
        output_dir = self.config.get('output_dir', 'patches_dataset/random_patches')
        
        # Save patches
        print("\n" + "=" * 60)
        print("SAVING PATCHES")
        print("=" * 60)
        # NG patches already saved incrementally during generation
        print("NG patches already saved incrementally during generation")
        self.save_patches(train_ok_patches, output_dir, 'train', 'OK')
        self.save_patches(val_ok_patches, output_dir, 'val', 'OK')
        
        # Print final summary
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"\nTRAIN SET:")
        print(f"  OK: {len(train_ok_patches)}")
        print(f"  NG: {len(train_ng_patches)}")
        print(f"  Total: {len(train_ok_patches) + len(train_ng_patches)}")
        if len(train_ok_patches) + len(train_ng_patches) > 0:
            train_ng_ratio = len(train_ng_patches) / (len(train_ok_patches) + len(train_ng_patches)) * 100
            print(f"  NG ratio: {train_ng_ratio:.1f}%")
        
        print(f"\nVAL SET:")
        print(f"  OK: {len(val_ok_patches)}")
        print(f"  NG: {len(val_ng_patches)}")
        print(f"  Total: {len(val_ok_patches) + len(val_ng_patches)}")
        if len(val_ok_patches) + len(val_ng_patches) > 0:
            val_ng_ratio = len(val_ng_patches) / (len(val_ok_patches) + len(val_ng_patches)) * 100
            print(f"  NG ratio: {val_ng_ratio:.1f}%")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    # Default config path - can be overridden via command line
    config_path = 'config.yaml'
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_path)
    
    # Check if random patch extraction is enabled
    random_patch_config = config.get('random_patch', {})
    if not random_patch_config.get('enabled', False):
        print("Random patch extraction is disabled in config. Set 'random_patch.enabled: true' to enable.")
        return
    
    # Run extraction
    extractor = RandomPatchExtractor(random_patch_config)
    extractor.run()


if __name__ == "__main__":
    main()