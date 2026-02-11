"""
Albumentations-compatible wrappers for custom transforms
"""

import cv2
import json
from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
try:
    from utils.fda_utils import FDA_source_to_target_np 
except ImportError:
    print("Warning: utils.py not found. FDA will be skipped.")

class FDATransform(A.ImageOnlyTransform):
    """
    Fourier Domain Adaptation using custom utils.FDA_source_to_target_np.
    Optimized: loads reference image once in constructor.
    """
    def __init__(self, reference_images_path: str, beta_limit: float = 0.0005, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.reference_path = Path(reference_images_path)
        self.beta = beta_limit
        self.reference_image = self._load_reference_image()

    def _load_reference_image(self):
        """Load and cache reference image for performance."""
        if not self.reference_path.exists():
            print(f'FDA Skip: Reference file not found at {self.reference_path}')
            return None
        
        try:
            # Load reference image once
            ref_img = Image.open(self.reference_path).convert('RGB')
            return ref_img
        except Exception as e:
            print(f"FDA Error loading reference: {e}")
            return None

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if self.reference_image is None:
            return image
        
        try:
            # Resize cached reference image to match input size
            im_trg = self.reference_image.resize((image.shape[1], image.shape[0]), Image.BICUBIC)
            im_trg = np.asarray(im_trg, np.float32) / 255.0
            
            # Prep source image (content)
            im_src = image.astype(np.float32) / 255.0
            
            # Transpose to (C, H, W) for FDA utility
            im_src = im_src.transpose((2, 0, 1))
            im_trg = im_trg.transpose((2, 0, 1))
            
            # Apply custom FDA function
            src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=self.beta)
            
            # Convert back to (H, W, C) and uint8
            src_in_trg = src_in_trg.transpose((1, 2, 0))
            return np.clip(src_in_trg * 255.0, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"FDA Error: {e}")
            return image

    @property
    def targets_as_params(self):
        return ["image"]
  
class HistogramNormalization(A.ImageOnlyTransform):
    """
    Global Histogram Equalization (GHE) based on OpenCV logic.
    """
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equ = cv2.equalizeHist(gray)
        return cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)

# class BackgroundMasking(A.ImageOnlyTransform):
#     """Skip for now."""
#     def apply(self, image, **params): return image

class BackgroundMasking(A.ImageOnlyTransform):

    def __init__(self, json_path: str=None, original_size: tuple=None, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.json_path = Path(json_path)
        self.original_size = original_size  # (width, height) of original image
        self.mask_pth = self._load_json()

    def _load_json(self):
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return None

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if self.mask_pth is None:
            print(f"[BG Mask] JSON not loaded, path: {self.json_path}")
            return image

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        found_shape = False
        
        # Calculate scale factors if original_size is provided
        if self.original_size:
            orig_w, orig_h = self.original_size
            scale_x = w / orig_w
            scale_y = h / orig_h
        else:
            scale_x = scale_y = 1.0
        
        for shape in self.mask_pth.get('shapes', []):
            if shape['label'] == 'target_object':
                pts = np.array(shape['points'], dtype=np.float32)
                # Scale points to match resized image
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
                found_shape = True

        if not found_shape:
            print("[BG Mask] No 'target_object' label found")
            return image
        
        result = np.zeros_like(image)
        is_object = mask > 0
        result[is_object] = image[is_object]

        if image.shape[2] == 4:
            result[~is_object, 3] = 255

        return result

