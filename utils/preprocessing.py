"""
Albumentations-compatible wrappers for custom transforms
"""

import cv2
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
    Optimized for a single reference image file.
    """
    def __init__(self, reference_images_path: str, beta_limit: float = 0.0005, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.reference_path = Path(reference_images_path)
        self.beta = beta_limit

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # Check if the file exists
        if not self.reference_path.exists():
            print(f'FDA Skip: Reference file not found at {self.reference_path}')
            return image
        
        try:
            # Load and prep target image (style)
            im_trg = Image.open(self.reference_path).convert('RGB')
            im_trg = im_trg.resize((image.shape[1], image.shape[0]), Image.BICUBIC)
            im_trg = np.asarray(im_trg, np.float32) / 255.0
            
            # Prep source image (content)
            im_src = image.astype(np.float32) / 255.0
            
            # Transpose to (C, H, W) for the FDA utility
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

class BackgroundMasking(A.ImageOnlyTransform):
    """Skip for now."""
    def apply(self, image, **params): return image

