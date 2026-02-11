"""
Albumentations-compatible wrappers for custom transforms
"""
import json
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

    def __init__(self, json_path: str=None, always_apply=False, p=1.0):
        print("init bg masking")
        super().__init__(always_apply, p)
        self.json_path = Path(json_path)
        self.mask_pth = self._load_json()

    def _load_json(self):
        print(f"Loading background mask JSON: {self.json_path}")
        if self.json_path.exists():
            print(f"JSON found. Loading mask data from {self.json_path}")
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return None

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        print("==============================================")
        print(f"Applying BackgroundMasking with JSON: {self.json_path}")
        print(f"Applying Masking. Image shape: {image.shape}")
        if self.mask_pth is None:
            return image

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        found_shape = False
        for shape in self.mask_pth.get('shapes', []):
            if shape['label'] == 'target_object':
                pts = np.array(shape['points'], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
                found_shape = True

        if not found_shape:
            return image
        
        result = np.zeros_like(image)
        
        is_object = mask > 0
        result[is_object] = image[is_object]

        if image.shape[2] == 4:
            result[~is_object, 3] = 255

        return result



