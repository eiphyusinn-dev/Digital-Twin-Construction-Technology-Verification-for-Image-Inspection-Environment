import os
import json
import cv2
import numpy as np
import glob
from pathlib import Path

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

def run_patch_pipeline(img_dir, seg_dir, json_mask_dir, output_root, patch_size=224, stride=112):
    # Setup Folders
    for label in ["OK", "NG", "visualization"]:
        os.makedirs(os.path.join(output_root, label), exist_ok=True)

    # Defect color from your JSON: (204, 61, 61) in RGB -> (61, 61, 204) in BGR
    DEFECT_BGR = np.array([61, 61, 204]) 
    ALPHA = 0.25  # Transparency for the NG highlight

    img_files = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_path in img_files:
        base_name = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        # 1. Apply Background Mask
        masker = BackgroundMasking(json_path=json_mask_dir, original_size=(w, h))
        full_mask = masker.get_mask(h, w)
        img_masked = cv2.bitwise_and(img, img, mask=full_mask)

        # 2. Load and isolate Defect Mask
        idx = base_name.split('_')[-1]
        seg_path = os.path.join(seg_dir, f"instance_segmentation_{idx}.png")
        if not os.path.exists(seg_path):
            print(f"Warning: Seg map not found at {seg_path}")
            continue
            
        seg_map = cv2.imread(seg_path) 
        defect_mask = cv2.inRange(seg_map, DEFECT_BGR, DEFECT_BGR)
        defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=full_mask)

        # 3. Prepare Visualization Layers
        # vis_img will hold the boxes, overlay will hold the transparent fills
        vis_img = img_masked.copy()
        overlay = vis_img.copy()
        
        y_coords = list(range(0, h - patch_size + 1, stride))
        if y_coords[-1] != h - patch_size: y_coords.append(h - patch_size)
        x_coords = list(range(0, w - patch_size + 1, stride))
        if x_coords[-1] != w - patch_size: x_coords.append(w - patch_size)

        count_ng = 0

        for r, y in enumerate(y_coords):
            for c, x in enumerate(x_coords):
                # Extract patch for processing
                patch_data = img_masked[y:y+patch_size, x:x+patch_size]
                
                # Skip nearly empty patches (outside background mask)
                if np.count_nonzero(patch_data) < 10: continue 

                # Check defect mask within this patch
                patch_defect_pixels = defect_mask[y:y+patch_size, x:x+patch_size]
                num_defect_pixels = np.count_nonzero(patch_defect_pixels)
                
                label = "NG" if num_defect_pixels >= 10 else "OK"
                
                # Draw Visualization
                if label == "NG":
                    count_ng += 1
                    # Red semi-transparent fill on overlay
                    cv2.rectangle(overlay, (x, y), (x + patch_size, y + patch_size), (0, 0, 255), -1)
                    # Thin Yellow border for sharp visibility
                    cv2.rectangle(vis_img, (x, y), (x + patch_size, y + patch_size), (0, 255, 255), 1)
                    # Label the patch index
                    cv2.putText(vis_img, f"NG_{r}_{c}", (x + 5, y + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                else:
                    pass

                cv2.imwrite(os.path.join(output_root, label, f"{base_name}_r{r}_c{c}_{label}.png"), patch_data)

        cv2.addWeighted(overlay, ALPHA, vis_img, 1 - ALPHA, 0, vis_img)
        
        vis_img[defect_mask > 0] = [255, 255, 255]

        vis_save_path = os.path.join(output_root, "visualization", f"{base_name}_map.jpg")
        cv2.imwrite(vis_save_path, vis_img)
        print(f"Processed {base_name}: Found {count_ng} NG patches.")

if __name__ == "__main__":
    run_patch_pipeline(
        img_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/rgb",
        seg_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/instance_segmentation", 
        json_mask_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/mask/cg_mask_label.json",
        output_root="patches_dataset"
    )