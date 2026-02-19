import os
import json
import cv2
import numpy as np
import glob
from pathlib import Path

# --- TASK 5: Background Masking ---
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
        
        if self.original_size:
            orig_w, orig_h = self.original_size
            scale_x, scale_y = w / orig_w, h / orig_h
        else:
            scale_x = scale_y = 1.0
        
        for shape in self.mask_data.get('shapes', []):
            if shape['label'] == 'target_object':
                pts = np.array(shape['points'], dtype=np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
        return mask


def run_patch_pipeline(img_dir, npy_dir, json_mask_dir, output_root, patch_size=224, stride=112):
    # Setup Folders
    for label in ["OK", "NG", "visualization"]:
        os.makedirs(os.path.join(output_root, label), exist_ok=True)

    # Distinct colors for overlapping NG patches (BGR format)
    NG_COLORS = [
        (0, 0, 255),    # Red
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
        (0, 255, 255),  # Yellow-ish
        (255, 255, 0),  # Cyan
        (255, 0, 0),    # Blue
        (128, 0, 128),  # Purple
        (0, 128, 255),  # Sky Blue
        (128, 128, 0)   # Olive
    ]

    img_files = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_path in img_files:
        base_name = Path(img_path).stem
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        masker = BackgroundMasking(json_path=json_mask_dir, original_size=(w, h))
        full_mask = masker.get_mask(h, w)
        img_masked = cv2.bitwise_and(img, img, mask=full_mask)

        idx = base_name.split('_')[-1]
        npy_path = os.path.join(npy_dir, f"bounding_box_2d_tight_{idx}.npy")
        bboxes = np.load(npy_path) if os.path.exists(npy_path) else []

        # Visualization base
        vis_img = img_masked.copy()
        
        y_coords = list(range(0, h - patch_size + 1, stride))
        if y_coords[-1] != h - patch_size: y_coords.append(h - patch_size)
        x_coords = list(range(0, w - patch_size + 1, stride))
        if x_coords[-1] != w - patch_size: x_coords.append(w - patch_size)

        count_ng = 0

        for r, y in enumerate(y_coords):
            for c, x in enumerate(x_coords):
                patch_data = img_masked[y:y+patch_size, x:x+patch_size]
                if np.count_nonzero(patch_data) < 10: continue 

                # Consistent Label Logic
                is_ng = False
                for bbox in bboxes:
                    _, bx1, by1, bx2, by2, _ = bbox
                    ix1, iy1 = max(x, bx1), max(y, by1)
                    ix2, iy2 = min(x + patch_size, bx2), min(y + patch_size, by2)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        inter_mask = full_mask[int(iy1):int(iy2), int(ix1):int(ix2)]
                        if np.any(inter_mask > 0):
                            is_ng = True
                            break
                
                label = "NG" if is_ng else "OK"
                
                # --- COLORFUL VISUALIZATION ---
                if label == "NG":
                    # Cycle through colors and apply a 5px inward shift for each NG found
                    color = NG_COLORS[count_ng % len(NG_COLORS)]
                    padding = 0
                    thickness = 2
                    count_ng += 1
                else:
                    color = (0, 255, 0) # Constant Green for OK
                    padding = 0
                    thickness = 1

                # Draw the patch boundary
                cv2.rectangle(vis_img, (x + padding, y + padding), 
                              (x + patch_size - padding, y + patch_size - padding), 
                              color, thickness)
                
                # Add Grid Text (r, c)
                # Position logic: OK labels at top, NG labels shifted with their boxes
                text = f"r{r}c{c}"
                text_y = y + 15 if label == "OK" else y + 30 + padding
                cv2.putText(vis_img, text, (x + 5 + padding, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Save the patch file
                cv2.imwrite(os.path.join(output_root, label, f"{base_name}_r{r}_c{c}_{label}.png"), patch_data)

        for bbox in bboxes:
            _, x1, y1, x2, y2, _ = bbox
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
            cv2.putText(vis_img, "GT SCRATCH", (int(x1), int(y1)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        vis_save_path = os.path.join(output_root, "visualization", f"{base_name}_map.jpg")
        cv2.imwrite(vis_save_path, vis_img)
        print(f"Processed {base_name}: {count_ng} NG patches saved with distinct colors.")

if __name__ == "__main__":
    run_patch_pipeline(
        img_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/rgb",
        npy_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/bounding_box_2d_tight", 
        json_mask_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/mask/cg_mask_label.json",
        output_root="patches_dataset_by_bbox"
    )