import os
import cv2
import numpy as np
import glob
from pathlib import Path

def run_patch_pipeline_no_mask(img_dir, seg_dir, output_root, patch_size=224, stride=112):
    # Setup Folders
    for label in ["OK", "NG", "visualization"]:
        os.makedirs(os.path.join(output_root, label), exist_ok=True)

    # Defect color: (204, 61, 61) in RGB -> (61, 61, 204) in BGR
    DEFECT_BGR = np.array([61, 61, 204]) 
    ALPHA = 0.25 

    img_files = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_path in img_files:
        base_name = Path(img_path).stem
        img = cv2.imread(img_path) # Original image (No Masking)
        if img is None: continue
        h, w = img.shape[:2]

        # 1. Load Defect Mask (Original segmentation map)
        idx = base_name.split('_')[-1]
        seg_path = os.path.join(seg_dir, f"instance_segmentation_{idx}.png")
        if not os.path.exists(seg_path):
            print(f"Warning: Seg map not found at {seg_path}")
            continue
            
        seg_map = cv2.imread(seg_path) 
        # Isolate the defect pixels
        defect_mask = cv2.inRange(seg_map, DEFECT_BGR, DEFECT_BGR)

        # 2. Prepare Visualization Layers
        vis_img = img.copy()
        overlay = vis_img.copy()
        
        # Calculate full grid coordinates
        y_coords = list(range(0, h - patch_size + 1, stride))
        if y_coords[-1] != h - patch_size: y_coords.append(h - patch_size)
        x_coords = list(range(0, w - patch_size + 1, stride))
        if x_coords[-1] != w - patch_size: x_coords.append(w - patch_size)

        count_ng = 0

        # Loop through the entire grid
        for r, y in enumerate(y_coords):
            for c, x in enumerate(x_coords):
                # Extract raw patch (contains floor/background)
                patch_data = img[y:y+patch_size, x:x+patch_size]
                
                # Check for defects in this specific window
                patch_defect_pixels = defect_mask[y:y+patch_size, x:x+patch_size]
                num_defect_pixels = np.count_nonzero(patch_defect_pixels)
                
                # Labeling logic
                label = "NG" if num_defect_pixels >= 10 else "OK"
                
                # Visualization for the Map
                if label == "NG":
                    count_ng += 1
                    cv2.rectangle(overlay, (x, y), (x + patch_size, y + patch_size), (0, 0, 255), -1)
                    cv2.rectangle(vis_img, (x, y), (x + patch_size, y + patch_size), (0, 255, 255), 1)
                    cv2.putText(vis_img, f"NG_{r}_{c}", (x + 5, y + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Save the patch with row/col in filename for CoordConv dataloader
                save_filename = f"{base_name}_r{r}_c{c}_{label}.png"
                cv2.imwrite(os.path.join(output_root, label, save_filename), patch_data)

        # Final map generation
        cv2.addWeighted(overlay, ALPHA, vis_img, 1 - ALPHA, 0, vis_img)
        vis_img[defect_mask > 0] = [255, 255, 255] # Draw actual defect in white

        vis_save_path = os.path.join(output_root, "visualization", f"{base_name}_map.jpg")
        cv2.imwrite(vis_save_path, vis_img)
        print(f"Processed {base_name}: Found {count_ng} NG patches (No Masking).")

if __name__ == "__main__":
    run_patch_pipeline_no_mask(
        img_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/rgb",
        seg_dir="/home/gwm-279/Desktop/tao_experiments/custom_pytorch_pipeline/patch_splitting_data/02_omniverse/_defects.pos4/instance_segmentation", 
        output_root="patches_dataset/patches_dataset_unmasked"
    )