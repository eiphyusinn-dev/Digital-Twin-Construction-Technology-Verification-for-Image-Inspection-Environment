import torch
import cv2
import numpy as np
import yaml
import os
import argparse
from pathlib import Path
from dataset import get_dataloaders

def verify_batches_with_grid(num_batches=5, output_dir="dataloader_grid_check"):
    # 1. Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['coordconv']['enabled'] = True
    train_loader, _ = get_dataloaders(config)
    
    # Image and Grid Params
    H, W = 4608, 5328
    grid_rows = config['coordconv']['grid_rows'] # 41
    grid_cols = config['coordconv']['grid_cols'] # 47
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizing {num_batches} batches with full grid overlay...")

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches: break
        
        images, labels, coords = batch
        # Create canvas at 1/4 scale
        canvas = np.zeros((H // 4, W // 4, 3), dtype=np.uint8)

        # --- DRAW FULL GRID LINES ---
        # Draw horizontal lines for Rows
        for r in range(grid_rows):
            # Calculate pixel Y for this row index
            norm_y = r / (grid_rows - 1)
            y_pixel = int(norm_y * (H - 224)) // 4
            cv2.line(canvas, (0, y_pixel), (W // 4, y_pixel), (50, 50, 50), 1)
            
        # Draw vertical lines for Columns
        for c in range(grid_cols):
            # Calculate pixel X for this column index
            norm_x = c / (grid_cols - 1)
            x_pixel = int(norm_x * (W - 224)) // 4
            cv2.line(canvas, (x_pixel, 0), (x_pixel, H // 4), (50, 50, 50), 1)

        # 2. Place Patches on the Grid
        for i in range(images.size(0)):
            r_norm, c_norm = coords[i].numpy()
            
            # Map back to raw indices for labeling
            raw_r = int(round(r_norm * (grid_rows - 1)))
            raw_c = int(round(c_norm * (grid_cols - 1)))
            
            # Map to pixel space
            y_top = int(r_norm * (H - 224))
            x_left = int(c_norm * (W - 224))
            
            # Un-normalize patch image
            patch = images[i].permute(1, 2, 0).numpy()
            patch = (patch * 0.229 + 0.485) * 255 
            patch = patch.clip(0, 255).astype(np.uint8)
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            
            # Resize and place
            small_patch = cv2.resize(patch, (224 // 4, 224 // 4))
            py, px = y_top // 4, x_left // 4
            ph, pw = small_patch.shape[:2]
            
            # Blend patch onto canvas so grid is still visible
            roi = canvas[py:py+ph, px:px+pw]
            canvas[py:py+ph, px:px+pw] = cv2.addWeighted(roi, 0.3, small_patch, 0.7, 0)
            
            # Label
            cv2.rectangle(canvas, (px, py), (px+pw, py+ph), (0, 255, 0), 1)
            cv2.putText(canvas, f"{raw_r},{raw_c}", (px+2, py+12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        save_path = os.path.join(output_dir, f"grid_batch_{batch_idx:03d}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f"Saved Batch {batch_idx}")

if __name__ == "__main__":
    verify_batches_with_grid(num_batches=5)