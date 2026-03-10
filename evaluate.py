#!/usr/bin/env python3
"""
Evaluation script for patch-based defect detection.
"""

import os
import re 
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple

class PatchEvaluator:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        # Pre-compile regex for filename parsing: base_r{row}_c{col}_{LABEL}.png
        self.filename_re = re.compile(r"(.+)_r(\d+)_c(\d+)_(NG|OK)$")

    def load_inference_csvs(self, csv_dir: str) -> Dict[str, List[Dict]]:
        """Load all patch probability CSV files from directory."""
        csv_files = list(Path(csv_dir).glob("patch_probs_*.csv"))
        all_data = {f.stem.replace("patch_probs_", ""): pd.read_csv(f).to_dict('records') for f in csv_files}
        print(f"Loaded {len(csv_files)} CSV files with patch probabilities")
        return all_data
    
    def load_ground_truth(self, gt_dir: str) -> Dict[str, Dict[Tuple[int, int], str]]:
        """Load ground truth labels from NG/OK folders using Regex."""
        gt_data = defaultdict(dict)
        gt_path = Path(gt_dir)
        
        for label in ["NG", "OK"]:
            label_dir = gt_path / label
            if not label_dir.exists():
                continue
            for img_file in label_dir.glob("*.png"):
                match = self.filename_re.match(img_file.stem)
                if match:
                    base_name, r, c, _ = match.groups()
                    gt_data[base_name][(int(r), int(c))] = label
                else:
                    print(f"Warning: Could not parse format: {img_file.name}")
        
        print(f"Loaded ground truth for {len(gt_data)} images")
        return gt_data
    
    def create_visual_comparison(self, inference_data: Dict[str, List[Dict]], 
                                ground_truth: Dict[str, Dict], output_dir: str, threshold: float):
        """Create visual comparison of ground truth vs predictions."""
        vis_dir = Path(output_dir) / f"threshold_{threshold}" / "visual_comparisons"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        colors = {'TP': 'green', 'FP': 'red', 'FN': 'orange', 'TN': 'lightblue'}

        for image_name, patches in inference_data.items():
            if image_name not in ground_truth: continue
                
            gt_patches = ground_truth[image_name]
            categories = defaultdict(lambda: {'r': [], 'c': []})
            
            for p in patches:
                row, col, prob = p['row_idx'], p['col_idx'], p['ng_probability']
                pred_ng = prob >= threshold
                actual_ng = gt_patches.get((row, col)) == "NG"
                
                cat = ('TP' if pred_ng and actual_ng else 
                       'FP' if pred_ng and not actual_ng else 
                       'FN' if not pred_ng and actual_ng else 'TN')
                categories[cat]['r'].append(row)
                categories[cat]['c'].append(col)

            plt.figure(figsize=(8, 6))
            for cat, data in categories.items():
                plt.scatter(data['c'], data['r'], c=colors[cat], s=50, alpha=0.8, 
                            label=f'{cat} ({len(data["r"])})', edgecolors='black', linewidth=0.5)
            
            plt.title(f'Patch Comparison: {image_name} (Threshold: {threshold})')
            plt.legend()
            plt.gca().invert_yaxis()
            plt.savefig(vis_dir / f"{image_name}_comparison.png", dpi=150)
            plt.close()
            
    def _calculate_metrics(self, tp, fp, tn, fn):
        """Standard metric calculation helper."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        return precision, recall, f1, acc

    def evaluate_patches(self, inference_data: Dict, ground_truth: Dict) -> Dict:
        """Evaluate patch-level predictions."""
        tp, fp, tn, fn = 0, 0, 0, 0
        all_probs, all_labels = [], []

        for image_name, patches in inference_data.items():
            if image_name not in ground_truth: continue
            
            gt_img = ground_truth[image_name]
            for p in patches:
                r, c, prob = p['row_idx'], p['col_idx'], p['ng_probability']
                gt_label = gt_img.get((r, c))
                if gt_label is None: continue
                
                actual = 1 if gt_label == "NG" else 0
                pred = 1 if prob >= self.threshold else 0
                
                all_probs.append(prob)
                all_labels.append(actual)
                
                if actual == 1:
                    if pred == 1: tp += 1
                    else: fn += 1
                else:
                    if pred == 1: fp += 1
                    else: tn += 1
        
        prec, rec, f1, acc = self._calculate_metrics(tp, fp, tn, fn)
        return {
            'patch_level': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 
                            'precision': prec, 'recall': rec, 'f1_score': f1, 'accuracy': acc, 
                            'total_patches': tp+fp+tn+fn},
            'all_probs': all_probs, 'all_labels': all_labels
        }
    
    def evaluate_images(self, inference_data: Dict, ground_truth: Dict) -> Dict:
        """Evaluate work-level predictions using OR aggregation."""
        tp, fp, tn, fn = 0, 0, 0, 0
        for name, patches in inference_data.items():
            if name not in ground_truth: continue
            pred_ng = any(p['ng_probability'] >= self.threshold for p in patches)
            actual_ng = any(v == "NG" for v in ground_truth[name].values())
            
            if actual_ng:
                if pred_ng: tp += 1
                else: fn += 1
            else:
                if pred_ng: fp += 1
                else: tn += 1
        
        prec, rec, f1, acc = self._calculate_metrics(tp, fp, tn, fn)
        return {'work_level': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 
                               'precision': prec, 'recall': rec, 'f1_score': f1, 'accuracy': acc, 
                               'total_images': tp+fp+tn+fn}}

    def generate_plots(self, results, roc_results, output_dir):
        """Unified plot generator with threshold-specific folders."""
        threshold = results['threshold']
        threshold_dir = Path(output_dir) / f"threshold_{threshold}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrices
        for level in ['patch', 'work']:
            res = results[f'{level}_level']
            cm = np.array([[res['tn'], res['fp']], [res['fn'], res['tp']]])
            
            plt.figure(figsize=(7, 6))
            plt.imshow(cm, cmap=plt.cm.Blues)
            plt.title(f'{level.capitalize()}-level Confusion Matrix\n(Threshold: {threshold})', pad=20)
            
            # --- FIX: ADD CLASS LABELS ---
            classes = ['OK', 'NG']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)

            # --- ADD BORDER AND GRID LINES ---
            ax = plt.gca()
            # This adds a solid border around the entire graph
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('black')
            
            # This adds the white lines between the OK/NG boxes
            ax.set_xticks(np.arange(len(classes)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(classes)) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
            ax.tick_params(which="minor", size=0)
            # ---------------------------------

            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'), 
                         ha="center", va="center", 
                         color="white" if cm[i, j] > cm.max()/2 else "black",
                         fontsize=14, fontweight='bold')
            
            plt.ylabel('Actual Label (Ground Truth)', fontsize=12)
            plt.xlabel('Predicted Label (Inference)', fontsize=12)
            plt.tight_layout()
            plt.savefig(threshold_dir / f'{level}_confusion_matrix.png')
            plt.close()

        # ROC Curve
        plt.figure(figsize=(6, 5))
        plt.plot(roc_results['fpr'], roc_results['tpr'], label=f"AUC={results['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], '--', color='navy')
        plt.title(f'ROC Curve (Threshold: {threshold})'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(threshold_dir / 'roc_curve.png')
        plt.close()

    def evaluate(self, inference_csv_dir: str, ground_truth_dir: str, output_dir: str = "evaluation_results") -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        inf_data = self.load_inference_csvs(inference_csv_dir)
        gt_data = self.load_ground_truth(ground_truth_dir)

        p_res = self.evaluate_patches(inf_data, gt_data)
        i_res = self.evaluate_images(inf_data, gt_data)
        
        fpr, tpr, _ = roc_curve(p_res['all_labels'], p_res['all_probs'])
        roc_auc = auc(fpr, tpr)
        
        results = {'threshold': self.threshold, 'patch_level': p_res['patch_level'], 
                   'work_level': i_res['work_level'], 'roc_auc': roc_auc}
        
        self.generate_plots(results, {'fpr': fpr, 'tpr': tpr}, output_dir)
        self.create_visual_comparison(inf_data, gt_data, output_dir, self.threshold)
        
        # Save results in threshold-specific folder
        threshold_dir = Path(output_dir) / f"threshold_{self.threshold}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        with open(threshold_dir / f'evaluation_results_threshold_{self.threshold}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.print_summary(results)
        return results

    def print_summary(self, r):
        print("\n" + "="*30 + f"\nRESULTS (Thresh: {r['threshold']})\n" + "="*30)
        for lvl in ['patch_level', 'work_level']:
            print(f"\n{lvl.upper()}:")
            for k, v in r[lvl].items(): print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\nROC AUC: {r['roc_auc']:.3f}")
        print("✅ REQ MET" if r['work_level']['recall'] >= 0.999 else "❌ REQ NOT MET")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_csvs', required=True)
    parser.add_argument('--ground_truth', required=True)
    parser.add_argument('--output_dir', default='evaluation_results')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.5,0.6,0.7,0.8,0.9])
    args = parser.parse_args()

    evaluator = PatchEvaluator(threshold=0.7)  # Initial threshold, will be overridden
    
    if args.thresholds:
        multi_results = []
    for t in args.thresholds:
        evaluator.threshold = t
        res = evaluator.evaluate(args.inference_csvs, args.ground_truth, args.output_dir)
        multi_results.append({'threshold': t, 'work_recall': res['work_level']['recall'], 
                                 'work_precision': res['work_level']['precision']})
        pd.DataFrame(multi_results).to_csv(os.path.join(args.output_dir, 'multi_threshold_results.csv'), index=False)

if __name__ == "__main__":
    main()