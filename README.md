# Custom PyTorch Pipeline

A custom PyTorch training and inference pipeline for image classification with support for TAO pretrained weights.

## Requirements

- **Python**: 3.10.19
- **PyTorch**: 2.0+ 
- **CUDA**: 11.0+ (for GPU training)

## Architecture

The ConvNeXt-V2 model implementation is based on the official Facebook Research architecture:

- **Source**: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
- **Compatibility**: Modified to support TAO pretrained weights
- **Features**: Standard ConvNeXt-V2 blocks with GRN (Global Response Normalization)

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Or with conda
conda install --file requirements.txt
```

### 2. Prepare Dataset

```
dataset/
├── train/
│   ├── cat/
│   │   ├── image1.jpg
│   │   └── ...
│   └── dog/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── cat/
│   └── dog/
└── classes.txt
```

Create `classes.txt` with your class names:
```
cat
dog
```

### 3. Configure Training

Edit `config.yaml` to match your setup:
- Update dataset paths
- Set number of classes
- Adjust training parameters
- Configure TAO weights path

### 4. Run Training

```bash
python3 train.py
```

### 5. Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir=logs/
# Open http://localhost:6006
```

### 6. Visualize Data Loading

```bash
# Visualize data loading and augmentation pipeline
python3 vis_dataloader.py --n 10

# This script helps you:
# - Verify data loading works correctly
# - Check augmentation effects
# - Check preprocessing steps
# - Validate class name mapping
```

### 7. Run Inference

```bash
# Single image inference
python3 inference.py --model-path checkpoints/best_model.pth --input path/to/image.jpg

# Batch inference
python3 inference.py --model-path checkpoints/best_model.pth --input path/to/images/ --output results.json

# With custom threshold
python3 inference.py --model-path checkpoints/best_model.pth --input path/to/image.jpg --threshold 0.7
```

### Training
- `checkpoints/best_model.pth`: Best model checkpoint
- `logs/training.log`: Training log file
- `logs/tensorboard_*/`: TensorBoard logs

### Inference
- `annotated_images/`: Images with predictions drawn
- `results.json`: Batch inference results (if output specified)


## File Structure

```
├── config.yaml              # Main configuration file
├── train.py                 # Training script
├── inference.py              # Inference script
├── model.py                 # Model definitions
├── model_standard.py         # Standard ConvNeXt-V2 for TAO weights
├── dataset.py               # Dataset and data loading
├── utils/                   # Utility functions
│   ├── preprocessing.py       # Data transforms
│   └── fda_utils.py         # FDA implementation
├── checkpoints/             # Model checkpoints
├── logs/                   # Training logs
├── annotated_images/         # Inference results
└── dataset/                # Training data
```
