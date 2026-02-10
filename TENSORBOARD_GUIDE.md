# TensorBoard Monitoring Guide

## Overview
TensorBoard integration has been added to the training script to provide real-time monitoring of training progress.

## What's Logged
- **Training Metrics:**
  - Train Loss per epoch
  - Train Accuracy per epoch
  - Learning Rate per epoch

- **Validation Metrics:**
  - Validation Loss per epoch
  - Validation Accuracy per epoch

## How to View TensorBoard

### During Training
1. **Start training:**
   ```bash
   python3 train.py
   ```

2. **View the TensorBoard URL output:**
   ```
   TensorBoard logs will be saved to: logs/tensorboard_1641234567
   View TensorBoard with: tensorboard --logdir=logs/tensorboard_1641234567
   ```

3. **Start TensorBoard server:**
   ```bash
   tensorboard --logdir=logs/tensorboard_1641234567
   ```

4. **Open in browser:**
   Navigate to `http://localhost:6006`

### After Training
View any training run:
```bash
tensorboard --logdir=logs/
```

## TensorBoard Features

### Scalars Tab
- **Train/Loss:** Training loss progression
- **Train/Accuracy:** Training accuracy progression  
- **Train/Learning_Rate:** Learning rate schedule
- **Val/Loss:** Validation loss progression
- **Val/Accuracy:** Validation accuracy progression

### Usage Tips
1. **Compare Runs:** TensorBoard automatically loads all runs in the log directory
2. **Smooth Curves:** Use the smoothing slider to reduce noise
3. **Download Data:** Export plots as PNG or CSV
4. **Real-time Updates:** Refresh automatically during training

## Log Locations
- **Default:** `logs/tensorboard_[timestamp]/`
- **Configurable:** Set via `paths.logs` in `config.yaml`
- **Multiple Runs:** All runs stored in `logs/` for easy comparison

## Troubleshooting

### Port Already in Use
```bash
tensorboard --logdir=logs/ --port=6007
```

### Remote Server Access
```bash
tensorboard --logdir=logs/ --host=0.0.0.0 --port=6006
```

### Clear Old Logs
```bash
rm -rf logs/tensorboard_*
```

## Integration Details
The TensorBoard writer is automatically:
- **Initialized** when training starts
- **Updated** after each epoch
- **Closed** when training completes
- **Logged** with timestamp for unique runs

No additional configuration required - just run training and monitor!
