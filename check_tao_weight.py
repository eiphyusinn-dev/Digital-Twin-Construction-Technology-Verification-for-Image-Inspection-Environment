import torch
ckpt = torch.load('/home/gwm-279/Desktop/tao_experiments/pretrained_models/pretrained_convnextv2_vconvnextv2_large_v1.0/convnextv2_large_trainable_v1.0.pth')
print(list(ckpt['state_dict'].keys()))