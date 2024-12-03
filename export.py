import torch
import torchvision
from models import FaceNetModel

device  = torch.device('cpu')

model = FaceNetModel(64, 40000).to(device)
checkpoint = torch.load('C:/a/facenet-light/log/checkpoint_epoch299.pth')

model.load_state_dict(checkpoint['state_dict'])

#prune last layer: norm as not supported by onnx->opencv
model.l2_norm = lambda x: x

model.eval()

dummy_input = torch.randn(1, 3, 96, 96, device = 'cpu')
input_names = ['in']
output_names = ['out']

torch.onnx.export(model, dummy_input, "output9.onnx", verbose=True, input_names=input_names, output_names=output_names)