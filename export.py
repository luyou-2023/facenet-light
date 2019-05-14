import torch
import torchvision
from models import FaceNetModel

device  = torch.device('cpu')

model = FaceNetModel(128, 10000).to(device)
checkpoint = torch.load('C:/a/facenet-light/log/checkpoint_epoch1.pth')

model.load_state_dict(checkpoint['state_dict'])
model.eval()

dummy_input = torch.randn(1, 3, 96, 96, device = 'cpu')
input_names = ['in']
output_names = ['out']

torch.onnx.export(model, dummy_input, "output.onnx", verbose=True, input_names=input_names, output_names=output_names)