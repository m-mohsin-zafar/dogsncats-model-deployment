import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from network import TenLayerConvNet

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')


def get_model():
    chk_pt = os.path.join(checkpoint_dir, 'cnn_trained_10layer.pth')
    model = TenLayerConvNet()
    model.load_state_dict(torch.load(chk_pt, map_location='cpu'), strict=False)
    model.eval()
    return model


def get_tensor(image_bytes):
    transformations = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transformations(image).unsqueeze(0)
