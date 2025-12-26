import torch
from src.model.classifier import LitASLViT

def load_model(checkpoint_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LitASLViT.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model
