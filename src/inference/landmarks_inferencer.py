import torch
import torch.nn.functional as F
from src.models.landmarks_mlp import LandmarksMLP


class LandmarksInferencer:
    def __init__(self, checkpoint_path, class_names, device):
        self.class_names = class_names
        self.device = device

        self.model = LandmarksMLP(num_classes=len(class_names))
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, landmarks):
        x = torch.from_numpy(landmarks).float().unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

        return self.class_names[idx.item()], conf.item()
