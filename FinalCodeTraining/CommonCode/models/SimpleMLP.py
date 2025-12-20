import torch
import torch.nn as nn
import torch.nn.functional as F  # only if you use F.*
from config import emotion_to_idx
# =========================
# Model (simple head)
# =========================
class FrameClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512,dropout=0.3, num_classes=len(emotion_to_idx)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.net(x)
