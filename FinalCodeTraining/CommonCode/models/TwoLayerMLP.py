# models/TwoLayerMLP.py
import torch
import torch.nn as nn

class FrameClassifier(nn.Module):
    """
    Flexible MLP head:
      - If hidden_dim2 is None -> 1 hidden layer
      - Else                    -> 2 hidden layers
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 hidden_dim2: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        # first hidden
        layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout and dropout > 0: layers.append(nn.Dropout(dropout))

        if hidden_dim2 is not None and hidden_dim2 > 0:
            # second hidden
            layers += [nn.Linear(hidden_dim, hidden_dim2), nn.ReLU()]
            if dropout and dropout > 0: layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden_dim2, num_classes)]
        else:
            # straight to logits
            layers += [nn.Linear(hidden_dim, num_classes)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

