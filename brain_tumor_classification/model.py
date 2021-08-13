import torch
import torch.nn as nn

from efficientnet_pytorch_3d import EfficientNet3D

class BrainTumorClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet3D.from_name(
            "efficientnet-b0", override_params={"num_classes": 2}, in_channels=1
        )
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        return self.net(x)
