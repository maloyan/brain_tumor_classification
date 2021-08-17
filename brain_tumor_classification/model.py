import torch
import torch.nn as nn

from monai.networks.nets import SEResNet50, DenseNet201
from efficientnet_pytorch_3d import EfficientNet3D

class BrainTumorClassificationModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.net = SEResNet50(in_channels=1, spatial_dims=3,dropout_prob=0.3, num_classes=1)
        # EfficientNet3D.from_name(
        #     backbone, override_params={"num_classes": 2}, in_channels=1
        # )
        # n_features = self.net._fc.in_features
        # self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        return self.net(x)
