import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Load the pretrained Resnet-18 model
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        # Freeze the layers
        for p in resnet18.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Extract features from the last pooling layer
        pooled_feature = self.resnet18(x)
        # concatenate features for the targets
        concat_features=pooled_feature.view(pooled_feature.size(0), -1)
        # Normalization
        normed_feature = concat_features / torch.clamp(torch.norm(concat_features, 2, 1, keepdim=True), min=1e-6)

        return normed_feature


