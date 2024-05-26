import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.transforms import Normalize

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        for x in range(14):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for param in self.slice1.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = self.slice1(x)
        y = self.slice1(y)
        loss = torch.mean((x - y) ** 2)
        return loss
