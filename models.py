import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18


class FaceNetModel(nn.Module):
    def __init__(self, embedding_size, num_classes, pretrained=False):
        super (FaceNetModel, self).__init__()
        
        self.model            = resnet18(pretrained)

        #prune not needed
        del self.model.layer4
        del self.model.layer3

        self.embedding_size   = embedding_size
        self.model.fc         = nn.Linear(18432, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
    
    
    def l2_norm(self, input):
        n = input.norm(p=2, dim=1, keepdim=True)
        result = input.div(n)
        return result

    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
#        x = self.model.layer3(x)
#        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

#        self.features = x
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        # alpha         = 10
        # self.features = self.features*alpha
        
        return self.features
    
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res      = self.model.classifier(features)

        return res
