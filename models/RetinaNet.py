import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from .Backbone import featurePyramid
from .Subnet import classification, boxRegression

class retinaNet(nn.Module):
    def __init__(self, n_cls, n_anchor, resume):
        super(retinaNet, self).__init__()
    
        self.feature_pyramid = featurePyramid(False if resume else True)
        self.conf_layer = classification(n_cls, n_anchor)
        self.loc_layer = boxRegression(n_anchor)

    def forward(self, x):
        
        features = self.feature_pyramid(x)

        conf_preds = []
        loc_preds = []

        for feature in features:
            conf_preds.append(self.conf_layer(feature))
            loc_preds.append(self.loc_layer(feature))

        conf_preds = torch.cat(conf_preds, 1)
        loc_preds = torch.cat(loc_preds, 1)

        return (conf_preds, loc_preds)

    def freeze_bn(self):
        """
            following this issue: https://github.com/kuangliu/pytorch-retinanet/issues/18
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

if __name__ == '__main__':
    x = torch.rand(2, 3, 800, 800)
    x = Variable(x)

    net = retinaNet(8, 9)
    conf_preds, loc_preds = net(x)
    print(conf_preds.size())
    print(loc_preds.size())
