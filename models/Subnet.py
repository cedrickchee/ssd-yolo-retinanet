import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# Repeated part can be merged into baseNet... or merged into one class and make 2 different instance
# The structure of two networks are same, but different parameter

def init_conv_weight(layer, std = 0.01, bias = 0):
    nn.init.normal(layer.weight.data, std = std)
    nn.init.constant(layer.bias.data, val = bias)
    return layer

class classification(nn.Module):
    def __init__(self, n_cls, n_anchor):
        super(classification, self).__init__()

        self.n_cls = n_cls
        self.n_anchor = n_anchor

        self.subnet1 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet1 = init_conv_weight(self.subnet1)
        self.subnet2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet2 = init_conv_weight(self.subnet2)
        self.subnet3 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet3 = init_conv_weight(self.subnet3)
        self.subnet4 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet4 = init_conv_weight(self.subnet4)

        self.conf_layer = nn.Conv2d(256, (self.n_cls + 1) * self.n_anchor, kernel_size = 3, padding = 1)
        self._last_layer_init(self.conf_layer.bias.data)

    def forward(self, x):
        
        x = F.relu(self.subnet1(x))
        x = F.relu(self.subnet2(x))
        x = F.relu(self.subnet3(x))
        x = F.relu(self.subnet4(x))

        conf_pred = self.conf_layer(x)
        conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(
                        conf_pred.size(0), -1, self.n_cls + 1)
        # conf_pred: [batch, candidates, n_cls+1]

        # predicted classes from x feature map
        return conf_pred

    def _last_layer_init(self, tensor, pi = 0.01):
        fill_constant = -math.log((1 - pi) / pi)
        if isinstance(tensor, Variable):
            self._last_layer_init(tensor.data)

        return tensor.fill_(fill_constant)

class boxRegression(nn.Module):
    def __init__(self, n_anchor):
        super(boxRegression, self).__init__()

        self.n_anchor = n_anchor

        self.subnet1 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet1 = init_conv_weight(self.subnet1)
        self.subnet2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet2 = init_conv_weight(self.subnet2)
        self.subnet3 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet3 = init_conv_weight(self.subnet3)
        self.subnet4 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.subnet4 = init_conv_weight(self.subnet4)

        self.loc_layer = nn.Conv2d(256, self.n_anchor * 4, kernel_size = 3, padding = 1)

    def forward(self, x):
        
        x = F.relu(self.subnet1(x))
        x = F.relu(self.subnet2(x))
        x = F.relu(self.subnet3(x))
        x = F.relu(self.subnet4(x))

        loc_pred = self.loc_layer(x)
        loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(
                        loc_pred.size(0), -1, 4)
        # loc_pred: [batch, candidates, 4] # relative offset

        return loc_pred

if __name__ == '__main__':
    x1 = Variable(torch.rand(2, 256, 63, 63))
    x2 = Variable(torch.rand(2, 256, 32, 32))
    x3 = Variable(torch.rand(2, 256, 16, 16))
    x4 = Variable(torch.rand(2, 256, 8, 8))
    x5 = Variable(torch.rand(2, 256, 4, 4))

    xs = [x1, x2, x3, x4, x5]

    net1 = classification(8, 9)
    net2 = boxRegression(9)

    for x in xs:
        conf_pred = net1(x)
        loc_pred = net2(x)
        print("conf_pred", conf_pred.size())
        print("loc_pred", loc_pred.size())



