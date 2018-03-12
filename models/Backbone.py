import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .ResNet import resnet50

# Build FPN on top of the ResNet architecture

class featurePyramid(nn.Module):
    def __init__(self, pretrained = True):
        super(featurePyramid, self).__init__()
        # Can we remove hard code?

        self.resnet = resnet50(pretrained = pretrained) # which is better?

        self.conv_3_1 = nn.Conv2d(512, 256, kernel_size = 1)
        self.conv_4_1 = nn.Conv2d(1024, 256, kernel_size = 1) 
        self.conv_5_1 = nn.Conv2d(2048, 256, kernel_size = 1)
        self.conv_6 = nn.Conv2d(2048, 256, kernel_size = 3, stride = 2, padding = 1) # p_6, p_7 is additional layer compare to FPN
        self.conv_7 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1)

        self.conv_3_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1) # apply to merged layer
        self.conv_4_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1) 
        self.conv_5_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)

    def forward(self, x):

        _, c_3, c_4, c_5 = self.resnet(x)
        # c_2 is not used for computation reason

        p_3 = self.conv_3_1(c_3)
        p_4 = self.conv_4_1(c_4)

        p_5 = self.conv_5_1(c_5)
        p_6 = self.conv_6(c_5)
        p_7 = self.conv_7(F.relu(p_6))

        # 
        upsampled_p5 = F.upsample(p_5, size = p_4.size()[2:], mode = 'bilinear')
        p_5 = self.conv_5_2(p_5)
        p_4 = p_4 + upsampled_p5
        upsampled_p4 = F.upsample(p_4, size = p_3.size()[2:], mode = 'bilinear')
        p_4 = self.conv_4_2(p_4)
        p_3 = p_3 + upsampled_p4
        p_3 = self.conv_3_2(p_3 + upsampled_p4)

        return p_3, p_4, p_5, p_6, p_7
if __name__ == '__main__':
#TODO: test
    x = torch.rand(2, 3, 300, 300)
    x = Variable(x)

    net = featurePyramid()
    outputs = net(x)

    for output in outputs:
        print(output.size())

