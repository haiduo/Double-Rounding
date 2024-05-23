# Refer to https://arxiv.org/abs/1512.03385
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BN import batchnorm2d_fn
from .opts_dorefa import q_conv2d_dorefa, q_linear_dorefa, q_activate_dorefa, q_identity_dorefa
from .opts_lsq import q_conv2d_lsq, q_linear_lsq, q_activate_lsq, q_identity_lsq
from .opts_lsq_plus import q_conv2d_lsq_plus, q_linear_lsq_plus, q_activate_lsq_plus, q_identity_lsq_plus
import numpy as np

__all__ = ['resnet18q', 'resnet20q', 'resnet50q']


#### Block Pre-activate###
class PreActBasicBlockQ(nn.Module):
    expansion = 1
    def __init__(self, bit_list, in_planes, out_planes, stride=1, w_schema=None, a_schema=None, Trans_BN=False):
        super(PreActBasicBlockQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.w_schema = w_schema
        self.a_schema = a_schema

        if self.w_schema == "dorefa":
            Conv2d = q_conv2d_dorefa(self.bit_list)
        elif self.w_schema == "lsq":
            Conv2d = q_conv2d_lsq(self.bit_list)
        elif self.w_schema == "lsq_plus":
            Conv2d = q_conv2d_lsq_plus(self.bit_list)

        if self.w_schema == "dorefa":
            Activate = q_activate_dorefa(self.bit_list)
        elif self.w_schema == "lsq":
            Activate = q_activate_lsq(self.bit_list)
        elif self.w_schema == "lsq_plus":
            Activate = q_activate_lsq_plus(self.bit_list)

        NormLayer = batchnorm2d_fn(self.bit_list, Trans_BN)

        self.bn0 = NormLayer(in_planes)
        self.act0 = Activate(quantize=True, signed=False)
        self.conv0 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=in_planes, out_planes=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(out_planes)
        self.act1 = Activate(quantize=True, signed=False)
        self.conv1 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=out_planes, out_planes=out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_conv = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.skip_conv = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=in_planes, out_planes=out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            # self.skip_bn = NormLayer(out_planes)

    def forward(self, x):
        out = self.act0(self.bn0(x))
        shortcut = self.skip_conv(out) if self.skip_conv is not None else x
        out1 = self.conv0(out)
        out2 = self.conv1(self.act1(self.bn1(out1)))
        out2 += shortcut
        return out2

#### Bottleneck Pre-activate ###
class PreActBottleneckQ(nn.Module):
    expansion = 4
    def __init__(self, bit_list, in_planes, out_planes, stride=1, w_schema=None, a_schema=None, Trans_BN=False):
        super(PreActBottleneckQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.w_schema = w_schema
        self.a_schema = a_schema

        if self.w_schema == "dorefa":
            Conv2d = q_conv2d_dorefa(self.bit_list)
        elif self.w_schema == "lsq":
            Conv2d = q_conv2d_lsq(self.bit_list)
        elif self.w_schema == "lsq_plus":
            Conv2d = q_conv2d_lsq_plus(self.bit_list)

        if self.w_schema == "dorefa":
            Activate = q_activate_dorefa(self.bit_list)
        elif self.w_schema == "lsq":
            Activate = q_activate_lsq(self.bit_list)
        elif self.w_schema == "lsq_plus":
            Activate = q_activate_lsq_plus(self.bit_list)

        norm_layer = batchnorm2d_fn(self.bit_list, Trans_BN)

        self.bn0 = norm_layer(in_planes)
        self.act0 = Activate(quantize=True, signed=False)
        self.conv0 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=in_planes, out_planes=out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(out_planes)
        self.act1 = Activate(quantize=True, signed=False)
        self.conv1 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=out_planes, out_planes=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.act2 = Activate(quantize=True, signed=False)
        self.conv2 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=out_planes, out_planes=out_planes*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)

        self.skip_conv = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.skip_conv = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=in_planes, out_planes=out_planes*self.expansion, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):     
        out = self.act0(self.bn0(x)) 
        shortcut = self.skip_conv(out) if self.skip_conv is not None else x
        out = self.conv0(out)
        out = self.conv1(self.act1(self.bn1(out)))
        out = self.conv2(self.act2(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet20(nn.Module):
    def __init__(self, block, num_units, bit_list, num_classes, w_schema=None, a_schema=None, Trans_BN=False):
        super(PreActResNet20, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        NormLayer = batchnorm2d_fn(self.bit_list, Trans_BN)

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
        in_planes = 16
        self.layers = nn.ModuleList()
        for stride, channel in zip(strides, channels):
            self.layers.append(block(self.bit_list, in_planes, channel, stride, w_schema=w_schema, a_schema=a_schema, Trans_BN=Trans_BN))
            in_planes = channel

        self.bn = NormLayer(64)
        self.fc = nn.Linear(64, num_classes) 

    def forward(self, x):
        out = self.conv0(x)

        for layer in self.layers:
            out = layer(out)

        out = F.relu(self.bn(out))
        out = out.mean(dim=(2,3))
        out = self.fc(out)
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, layers, bit_list, num_classes, w_schema=None, a_schema=None, Trans_BN=False):
        super(PreActResNet, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.w_schema = w_schema
        self.a_schema = a_schema
        self.Trans_BN = Trans_BN

        self.norm_layer = batchnorm2d_fn(self.bit_list, Trans_BN)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn = self.norm_layer(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)         

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.bit_list, self.inplanes, planes, stride, w_schema=self.w_schema, a_schema=self.a_schema, Trans_BN=self.Trans_BN))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.bit_list, self.inplanes, planes, w_schema=self.w_schema, a_schema=self.a_schema, Trans_BN=self.Trans_BN))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# For CIFAR10
def resnet20q(bit_list, num_classes=10, w_schema=None, a_schema=None, Trans_BN=False):
    return PreActResNet20(PreActBasicBlockQ, [3, 3, 3], bit_list, num_classes=num_classes, w_schema=w_schema, a_schema=a_schema, Trans_BN=Trans_BN)


# For ImageNet
def resnet18q(bit_list, num_classes=1000, w_schema=None, a_schema=None, Trans_BN=False):
    return PreActResNet(PreActBasicBlockQ, [2, 2, 2, 2], bit_list, num_classes=num_classes, w_schema=w_schema, a_schema=a_schema, Trans_BN=Trans_BN)


def resnet50q(bit_list, num_classes=1000, w_schema=None, a_schema=None, Trans_BN=False):
    return PreActResNet(PreActBottleneckQ, [3, 4, 6, 3], bit_list, num_classes=num_classes, w_schema=w_schema, a_schema=a_schema, Trans_BN=Trans_BN)
