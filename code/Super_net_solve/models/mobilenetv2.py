# Refer to https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .BN import batchnorm2d_fn
from .opts_dorefa import q_conv2d_dorefa, q_linear_dorefa, q_activate_dorefa, q_identity_dorefa
from .opts_lsq import q_conv2d_lsq, q_linear_lsq, q_activate_lsq, q_identity_lsq
from .opts_lsq_plus import q_conv2d_lsq_plus, q_linear_lsq_plus, q_activate_lsq_plus, q_identity_lsq_plus
import numpy as np

__all__ = ['mobilenetv2q',]

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, bit_list=None, w_schema=None, a_schema=None, Trans_BN=False):
        super(InvertedResidual, self).__init__()
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
            Identity = q_identity_dorefa(self.bit_list)
        elif self.w_schema == "lsq":
            Activate = q_activate_lsq(self.bit_list)
            Identity = q_identity_lsq(self.bit_list)
        elif self.w_schema == "lsq_plus":
            Activate = q_activate_lsq_plus(self.bit_list)
            Identity = q_identity_lsq_plus(self.bit_list)

        norm_layer = batchnorm2d_fn(self.bit_list, Trans_BN)

        assert stride in [1, 2]
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        # pw
        self.conv1 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=inp, out_planes=hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(hidden_dim)
        self.act1 = Activate(quantize=True, signed=False)
        # dw
        self.conv2 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=hidden_dim, out_planes=hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = norm_layer(hidden_dim)
        self.act2 = Activate(quantize=True, signed=False)
        # pw-linear
        self.conv3 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=hidden_dim, out_planes=oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(oup)
        self.identity3 = Identity(quantize=True, signed=False)


    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)), relu6=True)
        out = self.act2(self.bn2(self.conv2(out)), relu6=True)
        out = self.identity3(self.bn3(self.conv3(out)))
        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(self, bit_list, width_mult=1, num_classes=1000, w_schema=None, a_schema=None, Trans_BN=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s => # (expansion, out_planes, num_blocks, stride)
            # [1, 16, 1, 1],
            [6, 24, 2, 1] if num_classes==10 else (6, 24, 2, 2), # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],]

        self.num_classes = num_classes
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
            Identity = q_identity_dorefa(self.bit_list)
        elif self.w_schema == "lsq":
            Activate = q_activate_lsq(self.bit_list)
            Identity = q_identity_lsq(self.bit_list)
        elif self.w_schema == "lsq_plus":
            Activate = q_activate_lsq_plus(self.bit_list)
            Identity = q_identity_lsq_plus(self.bit_list)

        norm_layer = batchnorm2d_fn(self.bit_list, Trans_BN)

        # building first layer
        if self.num_classes == 10:
            # NOTE: change stride 2 -> 1 for CIFAR10
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False) 
        self.bn1 = norm_layer(32)
        self.act1 = Activate(quantize=True, signed=False)

        # building first inverted residual blocks => (1, 16, 1, 1)
        self.conv2 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.bn2 = norm_layer(32)
        self.act2 = Activate(quantize=True, signed=False)
        self.conv3 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=32, out_planes=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(16)
        self.identity3 = Identity(quantize=True, signed=False)

        input_channel = 16
        layers = []
        # building remain inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    layers.append(block(input_channel, output_channel, s, expand_ratio=t, bit_list=bit_list, w_schema=self.w_schema, a_schema=a_schema, Trans_BN=Trans_BN))
                else:
                    layers.append(block(input_channel, output_channel, 1, expand_ratio=t, bit_list=bit_list, w_schema=self.w_schema, a_schema=a_schema, Trans_BN=Trans_BN))
                input_channel = output_channel
        
        self.layers = nn.Sequential(*layers)

        # building classifier
        self.conv4 = Conv2d(quantize=True, signed=True, perchannel=True, in_planes=320, out_planes=1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = norm_layer(1280)
        self.linear = nn.Linear(1280, num_classes) 

        self._initialize_weights()


    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(self.bn1(out), relu6=True)

        # (1, 16, 1, 1)
        out = self.act2(self.bn2(self.conv2(out)), relu6=True)  #dw
        out = self.identity3(self.bn3(self.conv3(out)))         #pw-linear

        out = self.layers(out) # t=6
 
        out = F.relu6(self.bn4(self.conv4(out))) 
        if self.num_classes == 10:
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            out = F.avg_pool2d(out, 4)
        else:
            out = F.avg_pool2d(out, 7)

        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2q(bit_list, num_classes=1000, w_schema="lsq", a_schema="lsq", Trans_BN=False):
    model = MobileNetV2(bit_list, width_mult=1, num_classes=num_classes, w_schema=w_schema, a_schema=a_schema, Trans_BN=Trans_BN)
    return model

if __name__ == '__main__':
    net = mobilenetv2q(num_classes=10)
    x = torch.randn(2,3,32,32)   # cifar10
    y = net(x)
    print("cifar10:", y.size())
    net = mobilenetv2q(num_classes=1000)
    x = torch.randn(2,3,224,224) # imagent
    y = net(x)
    print("imagent:", y.size())