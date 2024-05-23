from .quan_lsq_plus import *


def q_conv2d_lsq_plus(bit_list):
    class QConv2d_lsq_plus(nn.Conv2d):
        def __init__(self, quantize=True, signed=True, perchannel=True, in_planes=None, out_planes=None, kernel_size=None, stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(QConv2d_lsq_plus, self).__init__(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)

            self.bit_list = bit_list 
            self.wbit = self.bit_list[-1]
            self.signed = signed
            self.out_planes = out_planes
            self.is_perchannel = perchannel
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.wbit < 32):
                self.quantize_fn = LSQPlusWeightQuantizer(self.bit_list, self.signed, self.out_planes, self.is_perchannel)

        def forward(self, input):
            if self.quantize and self.wbit < 32:
                weight_q = self.quantize_fn(self.weight)
            else:
                weight_q = self.weight
            output =  F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output
    return QConv2d_lsq_plus


def q_linear_lsq_plus(bit_list):
    class QLinear_lsq_plus(nn.Linear):
        def __init__(self, quantize=True, signed=True, perchannel=True, in_planes=None, out_planes=None, bias=True):
            super(QLinear_lsq_plus, self).__init__(in_planes, out_planes, bias)
            self.bit_list = bit_list
            self.wbit = self.bit_list[-1]
            self.signed = signed
            self.out_planes = out_planes
            self.is_perchannel = perchannel
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.wbit < 32):
                self.quantize_fn = LSQPlusWeightQuantizer(self.bit_list, self.signed, self.out_planes, self.is_perchannel)

        def forward(self, input):
            if self.quantize and self.wbit < 32:
                weight_q = self.quantize_fn(self.weight)
            else:
                weight_q = self.weight
            output =  F.linear(input, weight_q, self.bias)
            return output
    return QLinear_lsq_plus


def q_activate_lsq_plus(bit_list):
    class QActivate_lsq_plus(nn.Module):
        def __init__(self, quantize=True, signed=False):
            super(QActivate_lsq_plus, self).__init__()
            self.bit_list = bit_list
            self.abit = self.bit_list[-1]
            self.signed = signed
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.abit < 32):
                self.quantize_fn = LSQPlusActivationQuantizer(self.bit_list, self.signed)

        def forward(self, x, relu6=False):
            if relu6:
                x = F.relu6(x)
            else:
                x = F.relu(x)

            if self.quantize and self.abit !=32:
                q_x = self.quantize_fn(x)
            else:
                q_x = x
            return q_x
    return QActivate_lsq_plus


def q_identity_lsq_plus(bit_list):
    class QIdentity_lsq_plus(nn.Module):
        def __init__(self, quantize=True, signed=False):
            super(QIdentity_lsq_plus, self).__init__()
            self.bit_list = bit_list
            self.abit = self.bit_list[-1]
            self.signed = signed
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.abit < 32):
                self.quantize_fn = LSQPlusActivationQuantizer(self.bit_list, self.signed)

        def forward(self, x):
            if self.quantize and self.abit !=32:
                q_x = self.quantize_fn(x)
            else:
                q_x = x
            return q_x
    return QIdentity_lsq_plus

