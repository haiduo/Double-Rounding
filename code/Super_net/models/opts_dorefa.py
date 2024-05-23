from .quan_dorefa import *

def q_conv2d_dorefa(bit_list):
    class QConv2d_dorefa(nn.Conv2d):
        def __init__(self, quantize=True, signed=True, perchannel=True, in_planes=None, out_planes=None, kernel_size=None, 
                     stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(QConv2d_dorefa, self).__init__(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)

            self.bit_list = bit_list
            self.wbit = self.bit_list[-1]
            self.out_planes = out_planes
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.wbit < 32):
                self.quantize_fn = weight_quantize_fn(self.bit_list, w_ops=True)

        def forward(self, input):
            if self.quantize and self.wbit < 32:
                weight_q = self.quantize_fn(self.weight)
            else:
                weight_q = self.weight
            output =  F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output
    return QConv2d_dorefa


def q_linear_dorefa(bit_list):
    class QLinear_dorefa(nn.Linear):
        def __init__(self, quantize=True, signed=True, perchannel=True, in_planes=None, out_planes=None, bias=True):
            super(QLinear_dorefa, self).__init__(in_planes, out_planes, bias)
            self.bit_list = bit_list
            self.wbit = self.bit_list[-1]
            self.out_planes = out_planes
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.wbit < 32):
                self.quantize_fn = weight_quantize_fn(self.bit_list, w_ops=True)

        def forward(self, input):
            if self.quantize and self.wbit < 32:
                weight_q = self.quantize_fn(self.weight)
            else:
                weight_q = self.weight
            output =  F.linear(input, weight_q, self.bias)
            return output
    return QLinear_dorefa


def q_activate_dorefa(bit_list):
    class QActivate_dorefa(nn.Module):
        def __init__(self, quantize=True, signed=False):
            super(QActivate_dorefa, self).__init__()
            self.bit_list = bit_list
            self.abit = self.bit_list[-1]
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.wbit < 32):
                self.quantize_fn = weight_quantize_fn(self.bit_list)

        def forward(self, x, relu6=False):
            if relu6:
                x = F.relu6(x)
            else:
                x = F.relu(x)

            if self.quantize and self.abit < 32:
                q_x = self.quantize_fn(x)
            else:
                q_x = x
            return q_x
    return QActivate_dorefa


def q_identity_dorefa(bit_list):
    class QIdentity_dorefa(nn.Module):
        def __init__(self, quantize=True, signed=False):
            super(QIdentity_dorefa, self).__init__()
            self.bit_list = bit_list
            self.abit = self.bit_list[-1]
            self.quantize = quantize
            if self.quantize and (len(self.bit_list) > 1 or self.abit < 32):
                self.quantize_fn = weight_quantize_fn(self.bit_list)

        def forward(self, x):
            if self.quantize and self.abit < 32:
                q_x = self.quantize_fn(x)
            else:
                q_x = x
            return q_x
    return QIdentity_dorefa

