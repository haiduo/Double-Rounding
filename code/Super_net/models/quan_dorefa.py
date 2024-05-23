# Adapted from
# https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py and
# https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/dorefa.py#L25

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, bit_list, w_ops=False):
        super(weight_quantize_fn, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.w_ops = w_ops
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        E = torch.mean(torch.abs(x)).detach()
        weight = torch.tanh(x)
        weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
        weight_q = 2 * qfn.apply(weight, self.wbit) - 1
        weight_q = weight_q * E

        rescale = False
        if rescale and self.w_ops: 
        # if rescale:
            '''Afterwards the quantized value will be scaled back by dequantization for input or weight if rescale is True.'''
            rescale_type = rescale
            if rescale_type == 'stddev':
                weight_q_scale = torch.std(weight.detach()) / torch.std(weight_q.detach())
                if len(weight.shape) == 4:
                    weight_q.mul_(weight_q_scale)
                elif len(weight.shape) == 2 and self.training:
                    weight_q.mul_(weight_q_scale)
            elif rescale_type == 'constant':
                if len(weight.shape) == 4:
                    out_channels, in_channels, kernel_w, kernel_h = weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]
                    weight_q_scale = (1.0 / (out_channels * kernel_w * kernel_h) ** 0.5) / torch.std(weight_q.detach())
                    weight_q.mul_(weight_q_scale)
                elif len(weight.shape) == 2:
                    out_channels, in_channels= weight.shape[0], weight.shape[1]
                    weight_q_scale = (1.0 / (out_channels) ** 0.5) / torch.std(weight_q.detach())
                    if self.training:
                        weight_q.mul_(weight_q_scale)

        return weight_q
