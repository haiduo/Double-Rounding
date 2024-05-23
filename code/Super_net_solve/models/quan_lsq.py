import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from .roulette_algorithm import roulette_bit_switching

random_abit = 8
layer_i = 0    

def get_percentile_min_max(input, lower_percentile, upper_percentile):
    size = input.shape[-1]
    lower_index = round(size * lower_percentile)
    upper_index = round(size * upper_percentile)

    upper_bound = torch.kthvalue(input, k=upper_index, dim=1).values  

    if lower_percentile==0:
        low_bound = upper_bound * 0
    else:
        low_bound = -torch.kthvalue(-input, k=lower_index, dim=-1).values 
    
    return low_bound, upper_bound


def grad_scale(scale, g):
    y = scale
    y_grad = scale*g
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Alsq_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, g, Qn, Qp):

        ctx.save_for_backward(input, scale)
        ctx.other = g, Qn, Qp
        
        q_res = torch.clamp(torch.div(input, scale).round(), Qn, Qp)
        dq_res = q_res * scale
        return dq_res

    @staticmethod
    def backward(ctx, grad_input):
        input, scale = ctx.saved_tensors
        g, Qn, Qp = ctx.other

        q_res = torch.div(input, scale)

        indicate_small = (q_res < Qn).float()
        indicate_big = (q_res > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        grad_scale = indicate_small*Qn + indicate_big*Qp + indicate_middle*(q_res.round() - q_res)
        grad_scale = torch.sum(grad_scale*grad_input*g).expand(scale.size(dim=0))

        grad_input = indicate_middle * grad_input
        return grad_input, grad_scale, None, None, None


class activation_quantize_fn(nn.Module):
    def __init__(self, bit_list, signed=False):
        super(activation_quantize_fn, self).__init__()
        self.bit_list = bit_list
        self.abit = self.bit_list[-1]
        assert self.abit <= 8

        self.signed = signed   

        self.scale_dict = nn.ParameterDict()
        for i in self.bit_list:
            if torch.cuda.is_available():
                self.scale_dict[str(i)] = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
            else:
                self.scale_dict[str(i)] = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.register_buffer('init_state', torch.tensor(0))

    def forward(self, activation):
        global random_abit, layer_i
        if layer_i == len(self.cur_bits)-1:
            layer_i = 0  
        
        if self.mix_infer:
            self.abit = self.cur_bits[layer_i]
            random_abit = self.abit
        else:
            r = np.random.uniform(0, 1)
            # r = 0
            if self.sigma > r:
                random_abit = roulette_bit_switching(b=self.bit_list, t_l=self.HMT[layer_i], t_m=self.Avg_HMT)
                self.abit = random_abit
            else:
                random_abit = self.abit

        if self.signed:
            if self.abit != 1:
                # Qn, Qp = (-1* ((1 << self.abit-1)-1), (1 << self.abit-1)-1 )  #-127 -- 127 
                Qn, Qp = (-1* (1 << self.abit-1), (1 << self.abit-1)-1 )  #-128 -- 127
            else:
                Qn, Qp = (-1, 1)
        else:
            Qn, Qp = (0, (1 << self.abit)-1) # 0 -- 255

        g = 1.0 / math.sqrt(activation.numel() * Qp)

        if self.training and self.init_state < len(self.bit_list):
            self.scale_dict[str(self.abit)].data.copy_(2 * activation.detach().abs().mean() / math.sqrt(Qp))
            # print()
            self.init_state += 1
        
        q_a = Alsq_STE.apply(activation, self.scale_dict[str(self.abit)], g, Qn, Qp)

        # # Check activation quantization error
        # if self.sigma > 0 and self.init_mix_batch>0:
        #     mse = torch.sum((q_a.detach()-activation.detach())**2)
        #     self.logging.info(f"activation_layer_{list(activation.shape)}_mse:{mse:.4f}")

        return q_a


class Wlsq_STE(torch.autograd.Function):
    switch_w = True
    save_float = False
    double_round = True
    @staticmethod
    def forward(ctx, input, scale, g, Qn, Qp, i_bit, bit_list):
        # assert scale>0, "scale={}".format(scale)
        if i_bit < bit_list[-1]:
            diff_bit = bit_list[-1] - i_bit
            if not Wlsq_STE.switch_w:
                scale = scale * (2**diff_bit)  #scale << diff_bit
        else:
            diff_bit = 0

        ctx.save_for_backward(input, scale)
        ctx.other = g, Qn, Qp, diff_bit

        if Wlsq_STE.switch_w:
            q_res_before = torch.div(input, scale.reshape([scale.size(dim=0),1,1,1])).round()
            if Wlsq_STE.double_round:
                q_res_after = torch.clamp((q_res_before / (1 << diff_bit)).round(), Qn, Qp) #double rounding
            else:
                q_res_after = torch.clamp(q_res_before.round().to(torch.int) >> diff_bit, Qn, Qp)
            dq_res = q_res_after * scale.reshape([scale.size(dim=0),1,1,1]) * (1 << diff_bit)
        else:                 
            if Wlsq_STE.save_float:
                q_res = torch.clamp(torch.div(input, scale.reshape([scale.size(dim=0),1,1,1])).round(), Qn, Qp)   
            else:
                q_res = torch.clamp(torch.div(input, scale.reshape([scale.size(dim=0),1,1,1])).floor(), Qn, Qp)   
            dq_res = q_res * scale.reshape([scale.size(dim=0),1,1,1])
        return dq_res

    @staticmethod
    def backward(ctx, grad_input):
        input, scale = ctx.saved_tensors
        g, Qn, Qp, diff_bit = ctx.other

        if Wlsq_STE.switch_w:
            q_res = torch.div(input, scale.reshape([scale.size(dim=0),1,1,1]) * (1 << diff_bit)) 
        else:
            q_res = torch.div(input, scale.reshape([scale.size(dim=0),1,1,1]))  

        indicate_small = (q_res < Qn).float()
        indicate_big = (q_res > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        grad_scale = indicate_small*Qn + indicate_big*Qp + indicate_middle*(q_res.round() - q_res)
        
        grad_scale = torch.sum(grad_scale*grad_input*g).expand(scale.size(dim=0)) 
        grad_input = indicate_middle * grad_input

        return grad_input, grad_scale, None, None, None, None, None


class weight_quantize_fn(nn.Module):
    def __init__(self, bit_list, signed=True, out_channels=None, is_perchannel=True):
        super(weight_quantize_fn, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        assert self.wbit <= 8

        self.signed = signed    
        self.out_channels = out_channels 
        self.is_perchannel = is_perchannel

        if self.is_perchannel:
            if torch.cuda.is_available():
                self.scale = nn.Parameter(torch.FloatTensor(self.out_channels).cuda(), requires_grad=True)
            else:
                self.scale = nn.Parameter(torch.FloatTensor(self.out_channels), requires_grad=True)
        else:
            if torch.cuda.is_available():
                self.scale = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
            else:
                self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.register_buffer('init_state', torch.tensor(0))

    def forward(self, weight):
        global random_abit, layer_i
        if weight.shape[-1]!=1 and weight.shape[-2]!=1:
            layer_i += 1
            # print(f"weight shape: {weight.shape}")
        self.wbit = random_abit

        if self.signed:
            if self.wbit != 1:
                # Qn, Qp = (-1* ((1 << self.wbit-1)-1), (1 << self.wbit-1)-1 )  #-127 -- 127 
                Qn, Qp = (-1* (1 << self.wbit-1), (1 << self.wbit-1)-1 )  #-128 -- 127 
            else:
                Qn, Qp = (-1, 1) 
        else:
            Qn, Qp = (0, (1 << self.wbit)-1) # 0 -- 255

        if self.is_perchannel:
            num_perchannel = weight.numel() / weight.shape[0]
            g = 1.0 / math.sqrt(num_perchannel * Qp)  
        else:
            g = 1.0 / math.sqrt(weight.numel() * Qp)

        if self.training and self.init_state == 0:
            if self.is_perchannel:
                div = 2**(self.wbit-1)                                                                   
                mean = weight.detach().mean(axis=(1,2,3))
                std = weight.detach().std(axis=(1,2,3))
                clip_range = torch.max(torch.abs(mean-3*std), torch.abs(mean+3*std))
                self.scale.data.copy_(clip_range/div)
            else: 
                self.scale.data.copy_(2 * weight.detach().abs().mean() / math.sqrt(Qp)) 
               
            self.init_state.fill_(1)
        
        detach_STE = False
        if detach_STE:
            if self.wbit < 8:
                diff_bit = 8-self.wbit
                scale = self.scale * (2*diff_bit)
            else:
                scale = self.scale
            scale = grad_scale(scale, g) 
            q_w = round_pass((torch.div(weight, scale.reshape([scale.size(dim=0),1,1,1]))).clamp(Qn,Qp)) * (scale.reshape([scale.size(dim=0),1,1,1]))
        else:
            q_w = Wlsq_STE.apply(weight, self.scale, g, Qn, Qp, self.wbit, self.bit_list)

        # # Check weight quantization error
        # if self.sigma > 0 and self.init_mix_batch>0:
        #     mse = torch.sum((q_w.detach()-weight.detach())**2)
        #     self.logging.info(f"weight_layer_{list(weight.shape)}_mse:{mse:.4f}")
        
        return q_w







    