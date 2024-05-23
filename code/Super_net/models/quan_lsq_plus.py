# Adapted from
# https://github.com/ZouJiu1/LSQplus

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output
    
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ALSQPlus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, g, Qn, Qp, beta):
        # assert scale > 0, "scale={}".format(scale)
        ctx.save_for_backward(input, scale, beta)
        ctx.other = g, Qn, Qp
        i_q = Round.apply(torch.div((input - beta), scale).clamp(Qn, Qp))
        i_q = i_q * scale + beta
        return i_q

    @staticmethod
    def backward(ctx, grad_weight):
        input, scale, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_i = (input - beta) / scale
        smaller = (q_i < Qn).float()
        bigger = (q_i > Qp).float()
        between = 1.0 - smaller -bigger
        grad_scale = ((smaller * Qn + bigger * Qp + between * Round.apply(q_i) - between * q_i)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        
        grad_weight = between * grad_weight
        
        return grad_weight, grad_scale,  None, None, None, grad_beta


def quantization_error(input, scale, beta, g, Qn, Qp):
    q_input = ALSQPlus.apply(input, scale, g, Qn, Qp, beta)
    return torch.sum((q_input-input)**2)


class Quantization_error(nn.Module):
    def __init__(self, scale, g, Qn, Qp, beta):
        super().__init__()
        if torch.cuda.is_available():
            self.scale = nn.Parameter(torch.tensor(scale).cuda(), requires_grad=True)
            self.beta = nn.Parameter(torch.tensor(beta).cuda(), requires_grad=True)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
            self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)
        self.g = g
        self.Qn = Qn
        self.Qp = Qp
    def forward(self, input):
        return quantization_error(input, self.scale, self.beta, self.g, self.Qn, self.Qp)


def MSE_error(input, scale, g, Qn, Qp, beta):
    q_loss=0    
    min_error = Quantization_error(scale, g, Qn, Qp, beta)
    if torch.cuda.is_available():
        min_error = min_error.cuda()
        input = input.cuda()
    optimizer = torch.optim.Adam(min_error.parameters(), lr=0.0001)
    step = 500
    for i in range(step):
        mse_loss=min_error(input)
        optimizer.zero_grad()
        mse_loss.backward(retain_graph=True)
        optimizer.step()
        q_loss += mse_loss.item()
        if i%50 == 0:
            print(f"{i+1} step mse_loss:{mse_loss.item():.3f}, ave_loss:{q_loss/50:.3f}, scale:{min_error.scale.item():.3f}, beta:{min_error.beta.item():.3f}")
            q_loss=0
    return min_error.scale, min_error.beta


class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, bit_list, signed=False, batch_init=0, init_minmax=True):
        super(LSQPlusActivationQuantizer, self).__init__()
        self.bit_list = bit_list
        self.abit = self.bit_list[-1]
        assert self.abit <= 8 

        self.signed = signed          
        self.batch_init = batch_init   
        self.init_minmax = init_minmax 

        self.scale_dict = nn.ParameterDict()
        self.beta_dict = nn.ParameterDict()
        for i in self.bit_list:
            if torch.cuda.is_available():
                self.scale_dict[str(i)] = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
                self.beta_dict[str(i)] = nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
            else:
                self.scale_dict[str(i)] = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.beta_dict[str(i)] = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.register_buffer('init_state', torch.tensor(0))

    
    def forward(self, activation):
        if self.signed:
            if self.abit != 1:
                Qn, Qp = (-1* (1 << self.abit-1), (1 << self.abit-1)-1 )
            else:
                Qn, Qp = (-1, 1)
        else:
            Qn, Qp = (0, (1 << self.abit)-1)

        g = 1.0 / math.sqrt(activation.numel() * Qp)

        if self.training:
            if self.init_minmax:
                if self.init_state < len(self.bit_list):
                    mina = torch.min(activation.detach())
                    self.scale_dict[str(self.abit)].data.copy_((torch.max(activation.detach()) - mina)/(Qp-Qn))
                    self.beta_dict[str(self.abit)].data.copy_(mina - self.scale_dict[str(self.abit)].data*Qn)
                    self.init_state += 1
                    # print(f"{self.abit}bit_scale: {self.scale_dict[str(self.abit)].data}")
                    # print(f"{self.abit}bit_beta: {self.beta_dict[str(self.abit)].data}")
                elif self.batch_init>0 and self.init_state < len(self.bit_list)*self.batch_init:
                    alpha_s, alpha_beta = 0.9, 0.9
                    mina = torch.min(activation.detach())
                    self.scale_dict[str(self.abit)].data.copy_(self.scale_dict[str(self.abit)].data*alpha_s +
                                                               (1-alpha_s)*(torch.max(activation.detach()) - mina)/(Qp-Qn))
                    self.beta_dict[str(self.abit)].data.copy_(self.scale_dict[str(self.abit)].data*alpha_beta +
                                                               (1-alpha_beta)*(mina - self.scale_dict[str(self.abit)].data * Qn))
                    self.init_state += 1
            else:
                if self.init_state < len(self.bit_list):
                    mina = torch.min(activation.detach())
                    self.scale_dict[str(self.abit)].data.copy_((torch.max(activation.detach()) - mina)/(Qp-Qn))
                    self.beta_dict[str(self.abit)].data.copy_(mina - self.scale_dict[str(self.abit)].data*Qn)
                    self.scale_dict[str(self.abit)].data, self.beta_dict[str(self.abit)].data = MSE_error(\
                        activation.detach(), self.scale_dict[str(self.abit)].data, g, Qn, Qp, self.beta_dict[str(self.abit)].data)
                    self.init_state += 1
                    print(f"{self.abit}bit_scale: {self.scale_dict[str(self.abit)].data}")
                    print(f"{self.abit}bit_beta: {self.beta_dict[str(self.abit)].data}")
                elif self.batch_init>0 and self.init_state < len(self.bit_list)*self.batch_init:
                    self.scale_dict[str(self.abit)].data, self.beta_dict[str(self.abit)].data = MSE_error(\
                        activation.detach(), self.scale_dict[str(self.abit)].data, g, Qn, Qp, self.beta_dict[str(self.abit)].data)
                    self.init_state += 1

        q_a = ALSQPlus.apply(activation, self.scale_dict[str(self.abit)], g, Qn, Qp, self.beta_dict[str(self.abit)].data)
        return q_a


class WLSQPlus(torch.autograd.Function):
    switch_w = True

    @staticmethod
    def forward(ctx, input, scale, g, Qn, Qp, i_bit, bit_list):
        # assert scale>0, "scale={}".format(scale)
        if i_bit < bit_list[-1]:
            diff_bit = bit_list[-1] - i_bit
            if not WLSQPlus.switch_w:
                scale = scale << diff_bit
        else:
            diff_bit = 0

        ctx.save_for_backward(input, scale)
        ctx.other = g, Qn, Qp, diff_bit

        if WLSQPlus.switch_w: 
            q_res_before = torch.div(input, scale.reshape([scale.size(dim=0),1,1,1])).round()
            q_res_after = torch.clamp(torch.round(q_res_before >> diff_bit), Qn, Qp)
            dq_res = q_res_after * scale.reshape([scale.size(dim=0),1,1,1]) * (1 << diff_bit)
        else:                 
            q_res = torch.clamp(torch.div(input, scale.reshape([scale.size(dim=0),1,1,1])).floor(), Qn, Qp)
            dq_res = q_res * scale.reshape([scale.size(dim=0),1,1,1])
            
        return dq_res

    @staticmethod
    def backward(ctx, grad_input):
        input, scale = ctx.saved_tensors
        g, Qn, Qp, diff_bit = ctx.other

        if WLSQPlus.switch_w:
            q_res = torch.div(input, scale.reshape([scale.size(dim=0),1,1,1]) * (1 << diff_bit))
        else:
            q_res = torch.div(input, scale.reshape([scale.size(dim=0),1,1,1]))

        indicate_small = (q_res < Qn).float()
        indicate_big = (q_res > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        if WLSQPlus.switch_w:
            grad_scale = indicate_small*Qn + indicate_middle*Qp + indicate_middle*(q_res.round() - q_res)
        else:
            grad_scale = indicate_small*Qn + indicate_middle*Qp + indicate_middle*(q_res.floor() - q_res)

        grad_scale = torch.sum(grad_scale*grad_input*g).expand(scale.size(dim=0))
        grad_input = indicate_middle * grad_input

        return grad_input, grad_scale, None, None, None, None, None


class LSQPlusWeightQuantizer(nn.Module):
    def __init__(self, bit_list, signed=True, out_channels=None, is_perchannel=True):
        super(LSQPlusWeightQuantizer, self).__init__()
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
        if self.signed:
            if self.wbit != 1:
                # Qn, Qp = (-1* ((1 << self.wbit-1)-1), (1 << self.wbit-1)-1 )
                Qn, Qp = (-1* (1 << self.wbit-1), (1 << self.wbit-1)-1 ) 
            else:
                Qn, Qp = (-1, 1)
        else:
            Qn, Qp = (0, (1 << self.wbit)-1)

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
                div = 2**(self.wbit-1)  
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.scale.data = max([torch.abs(mean-3*std), torch.abs(mean+3*std)]) / div
            self.init_state += 1
        
        q_w = WLSQPlus.apply(weight, self.scale, g, Qn, Qp, self.wbit, self.bit_list)
        return q_w


    