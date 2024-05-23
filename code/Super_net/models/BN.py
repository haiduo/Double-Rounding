import torch.nn as nn
from itertools import product

old_bit = 8
first_bn = True

class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, num_features, bit_list, Trans_BN=False):
        super(SwitchBatchNorm2d, self).__init__()
        global first_bn
        self.Trans_BN = Trans_BN
        self.first_bn_ = False
        if self.Trans_BN:
            if first_bn:
                self.first_bn_ = True
                self.bit_list = bit_list
                first_bn = False
            else:
                self.bit_list = list(product(bit_list, repeat=2))
        else:
            self.bit_list = bit_list

        self.bn_dict = nn.ModuleDict()
        for i in self.bit_list:
            self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)

        self.abit = self.bit_list[-1]
        self.wbit = self.bit_list[-1]
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        if self.sigma > 0:
            global old_bit
            global random_abit
        from .quan_lsq import random_abit
        # from .quan_lsq_plus import random_abit  # For MobileV2 series
        if self.Trans_BN:
            if self.first_bn_:
                x = self.bn_dict[str(self.abit)](x)
                if self.sigma > 0:
                    old_bit = self.abit
            else:
                if self.sigma > 0:
                    x = self.bn_dict[str((old_bit, random_abit))](x)
                    old_bit = random_abit
                else:
                    x = self.bn_dict[str((self.abit, self.abit))](x)
        else:
            x = self.bn_dict[str(self.abit)](x)
        return x


def batchnorm2d_fn(bit_list, Trans_BN):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, bit_list=bit_list, Trans_BN=Trans_BN):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, bit_list=bit_list, Trans_BN=Trans_BN)

    return SwitchBatchNorm2d_


class SwitchBatchNorm1d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, num_features, bit_list):
        super(SwitchBatchNorm1d, self).__init__()
        self.bit_list = bit_list
        self.bn_dict = nn.ModuleDict()
        for i in self.bit_list:
            self.bn_dict[str(i)] = nn.BatchNorm1d(num_features)

        self.abit = self.bit_list[-1]
        self.wbit = self.bit_list[-1]
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x


def batchnorm1d_fn(bit_list):
    class SwitchBatchNorm1d_(SwitchBatchNorm1d):
        def __init__(self, num_features, bit_list=bit_list):
            super(SwitchBatchNorm1d_, self).__init__(num_features=num_features, bit_list=bit_list)

    return SwitchBatchNorm1d_

