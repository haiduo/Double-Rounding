
import torch.nn as nn

class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, num_features, bit_list):
        super(SwitchBatchNorm2d, self).__init__()
        self.bit_list = bit_list
        self.bn_dict = nn.ModuleDict()
        for i in self.bit_list:
            self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)

        self.abit = self.bit_list[-1]
        self.wbit = self.bit_list[-1]
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x


def batchnorm2d_fn(bit_list):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, bit_list=bit_list):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, bit_list=bit_list)

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
