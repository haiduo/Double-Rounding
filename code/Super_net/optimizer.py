import torch

def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias) 
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            if getattr(m, 'is_perchannel', False):
                params_no_decay.extend([*m.parameters()])
            else:
                params_no_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay


def print_parameters_info(parameters):
    for k, param in enumerate(parameters):
        print('[{}/{}] {}'.format(k+1, len(parameters), param.shape))


def get_optimizer_config(model, name, lr, weight_decay, sgd_momentum, sgd_dampening, adam_beta1, adam_beta2, scale_no_wd=False, double_optimizer=False):
    if double_optimizer:
        params_decay, params_no_decay = split_parameters(model)
        if scale_no_wd: # scale no weight_decay in multi_lr update
            if name == 'sgd':
                optimizer1 = torch.optim.SGD(params_no_decay, lr=lr, momentum=sgd_momentum, dampening=sgd_dampening, weight_decay=0)
                optimizer2 = torch.optim.SGD(params_decay, lr=lr, momentum=sgd_momentum, dampening=sgd_dampening, weight_decay=weight_decay)
            elif name == 'adam':
                optimizer1 = torch.optim.Adam(params_no_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=0)
                optimizer2 = torch.optim.Adam(params_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
            elif name == 'adamw':
                optimizer1 = torch.optim.AdamW(params_no_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=0)
                optimizer2 = torch.optim.AdamW(params_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
            elif name == 'sgd_adam':
                optimizer1 = torch.optim.SGD(params_no_decay, lr=lr, momentum=sgd_momentum, dampening=sgd_dampening, weight_decay=0)
                optimizer2 = torch.optim.Adam(params_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
            else: #'adamW-adam'
                optimizer1 = torch.optim.AdamW(params_no_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=0)
                optimizer2 = torch.optim.Adam(params_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
        else:
            if name == 'sgd':
                optimizer1 = torch.optim.SGD(params_no_decay, lr=lr, momentum=sgd_momentum, dampening=sgd_dampening, weight_decay=weight_decay) 
                optimizer2 = torch.optim.SGD(params_decay, lr=lr, momentum=sgd_momentum, dampening=sgd_dampening, weight_decay=weight_decay)
            elif name == 'adam':
                optimizer1 = torch.optim.Adam(params_no_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
                optimizer2 = torch.optim.Adam(params_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
            elif name == 'adamw':
                optimizer1 = torch.optim.AdamW(params_no_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
                optimizer2 = torch.optim.AdamW(params_decay, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    elif scale_no_wd:
        params_decay, params_no_decay = split_parameters(model)
        params_list = [{'params': params_decay,}, {'params': params_no_decay, 'weight_decay': 0,},]
        if name == 'sgd':
            optimizer = torch.optim.SGD(params_list, lr=lr, momentum=sgd_momentum, dampening=sgd_dampening, weight_decay=weight_decay) 
        elif name == 'adam':
            optimizer = torch.optim.Adam(params_list, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
        elif name == 'adamw':
            optimizer = torch.optim.AdamW(params_list, lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    else:
        if name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=sgd_momentum, dampening=sgd_dampening, weight_decay=weight_decay) 
        elif name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
        elif name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    
    if double_optimizer:
        return (optimizer1, optimizer2)
    else:
        return optimizer


def get_lr_scheduler(name, optimizer, lr_decay, epochs, last_epoch=-1, double_optimizer=False):
    if double_optimizer:
        if name == 'ConstantLR':
            lr_scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer[0], factor=0.5, total_iters=2)
            lr_scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer[1], factor=0.5, total_iters=2)
        elif name == 'MultiStepLR':
            lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer[0], lr_decay, gamma=0.1, last_epoch=last_epoch)
            lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer[1], lr_decay, gamma=0.1, last_epoch=last_epoch)
        elif name == 'StepLR':
            lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=30, gamma=0.1, last_epoch=last_epoch)
            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer[1], step_size=30, gamma=0.1, last_epoch=last_epoch)
        elif name == 'CosineAnnealingLR':
            lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], epochs, last_epoch=last_epoch)
            lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], epochs, last_epoch=last_epoch)
        elif name == 'ReduceLROnPlateau':
            lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], 'min')
            lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[1], 'min')
        return lr_scheduler1, lr_scheduler2
    else:
        if name == 'ConstantLR':
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=2)
        elif name == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay, gamma=0.1, last_epoch=last_epoch)
        elif name == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=last_epoch)
        elif name == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, last_epoch=last_epoch)
        elif name == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    return lr_scheduler