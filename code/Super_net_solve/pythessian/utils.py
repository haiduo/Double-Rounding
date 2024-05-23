import torch
import math
from torch.autograd import Variable
import numpy as np
import gc


def group_product(xs, ys, whole_model=False):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)]) if whole_model else [torch.sum(x * y).cpu().item() for (x, y) in zip(xs, ys)]


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v, whole_model=True)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model, full_dataset = False, logging=None):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if full_dataset:
            #.......
            if not param.requires_grad or param.grad is None or param.grad.dim()==1:
                continue
        else:
            #.......
            if not param.requires_grad or param.grad is None or param.grad.dim()==1:
                continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
        # logging.info(f"")
    return params, grads


def hessian_vector_product(gradsH, params, v, d_graph):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """

    if not d_graph:
        hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True)
    else:
        hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False)
        for p in params:
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v, whole_model=True))
    return normalization(w)
