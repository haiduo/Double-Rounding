import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .custom_datasets import *

import math
from typing import List, Optional, Union
import torch
from torch.utils.data.sampler import Sampler

data_means = {
    'cifar10': [x / 255 for x in [125.3, 123.0, 113.9]],
    'cifar100': [x / 255 for x in [129.3, 124.1, 112.4]],
    'mnist': [0.1307],
    'imagenet': [0.485, 0.456, 0.406]
}

data_stds = {
    'cifar10': [x / 255 for x in [63.0, 62.1, 66.7]],
    'cifar100': [x / 255 for x in [68.2, 65.4, 70.4]],
    'mnist': [0.3081],
    'imagenet': [0.229, 0.224, 0.225]
}

def gen_paths(data_url = None):
    # data_root =os.path.dirname(os.path.realpath(__file__)) + '/data'
    data_root = os.path.abspath('./') + '/data'
    if data_url:
        data_root =data_url

    data_paths = {
        'cifar10': os.path.join(data_root, 'cifar10'),
        'cifar100': os.path.join(data_root, 'cifar100'),
        'mnist': os.path.join(data_root, 'mnist'),
        'imagenet': {
            'train': os.path.join(data_root, 'imagenet/train'),
            'val': os.path.join(data_root, 'imagenet/val')
        }
    }
    return data_paths


def get_dataset(name, split='train', transform=None, target_transform=None, download=False, data_paths=None, PSKD=False):
    train = (split == 'train')
    if name == 'cifar10':
        if PSKD:
            dataset = Custom_CIFAR10(root=data_paths['cifar10'], 
                                                      train=train, 
                                                      download=download, 
                                                      transform=transform)
        else:
            dataset = datasets.CIFAR10(root=data_paths['cifar10'],
                            train=train,
                            transform=transform,               
                            target_transform=target_transform,
                            download=download)
        dataset.num_classes = 10
    elif name == 'cifar100':
        if PSKD:
            dataset = Custom_CIFAR100(root=data_paths['cifar100'], 
                                                      train=train, 
                                                      download=download, 
                                                      transform=transform)
        else:
            dataset = datasets.CIFAR100(root=data_paths['cifar100'],
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        dataset.num_classes = 100
    elif name == 'imagenet':
        if PSKD:
            dataset = Custom_ImageFolder(root=data_paths[name][split], 
                                                        transform=transform)
        else:
            dataset = datasets.ImageFolder(root=data_paths[name][split],
                                            transform=transform,
                                            target_transform=target_transform)
        dataset.num_classes = 1000

    return dataset


def get_transform(dataset, split):
    mean = data_means[dataset]
    std = data_stds[dataset]
    if 'cifar10' in dataset:
        t = {
            'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val':
            transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        }
    elif 'cifar100' in dataset:
        t = {
            'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val':
            transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        }
    elif 'imagenet' in dataset:
        t = {
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    return t[split]


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    ref:https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples

def distributed_concat(tensor: "torch.Tensor", num_total_examples: Optional[int] = None) -> torch.Tensor:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")

def distributed_broadcast_scalars(scalars: List[Union[int, float]], num_total_examples: Optional[int] = None) -> torch.Tensor:
    try:
        tensorized_scalar = torch.tensor(scalars).cuda()
        output_tensors = [tensorized_scalar.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensorized_scalar)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")

