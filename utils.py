import torch

__all__ = ['flatten', 'add_white_noise', 'accuracy']


def flatten(x):
    return x.view(x.size(0), -1)


def add_white_noise(x, mean=0, stddev=0.1):
    noise = x.data.new(x.size()).normal_(mean, stddev)
    return (x + noise).clamp(0, 1)


def accuracy(input, target):
    _, max_indices = torch.max(input.data, 1)
    acc = (max_indices == target).sum().float() / max_indices.size(0)
    return acc.item()
