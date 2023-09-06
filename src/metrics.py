import torch
from torch import nn
from typing import Union


def cw_n01(sample: torch.Tensor, gamma: Union[float, torch.Tensor]):
    batch_dim, feature_dim = sample.size()
    gamma = torch.as_tensor(gamma)

    K = 1.0 / (2.0 * feature_dim - 3.0)

    A1 = torch.pdist(sample) ** 2
    A = 2.0 * torch.rsqrt(gamma + K * A1).sum() / batch_dim ** 2 + torch.rsqrt(gamma) / batch_dim

    B1 = torch.linalg.norm(sample, 2, axis=1) ** 2
    B = torch.rsqrt(gamma + 0.5 + K * B1).mean()

    result = A + torch.rsqrt(1 + gamma) - 2.0 * B

    return result * 0.28209479177

