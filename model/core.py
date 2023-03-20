import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as weight_norm_fn


import torch


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_std", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dims)
            batch_std = x.std(dims, unbiased=False) + self.eps
            r = (
                batch_std.detach() / self.running_std.view_as(batch_std)
            ).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")


def my_groupnorm(num_hidden):
    # return nn.BatchNorm1d(num_hidden)
    num_g = 8
    assert num_hidden % num_g == 0
    return nn.GroupNorm(num_g, num_hidden)

def get_normalization(norm_type):
    act_normalization = {
        'none': nn.Identity,
        'batch': nn.BatchNorm1d,
        'layer': nn.LayerNorm,
        'group': my_groupnorm,
        'instance': nn.InstanceNorm1d,
        'batch_renorm': BatchRenorm1d,
    }[norm_type]
    return act_normalization

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation=nn.ReLU, act_normalization=nn.Identity, num_hiddens=3):
        super().__init__()
        if num_hiddens == 0:
            print("MLP is just a linear layer")
            self.block = nn.Linear(input_dim, output_dim)
            return
        assert num_hiddens > 0
        layers = [nn.Linear(input_dim, hidden_dim), act_normalization(hidden_dim), activation()]
        for _ in range(num_hiddens-1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_normalization(hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out
