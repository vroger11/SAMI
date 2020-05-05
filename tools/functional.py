"""Module to complete Pytorch functional module."""
from math import e

import torch


class PreProcess:
    """Class to preprocess features map."""

    def __init__(self, method):
        """Configure which method will be applied.

        Parameters
        ----------
        method: str or None, optional
            Can be either "l1", "l2", "max", "inf", None or "zscore".
        """
        self.method = method
        if self.method not in [None, "l1", "l2", "max", "inf", "zscore"]:
            raise Exception(f"Norm {self.method} not recognised")
        self.parameters = None

    def fit(self, tensor):
        """Fit the parameters with the given tensor.

        Parameters
        ----------
        tensor: torch.Tensor
            minimum of 2d tensor that will be normalized or standardized over
            each values in the last dimension
        """
        tensor_view = tensor.view([-1, tensor.shape[-1]])
        self.parameters = []
        if self.method == "l1":
            # L1 norm
            self.parameters.append(tensor_view.abs().sum(0))
        elif self.method == "l2":
            # L2 norm
            self.parameters.append(tensor_view.pow(2).sum(0).rsqrt())
        elif self.method == "max":
            # L_{max} norm
            self.parameters.append(tensor_view.max(0).values)
        elif self.method == "inf":
            # infinite norm
            self.parameters.append(tensor_view.abs().max(0).values)
        elif self.method is None:
            pass
        elif self.method == "zscore":
            self.parameters.append(tensor_view.mean(0))
            self.parameters.append(tensor_view.std(0))

    def transform(self, tensor):
        """Apply the previously fitted parameters to tensor."""
        tensor_view = tensor.view([-1, tensor.shape[-1]])
        if self.method in ["l1", "l2", "max", "inf"]:
            tensor_view.div_(*self.parameters)
        elif self.method is None:
            pass
        elif self.method == "zscore":
            tensor_view.sub_(self.parameters[0]).div_(self.parameters[1])

        return tensor


def softmax(input_x, dim, c=0):
    """Precise softmax designed to tackle underflow or overflow.

    It is little bit slower than pytorch softmax.

    Parameters
    ----------
    input_x: Tensor
        Input tensor on which softmax will be applied
    dim: int
        Dimension along which Softmax will be computed
    c: int, optional
        Constant used to avoid underflow or overflow in softmax computation.

    Returns
    -------
    softmax: Tensor
        A tensor with the same dimension as input_x
    """
    res = []
    xt = input_x.transpose(0, dim)
    summed = torch.pow(e, c + xt).sum(0)
    for xi in torch.pow(e, c + xt):
        res.append(xi/summed)

    res = torch.stack(res)
    return res.transpose(dim, 0)


def zscore(input_x, dim=0):
    """Compute zscore over dim dimension.

    Parameters
    ----------
    input_x: Tensor
        Input tensor on which zscore will be applied
    dim: int, optional
        Dimension along which to operate. Default is 0.

    Returns
    -------
    zscore: Tensor
        A standardized tensor with the sale dimension as input_x
    """
    xt = input_x.transpose(0, dim)
    xt = (xt - xt.mean(0)) / xt.std(0)
    return xt.transpose(dim, 0)


def zscore_(input_x, dim=0):
    """In place zscore computation over dim dimension.

    Parameters
    ----------
    input_x: Tensor
        Input tensor on which zscore will be applied
    dim: int, optional
        Dimension along which to operate. Default is 0.
    """
    input_x.transpose_(0, dim)
    input_x.sub_(input_x.mean(0)).div_(input_x.std(0))
    input_x.transpose_(dim, 0)
