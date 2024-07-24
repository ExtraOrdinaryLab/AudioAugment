from functools import partial
from collections.abc import Sequence

import torch
import numpy as np
import torch.nn as nn
from scipy import sparse


def to_tensor(X, device, accept_sparse=False):
    to_tensor_ = partial(to_tensor, device=device)

    if is_torch_data_type(X):
        return X.to(device)
    if isinstance(X, dict):
        return {key: to_tensor_(val) for key, val in X.items()}
    if isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]
    if np.isscalar(X):
        return torch.as_tensor(X, device=device)
    if isinstance(X, Sequence):
        return torch.as_tensor(np.array(X), device=device)
    if isinstance(X, np.ndarray):
        return torch.as_tensor(X, device=device)
    if sparse.issparse(X):
        if accept_sparse:
            return torch.sparse_coo_tensor(
                X.nonzero(), X.data, size=X.shape).to(device)
        raise TypeError("Sparse matrices are not supported. Set "
                        "accept_sparse=True to allow sparse matrices.")

    raise TypeError("Cannot convert this data type to a torch tensor.")


def is_torch_data_type(x):
    # pylint: disable=protected-access
    return isinstance(x, (torch.Tensor, nn.utils.rnn.PackedSequence))


def to_numpy(X):
    if isinstance(X, np.ndarray):
        return X

    if is_pandas_ndframe(X):
        return X.values

    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


def is_pandas_ndframe(x):
    # the sklearn way of determining this
    return hasattr(x, 'iloc')