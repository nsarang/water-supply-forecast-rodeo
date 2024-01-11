# from tianshou.data.batch import Batch
from numbers import Number
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def to_torch(
    x: Union[Dict, List, Tuple, np.number, np.bool_, Number, np.ndarray, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Union[Dict, List, tuple, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x).to(device)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)  # type: ignore
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    # elif isinstance(x, (dict, Batch)):
    #     if isinstance(x, Batch):
    #         x = dict(x.items())
    #     return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_torch(e, dtype, device) for e in x]
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


def to_torch_as(
    x: Union[dict, list, tuple, np.ndarray, torch.Tensor],
    y: torch.Tensor,
) -> Union[dict, list, tuple, torch.Tensor]:
    """Return an object without np.ndarray.
    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)
