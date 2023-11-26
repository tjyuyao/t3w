import torch, os, math
import numpy as np

from typing import List
from torch import nn, Tensor


suppress_traceback = []


# Some monkey-patch for more concise output.
if not os.environ.get("T3W_VERBOSE", False):

    def _ndarray_summary(self:np.ndarray):
        if len(self.shape):
            nan_count = np.isnan(self).sum() if isinstance(self, np.floating) else 0
            nan_str = f" nan={nan_count}," if nan_count else ""
            return f"NDArray({tuple(self.shape)}, {str(self.dtype).replace('torch.', '')},{nan_str} \"cpu\")"
        else:
            return f"Scalar({self.item()}, {str(self.dtype).replace('torch.', '')}, \"cpu\")"

    def _tensor_summary(self:Tensor):
        if len(self.shape):
            nan_count = torch.isnan(self).sum() if torch.is_floating_point(self) else 0
            nan_str = f" nan={nan_count}," if nan_count else ""
            return f"Tensor({tuple(self.shape)}, {str(self.dtype).replace('torch.', '')},{nan_str} \"{str(self.device)}\")"
        else:
            return f"Scalar({self.item()}, {str(self.dtype).replace('torch.', '')}, \"{str(self.device)}\")"

    def _module_summary(self:nn.Module):
        nparams = sum(p.numel() for p in self.parameters())
        try: device = next(self.parameters()).device
        except StopIteration: device = None
        head_repr = self.extra_repr()
        tail_repr = f"nparams={millify(nparams)}, device=\"{device}\""
        args_repr = f"{head_repr}, {tail_repr}" if head_repr else tail_repr
        return f"{self.__class__.__name__}({args_repr})"

    def _verbose_ndarray(x):
        np.set_string_function(None)
        x = repr(x)
        np.set_string_function(_ndarray_summary, repr=True)
        return x

    _module_raw_repr = nn.Module.__repr__

    def _verbose_module(x:nn.Module):
        nn.Module.__repr__ = _module_raw_repr
        x = repr(x)
        nn.Module.__repr__ = _module_summary
        return x

    _VERBOSE_REPRS = {
        Tensor: Tensor.__repr__,
        nn.Module: _verbose_module,
        np.ndarray: _verbose_ndarray,
    }

    class ReprStr(str):
        def __repr__(self):
            return self

    def verbose(data):
        for T, fn in _VERBOSE_REPRS.items():
            if isinstance(data, T):
                return ReprStr(fn(data))
        return data

    Tensor.__repr__ = _tensor_summary
    nn.Module.__repr__ = _module_summary
    np.set_string_function(_ndarray_summary, repr=True)

    try:

        # suppress_traceback.append(__file__)
        suppress_traceback.append(torch)

        import rich.traceback
        _old_from_exception = rich.traceback.Traceback.from_exception
        def _rich_traceback_from_exception(cls, *args, suppress:List=[], **kwargs):
            suppress.extend(suppress_traceback)
            kwargs['locals_max_length'] = os.environ.get("T3W_LOCALS_MAXLEN", 3)
            return _old_from_exception(*args, suppress=suppress, **kwargs)
        rich.traceback.Traceback.from_exception = classmethod(_rich_traceback_from_exception)
    except ImportError: pass

else:
    def verbose(data):
        return data


def millify(n:int, names=["", "K", "M", "B", "T"]):
    """format an integer number into a human-friendly string.

    Args:
        n (int): the number to be formatted.
        names (list, optional): postfix string for one, thousand, million, etc. Defaults to ["", " K", " M", " B", " T"].

    Returns:
        str: the fomatted string.
    """
    n = float(n)
    millidx = max(0, min(len(names) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return "{:.1f}{}".format(n / 10 ** (3 * millidx), names[millidx]).replace(".0", "")
