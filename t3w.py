"""
Copyright (C) 2023 Yuyao Huang - All Rights Reserved

You may use, distribute and modify this code under the terms of the Apache 2.0
license, which unfortunately won't be written for another century. You should
have received a copy of the Apache 2.0 license with this file. If not, please
write to: huangyuyao@outlook.com, or visit:
https://github.com/tjyuyao/t3w/blob/main/LICENSE 
"""

# docstring style reference: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html
# reStructuredText Primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
# reStructuredText Python: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects

from __future__ import annotations
import torch, gzip, random, numpy, os, math, weakref
from typing import Any, Sequence, NewType, Optional, Mapping, Type, Hashable, Callable, Dict, Literal, Union, List
from functools import wraps
from glob import glob
from torch.utils.data import DataLoader, Sampler, default_collate, DistributedSampler
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel
from jaxtyping import Float, Shaped, Bool
from dataclasses import dataclass, asdict
from pathlib import Path
import typer, click
import torch.distributed as dist
import torch.multiprocessing as mp

__version__ = '0.1.1dev1'


cli = typer.Typer(
    add_completion=False,
)
"""Typer: Typer is a library we choose for building type hints based CLI applications.

Example:

    Here is a simple example::

        import typer

        cli = typer.Typer()


        @cli.command()
        def hello(name: str):
            print(f"Hello {name}")


        @cli.command()
        def goodbye(name: str, formal: bool = False):
            if formal:
                print(f"Goodbye Ms. {name}. Have a good day.")
            else:
                print(f"Bye {name}!")


        if __name__ == "__main__":
            cli()

See https://github.com/tiangolo/typer for more usage information.
"""


def manual_seed(seed:int, strict:bool=False):
    """set all default random generators seed to ``seed``.

    Args:
        seed (int): the value to be set.
        strict (bool, optional): whether to use deterministic pytorch algorithms for better reproducibility. Defaults to False.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    if strict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def millify(n:int, names=["", " K", " M", " B", " T"]):
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


# Some monkey-patch for more concise output.
if not os.environ.get("T3W_VERBOSE", False):

    def _tensor_summary(self:Tensor, *, tensor_contents=None):
        if len(self.shape):
            return f"Tensor({tuple(self.shape)}, {str(self.dtype).replace('torch.', '')}, \"{str(self.device)}\")"
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

    Tensor.__repr__ = _tensor_summary
    nn.Module.__repr__ = _module_summary

    try:
        import rich.traceback
        _old_from_exception = rich.traceback.Traceback.from_exception
        def _rich_traceback_from_exception(cls, *args, suppress:List=[], **kwargs):
            suppress.append(__file__)
            suppress.append(torch)
            return _old_from_exception(*args, suppress=suppress, **kwargs)
        rich.traceback.Traceback.from_exception = classmethod(_rich_traceback_from_exception)
    except ImportError: pass


FloatScalarTensor = NewType("FloatScalarTensor", Float[Tensor, ""])
"""This is a type alias for "float scalar tensor", which is the return type for :meth:`IDatumMetric.forward` method.
"""

StepReturnDict = NewType("StepReturnDict", Dict[Literal["losses", "metrics"], Dict[str, FloatScalarTensor]])
"""This is the returned type of :meth:`TopLevelModule.step`. Typically, users receive this type of data in the event :meth:`ISideEffect.on_train_step_finished`.
"""


class Interface:
    """This is a special abstract class to indicate its direct subclasses are interface and should be implemented by the user.
    
    Note:
        Every direct subclass of :class:`Interface` has its name starting with letter "I".
    """
    pass


class IDatum(Interface):
    """The interface for a single datum. Your dataset is meant to have abundant of this type of objects.

    Note:
        :class:`TrainLoop` or :class:`EvalLoop` can directly make use of attributes and methods defined here through the
        :attr:`IDataset.datum_type` link.
    """

    uuid: Optional[Hashable]
    """universal unique identifier for tracking datum information, optional."""

    train_batch_size: int = 1
    """:class:`TrainLoop` will use this value through :attr:`IDataset.datum_type` as default batch size value."""

    val_batch_size: int = 1
    """:class:`EvalLoop` will use this value through :attr:`IDataset.datum_type` as default batch size value."""

    num_workers: int = 2
    """:class:`TrainLoop` or :class:`EvalLoop` will use this value through :attr:`IDataset.datum_type` for `DataLoader.num_workers`."""

    @staticmethod
    def collate(data: Sequence["IDatum"]) -> "IMiniBatch":
        """collate a mini batch of data into an :class:`IMiniBatch` consumable by user's model and :class:`IDatumMetric`.

        This function is called by :class:`DataLoader` in the :class:`TrainLoop` or :class:`EvalLoop` to collate independently sampled data
        into a mini batch, before which is passed to the :meth:`TopLevelModule.forward`.

        Note:
            It is strongly suggested not to use the pytorch's :func:`default_collate` function but override it here.
            This is because :func:`default_collate` does not support extra types other than numerical tensors, and make
            users not easy to manage their batch data in an object-oriented-programming paradigm. Be explicit, and
            write your own collate function to produce your :class:`IMiniBatch` type.

        Args:
            data (list(IDatum)): sampled mini batch of data as a sequence.

        Returns:
            IMiniBatch: the collated mini batch.
        """
        return default_collate(data)
    
    @classmethod
    def from_uuid(cls, uuid: Hashable) -> "IDatum":
        """build the exact datum from given ``uuid``.

        Ignore this factory method for basic training workflow. It is  
        useful for some workflows that mine, track and analyse specific interesting data.
        One may embed dataset_type, split, and index information to fully recover a datum's identity.

        Args:
            uuid (Hashable): universal unique identifier of the disired datum.

        Returns:
            IDatum: 
        """
        dataset_type, root, split, index = uuid
        return dataset_type(root, split)[index]


class IMiniBatch(Interface):
    """The interface for a minibatch of datum.
    
    It plays the central role of a data context passing around models, metrics, losses and callbacks.
    All interfaces in ``t3w`` has a strict typing requirement, this especially emphasize that functions
    are required to return specific type of data, instead of dynamic types. The flexibility of ``t3w`` 
    is not harmed though, because it is user's freedom to modify the definition of subclasses of the
    :class:`IMiniBatch` interface. And all the functions receives a :class:`IMiniBatch` as input, are 
    allowed (or supposed) to modify it in place. Therefore, standard behaviors of ``t3w`` core libraries
    and :class:``ISideEffect`` based plugin system can rely on the typing system, while users can write
    flexible code in their independent namespace.

    See also:
      * :meth:`TopLevelModule.forward`
      * :meth:`IDatumMetric.forward`
      * :meth:`ISideEffect.on_eval_step_started`
      * :meth:`ISideEffect.on_eval_step_finished`
      * :meth:`ISideEffect.on_train_step_started`
      * :meth:`ISideEffect.on_train_step_finished`
    """

    @property
    def batch_size(self) -> int:
        """valid batch size (exclude padding). See :attr:`padding_mask`."""
        return self.padding_mask.int().sum().item()
    
    @property
    def full_batch_size(self) -> int:
        """actually collated batch size (include padding). See :attr:`padding_mask`."""        
        return self._full_batch_size
        
    @batch_size.setter
    def batch_size(self, full_batch_size):
        self._full_batch_size = full_batch_size
        self.padding_mask = torch.ones((full_batch_size,), dtype=torch.bool)

    padding_mask: Bool[Tensor, "_"]
    """a bool tensor of shape ``(full_batch_size,)`` where ``True`` indicates the corresponding datum is **NOT** a padded one.

    Note:
        The ``full_batch_size`` property stores the ``batch_size`` argument passed to the :meth:`IMiniBatch.__init__` method. This may be actually less than the ``(val_)batch_size`` argument told :meth:`EvalLoop.__init__` because ``drop_last=False``.
    
    Note:
        In DDP mode, when data cannot be evenly distributed to multiple devices at the last iteration of a dataset epoch,
        :class:`DistributedSampler` will pad the samples to ensure same batch size on all devices. The padded samples would
        cause minor error in the evaluation result and should be avoided. Fortunately, :class:`EvalLoop` automatically marks this behavior to the ``padding_mask`` flag, and users can easily handle this case when implementing a custom :class:`IDatumMetric`. 
    
    See also:
      :meth:`EvalLoop.mark_padding`
    """

    model:"TopLevelModule"
    """The :class:`TopLevelModule` which is consuming and modifying this instance of :class:`IMiniBatch`.
    

    The :class:`TopLevelModule` will fill this attribute right before calling the `forward()` method of its internal user_model. Therefore, the ``user_model.forward()``, :meth:`ISideEffect.on_evak_step_finished`, and :meth:`ISideEffect.on_train_step_finished` can make use of it. Before that, this attribute is defaulted to ``None``.
    """

    def __init__(self, batch_size) -> None:
        """

        Args:
            batch_size (int): the actual number of current batch, including padded one. This is typically counted by :meth:`IDatum.collate`.
        """
        self.batch_size = batch_size
        self.model = None

    
    def to(self, device: torch.device):
        """defines how to move this mini-batch type into specified device.

        The default implementation in the base class :class:`IMiniBatch` moves all direct attributes which are instances of ``torch.Tensor`` to target device. This can be good enough for common usage, but nothing stops you from customizing it.

        Note:
            :meth:`TopLevelModule.forward` will call this function right before calling ``user_model.forward()``, so don't bother to do it yourself. You only need to specify the target device at the :class:`TopLevelModule` level using its :meth:`TopLevelModule.to` method.
        
        See also:
            :meth:`TopLevelModule.to`

        Args:
            device (torch.device): target device.
        """
        for name in self.__dict__:
            attr = getattr(self, name, None)
            if isinstance(attr, Tensor):
                setattr(self, name, attr.to(device))


class IDataset(Interface):
    """The interface for the entire dataset (and its split).
    
    This is very simillar to ``torch.Dataset`` (sized dataset) interface, with only subtle modifications
    that :meth:`__getitem__` is required to return an instance of :attr:`datum_type`.
    """

    datum_type: Type[IDatum]
    """User implemented IDatum subclass' **typename**.
    
    Note:
        Subclass of ``IDataset`` must specify this attribute in order to fetch the class attribute including :attr:`IDatum.train_batch_size`, :attr:`IDatum.val_batch_size`, and :attr:`IDatum.num_workers`, etc.
    """

    def __init__(self, root:str, split:str=None) -> None:
        """

        Args:
            root (str): path of the root directory of specified dataset.
            split (str, optional): identifier for a subset. Defaults to None.
        """
        self.root = root
        self.split = split

    def __len__(self) -> int:
        """

        Returns:
            int: the number of data in the current split of dataset.
        """
        pass

    def __getitem__(self, index: int) -> IDatum:
        """_summary_

        Args:
            index (int): _description_

        Returns:
            IDatum: the ``index``-th datum in the dataset.
        """
        pass


class IDatumMetric(nn.Module, Interface):
    """The interface of compute metric for datum in a mini-batch.

    Note:
        We differentiate the :class:`IDatumMetric` and :class:`IDatasetMetric`,
        where the former compute metric value for a batch of data, while the
        latter aggregate datum metric of each batch for a entire dataset (typically
        an "average meter"). The dataset level metric mainly focus on correct computation
        and synchronization of variable batch sizes and among devices.
    
    See also:
        :class:`ILoss`, :class:`IDatasetMetric`, :class:`AverageMetric`.
    """

    higher_better: bool = True
    """specifies whether higher value of the metric implies better performance.
    This can be useful for e.g. metric based best model saving. Better always
    explicitly specify it in your subclass definition.
    """

    def forward(self, mb: IMiniBatch) -> FloatScalarTensor:
        """

        Calling of this method is delegated to :class:`TrainLoop` or :class:EvalLoop` at their construction time.
        The ``mb`` argument must have been through the user_model's ``forward`` method already, and the loops pass
        it on to metrics to calculate the metric value, which must be a float scalar tensor.

        Args:
            mb (IMiniBatch): the mini-batch of data which have been processed by user_model.

        Returns:
            FloatScalarTensor: return metric scalar value.
        """
        pass


class ILoss(IDatumMetric):
    """The interface of compute loss for datum in a mini-batch.

    Note:
        In t3w, we adopt the fact that a loss function is a special type of metric that support backpropagation. Therefore :class:`ILoss` inherites :class:`IDatumMetric` and you can use an ``ILoss`` whereever an ``IDatumMetric`` is suited.
    """
    
    loss_reweight: float = 1.
    """Losses has the raw version and reweighted version in lots of circumstances. Use this standard attribute to specify your weight of current loss.
    
    Note:
        In the "losses" subdict of :class:`StepReturn`, the value of each loss will be reported as a pair of float value ``(loss_reweight, loss_raw_value)`` with event :meth:`on_train_step_finished` emitted. This is the standard behavior that extension codes can rely on. Non-loss metrics are reported as pure floats.
    """

    higher_better: bool = False
    """higher value of a loss always implies worse performance. Don't bother to specify it in the subclasses."""

    def forward(self, mb: IMiniBatch) -> FloatScalarTensor:
        """

        Calling of this method is delegated to :class:`TrainLoop` at its construction time.
        The ``mb`` argument must have been through the user_model's ``forward`` method already, and the loops pass
        it on to various losses to calculate the loss value, which must be a float scalar tensor. And 
        then the losses will be reweighted and summed together for a backward autodiff pass.

        Args:
            mb (IMiniBatch): the mini-batch of data which have been processed by user_model.

        Returns:
            FloatScalarTensor: return loss scalar value.
        """
        pass


class LearningRate(IDatumMetric):
    """This class report current learning rate through the standard metric interface.
    
    This is not a typical metric but it is a commonly used one agnostic to tasks. So we implement it early here.
    It is also a good demonstration of how to use the exposed :class:`TopLevelModule` as the :attr:`IMiniBatch.model` attribute. Since the metric computation is applied after the user_model's ``forward()``, the ``model`` attribute is absolutely available in a :meth:`IDatumMetric.forward` method.
    """

    def __init__(self, param_group=0) -> None:
        """

        Args:
            param_group (int, optional): to show learning rate of which. Defaults to 0.
        """        
        super().__init__()
        self.param_group = param_group

    def forward(self, mb: IMiniBatch) -> FloatScalarTensor:
        """

        Args:
            mb (IMiniBatch): a mini-batch during training.

        Returns:
            FloatScalarTensor: the learning rate of self.param_group
        """
        return torch.tensor(mb.model.optim.param_groups[self.param_group]['lr'])


class IDatasetMetric(IDatumMetric):
    """The interface of a dataset level metric (aggregation algorithm on multi devices).
    
    Note:
        We differentiate the :class:`IDatumMetric` and :class:`IDatasetMetric`,
        where the former compute metric value for a batch of data, while the
        latter aggregate datum metric of each batch for a entire dataset
        (typically an "average meter"). The dataset level metric mainly focus on
        correct computation and synchronization of variable batch sizes and
        among devices.
    
    Warning:
        Typically, :class:`TrainLoop` only accepts datum metric because of
        changing parameters, while :class:`EvalLoop` only accepts dataset metric
        because it want statistics on the entire dataset, and only call
        :meth:`IDatasetMetric.synchronize` once right before emitting event
        :meth:`on_eval_finished`. Other use cases are still possible and
        ``synchronize()`` should also remain correct after arbitrary times of
        call, e.g. a running average dataset metric can be used in a train loop,
        but it is still considered non-standard behavior in ``t3w`` and should
        prefer implemented and maintained in user space using the side effects
        system.
    
    See also:
        :class:`IDatumMetric`, :class:`AverageMetric`.
    """

    datum_metric: IDatumMetric
    """The dataset metric has a standard behavior to composite a datum metric instance, and the calling
    of the datum metric is delegated to :meth:`update`.
    """

    def __init__(self, datum_metric: IDatumMetric) -> None:
        """store ``datum_metric`` and reset the statistics.

        Args:
            datum_metric (IDatumMetric): internal datum_metric instance to embed.
        """
        super().__init__()
        self.datum_metric = datum_metric
        self.reset()
    
    def forward(self, mb: IMiniBatch) -> FloatScalarTensor:
        """allows using dataset metric like datum metric. 
        
        Warning:
            It is non-standard behavior so please be careful implementing a
            :class:`IDatasetMetric` that supposed to be used as a datum metric,
            especially ensure multiple call of synchronization among multiple
            devices does not have ambiguity, for example by separately store the
            synchronized and local version of states.

            **This will be changed in a future version, by using a breaking API with stateless synchronization.**

        Args:
            mb (IMiniBatch): a mini-batch of data.

        Returns:
            FloatScalarTensor: current dataset metric value.
        """
        self.update(mb)
        self.synchronize()
        return torch.tensor(self.eval())

    def eval(self) -> float:
        """get target metric value based on internal statistics."""
        pass

    def reset(self) -> None:
        """clear internal statistics."""
        pass

    def update(self, mb: IMiniBatch) -> None:
        """clear internal statistics."""
        pass

    def synchronize(self) -> None:
        """synchronize local statistics."""
        pass


class AverageMetric(IDatasetMetric):
    """The average metric across data.
    """

    def eval(self) -> float:
        return self.sum / max(self.cnt, 1)

    def reset(self) -> None:
        self.sum = 0.
        self.cnt = 0

    def update(self, mb: IMiniBatch) -> None:
        new_val = self.datum_metric(mb).item()
        k = mb.batch_size
        self.sum += new_val * k
        self.cnt += k
    
    def synchronize(self) -> None:
        if not dist.is_initialized(): return
        object_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            object_list=object_list,
            obj=dict(sum=self.sum, cnt=self.cnt)
        )
        if dist.get_rank() == 0:
            self.sum = sum(obj['sum'] for obj in object_list)
            self.cnt = sum(obj['cnt'] for obj in object_list)
        else:
            self.reset()


def _find_free_port():
    """ find a not used port for DDP.

    https://stackoverflow.com/a/45690594
    """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _subprocess(rank, loop: Union["EvalLoop", "TrainLoop"]):
    """spawned sub processes entrypoint function.

    a :class:`EvalLoop` or :class:`TrainLoop` will spawn multiple processes if
    the :class:`TopLevelModule` it attacted to are told to move its parameters
    to multiple devices by :meth:`TopLevelModule.to`. This implies a distributed
    execution of the loop. The sub processes will then init a communication
    group, actually place model to the target devices, wrap it with torch's
    :class:``DistributedDataParallel`` wrapper, and call the loop's ``__call__``
    again like a single process execution. Finally, it cleans up context and
    exit to the spawning point of the father process.

    Args:
        rank (int): the index of the subprocess, starting from ``0`` to ``len(distributed_devices) - 1``.
        loop (EvalLoop | TrainLoop): the entire loop context is passed to the subprocess.
    """    

    model = loop.model
    devices = model.distributed_devices
    model.to(devices[rank])
    model.distributed_devices = []
    model.ddp_rank = rank
    model.ddp_enabled = True

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(model.ddp_port))

    dist.init_process_group(
        backend=os.environ.get("T3W_DDP_BACKEND", 'nccl'),
        rank=rank,
        world_size=len(devices),
    )
    dist.barrier()

    if isinstance(loop, TrainLoop):
        model._fix_optim_states()
        loop.ddp_model = DistributedDataParallel(
            model,
            static_graph=os.environ.get("T3W_STATIC_GRAPH", '1')=='1',
            )
    
    loop()

    dist.destroy_process_group()


@dataclass
class TrainingProgress:
    """a class that count the total training steps and epochs number of the model.
    
    This progress is a part of the state_dict of the :class:`TopLevelModule`,
    and :class:`TrainLoop` makes use of it to resume training,
    :class:`SaveBestModelsSideEffect` makes use of it to label the checkpoint
    saving filename. So it is "the training progress" instead of "a training
    progress" of the model.
    """

    step: int = 0
    """number of the total times the ``optim.step`` method has been called.
    
    Note:
        It is not about how many actual iteration the for loop has run, it is
        the step of the optimizer has updated the model. Consider a gradient
        accumulation case, only after multiple "iteration steps" will the
        optimizer step once, and the :meth:`inc_step` will also be called only
        once.
    """

    epoch: int = 0
    """number of the total training epoch."""

    def inc_step(self):
        """increase the training step by 1."""
        self.step += 1
    
    def inc_epoch(self):
        """increase the training epoch by 1."""
        self.epoch += 1
    

class TopLevelModule(nn.Module):
    """A central API that manages the ``nn.Module`` related features.
    
    This top level module helps its owner loop on infrastructure the code, and helps user by providing useful low-level utilities, including
        * manages model checkpoint saving and loading,
        * moves user_model to other device(s) and trigger DDP execution mode,
        * computes losses and metrics specified by the loop.
    
    Note:
        We delegate losses computation to :class:`TrainLoop` (see
        :meth:`TrainLoop.__init__`), while the loop delegates it further to
        :meth:`TopLevelModule.forward`. This is clever because torch's
        ``DataDistributedParallel`` class wraps the top level module instead of
        the user's model therefore always has a loss tensor as output and be
        able to find unused parameters during training, while user can stick to
        the suggested standard OOP paradigm in the userspace by operating on the
        :class:`IMiniBatch` instance.
    """

    user_model: nn.Module
    """User's model consumes and updates the input :class:`IMiniBatch` data in its forward method, and optionally return a model specific regularizer loss. """

    optim:Optimizer
    """Stores user defined optimizer, the calling is further delegated to :class:`TrainLoop`. """

    lr_scheduler:_LRScheduler
    """Stores user defined learning rate scheduler, the calling is further delegated to :class:`TrainLoop`. """

    training_progress:TrainingProgress
    """Stores training process of current model. It is part of the checkpoint state_dict. Modification in userspace is not intended."""

    regularizer_reweight:float
    """Stores the weight of regularizer loss if the model has one. It is **NOT** part of the checkpoint state_dict.  Modification in userspace at training should be done through the ``loop`` argument in event :meth:`on_train_step_started`."""


    def __init__(
            self,
            model:nn.Module,
            optim:Optimizer = None,
            lr_scheduler:_LRScheduler = None,
            regularizer_reweight: float = 1.
        ) -> None:
        """

        Args:
            model (nn.Module): user defined model.
            optim (Optimizer, optional): user defined optimizer. Defaults to None.
            lr_scheduler (_LRScheduler, optional): user defined lr_scheduler. Defaults to None.
            regularizer_reweight (float, optional): the weight of the regularizer_loss if any. Defaults to 1.
        """        
        super().__init__()
        self.user_model:nn.Module = model
        self.optim:Optimizer = optim
        self.lr_scheduler:_LRScheduler = lr_scheduler
        self.training_progress = TrainingProgress()
        self.regularizer_reweight:float = regularizer_reweight

        self.distributed_devices = []
        self.ddp_enabled = False
        self.ddp_rank = 0
        self.ddp_port = 0

    def to(self, device):
        """move model to target device or trigger DDP execution if multiple target devices are provided.

        Warning:
            When multiple target devices are specified, the parameters will not be moved until the subprocesses
            are spawned by the :class:`TrainLoop` or :class:`EvalLoop`.
        
        Args:
            device (str): "cuda:0", "cuda:0,1", "cuda:0-3" are all valid inputs.
        """
        devices = self._parse_multi_device(device)
        if 1 == len(devices):
            self.user_model.to(device)
        else:
            self.distributed_devices = devices
            self.ddp_port = _find_free_port()
        return self

    def forward(
            self,
            mb: IMiniBatch,
            losses: Mapping[str, ILoss] = None,
            metrics: Mapping[str, IDatumMetric] = None,
            step_dict: StepReturnDict = None,
        ) -> Union[FloatScalarTensor, IMiniBatch]:
        """
        
        In training mode, this will call ``self.user_model(mb)``, compute ``losses`` and ``metrics``, collect results to fill in ``step_dict``, and return weighted sum of all losses (to be backward).

        In evaluation mode, this will call ``self.user_model(mb)`` and return mb.

        Args:
            mb (IMiniBatch): _description_
            losses (Mapping[str, ILoss], optional): _description_. Defaults to None.
            metrics (Mapping[str, IDatumMetric], optional): _description_. Defaults to None.
            step_dict (StepReturnDict, optional): _description_. Defaults to None.

        Returns:
            Union[FloatScalarTensor, IMiniBatch]: _description_
        """
        mb.to(self.device)
        mb.model = self
        if self.training:
            train_loss:Tensor = 0.
            regularizer_loss = self.user_model(mb)
            if regularizer_loss is not None:
                train_loss += self.regularizer_reweight * regularizer_loss
                step_dict["losses"]["regularizer_loss"] = (self.regularizer_reweight, regularizer_loss)
            for loss_name, loss_obj in losses.items():
                step_dict["losses"][loss_name] = (loss_obj.loss_reweight, loss_obj(mb))
                train_loss += loss_obj.loss_reweight * step_dict["losses"][loss_name][1]
            for metric_name, metric_obj in metrics.items():
                step_dict["metrics"][metric_name] = metric_obj(mb)
            return train_loss
        else:
            self.user_model(mb)
            return mb

    @property
    def device(self):
        """current device (not supporting model parallelzation)"""
        return next(self.user_model.parameters()).device
    
    def save(self, path):
        """save current training states to the disk.

        The states to be maintained include:
            * training progress
            * user model state_dict
            * user optimizer state_dict
            * user lr_scheduler state_dict
            * current random states of ``random``, ``numpy.random`` and ``torch.random``.

        Warning:
            No parent directory will be produced if not exist.

        Args:
            path (str): save path.
        """
        with gzip.open(path, 'wb') as ckpt_file:
            torch.save(
                {
                    "training_progress": asdict(self.training_progress),
                    "model": self.user_model.state_dict(),
                    "optim": self.optim.state_dict() if self.optim is not None else None,
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                    "random_states": {
                        "random": random.getstate(),
                        "numpy": numpy.random.get_state(),
                        "torch": torch.random.get_rng_state(),
                    }
                },
                ckpt_file
            )

    def load(self, path, strict=True):
        """load a previous top level model checkpoint

        Args:
            path (str): load file path
            strict (bool, optional): passed on to user_model's ``load_state_dict(strict=?)`` option. Defaults to True.

        Returns:
            _IncompitableKeys: returned incompitable keys of user_model's ``load_state_dict()`` method.
        """
        with gzip.open(path, 'rb') as ckpt_file:
            ckpt_dict = torch.load(ckpt_file, map_location=str(self.device))
        self.training_progress = TrainingProgress(**ckpt_dict['training_progress'])
        model_load_return = self.user_model.load_state_dict(ckpt_dict['model'], strict=strict)
        if self.optim is not None:
            self.optim.load_state_dict(ckpt_dict['optim'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
        random.setstate(ckpt_dict['random_states']['random'])
        numpy.random.set_state(ckpt_dict['random_states']['numpy'])
        torch.random.set_rng_state(ckpt_dict['random_states']['torch'].cpu())
        return model_load_return
    
    def _fix_optim_states(self):
        """
        _LRScheduler of PyTorch makes a monkey patch on the optimizer to help detect the proper order of calling
        ``optim.step`` and ``lr_scheduler.step``, this patch will be lost though in a multi-processing spawning.
        We reproduce this monkey patch in the sub process to repress the warning information.
        """
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper
        self.optim.step = with_counter(self.optim.step)
        self.optim._step_count = 0
    
    @staticmethod
    def _parse_multi_device(devices:str) -> List[torch.device]:
        """
        Args:
            - devices: e.g. "cuda", "cuda:0", "cuda:0,1", "cuda:0-2", "cuda:0-1,3".
        Return:
            List["torch.device"]
        """

        if isinstance(devices, (int, torch.device)): return ([devices], None)

        # determine device type
        if devices[:3] == "cpu": device_type = "cpu"
        elif devices[:4] == "cuda": device_type = "cuda"
        else: raise ValueError(f"Unsupported devices value: {repr(devices)}")

        # determine device indices
        idid = devices.find(":")
        if -1 == idid: out = [torch.device(device_type, 0)]
        else:
            out = []
            for indices in devices[idid+1:].split(","):
                if -1 != indices.find("-"):
                    s, e = indices.split("-")
                    for i in range(int(s), int(e) + 1):
                        out.append(torch.device(device_type, i))
                else:
                    out.append(torch.device(device_type, int(indices)))
            if 0 == len(out): raise ValueError("Empty devices indices.")
        return out


class EvalLoop:

    model: TopLevelModule
    dataset: IDataset
    batch_size: int
    loader: DataLoader
    metrics: Mapping[str, IDatasetMetric]

    def __init__(self,
                 model: TopLevelModule,
                 dataset: IDataset,
                 batch_size: int = None,
                 metrics: Mapping[str, IDatasetMetric] = dict(),
                 side_effects: Sequence["ISideEffect"] = [],
                 ) -> None:
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.handle:_EventHandler = _EventHandler(side_effects)
        self.loader:DataLoader = None
        self.batch_size:int = batch_size or self.dataset.datum_type.val_batch_size
    
    def __call__(self) -> None:

        if self.model.distributed_devices:
            return mp.start_processes(_subprocess, (self,), nprocs=len(self.model.distributed_devices), join=True, start_method='spawn')
        
        self.loader = DataLoader(
            self.dataset,
            sampler=DistributedSampler(self.dataset, shuffle=False) if self.model.ddp_enabled else None,
            batch_size=self.batch_size,
            collate_fn=self.dataset.datum_type.collate,
            num_workers=self.dataset.datum_type.num_workers,
            pin_memory=self.model.device.type!="cpu",
            pin_memory_device=str(self.model.device) if self.model.device.type!="cpu" else "",
        )

        self.model.train(False)
        for metric in self.metrics.values():
            metric.reset()
        self.handle('on_eval_started', self)
        for step, mb in enumerate(self.loader):
            self.handle('on_eval_step_started', self, step, mb)
            self.model(mb)
            self.mark_padding(step, mb)
            for metric in self.metrics.values():
                metric.update(mb)
            self.handle('on_eval_step_finished', self, step, mb)
        for metric in self.metrics.values():
            metric.synchronize()
        self.handle('on_eval_finished', self)
        self.model.train(True)

    @property
    def metric_values(self) -> Dict[str, float]:
        return {k:v.eval() for k, v in self.metrics.items()}
    
    def mark_padding(self, step, mb: IMiniBatch):
        if dist.is_initialized():
            processed = mb.full_batch_size * (step * dist.get_world_size() + dist.get_rank())
        else:
            processed = mb.full_batch_size * step
        valid = max(min(len(self.dataset) - processed, mb.full_batch_size), 0)
        mb.padding_mask[valid:].zero_()


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainLoop:

    model: TopLevelModule
    dataset: IDataset
    batch_size: int
    num_acc_grad: int
    epochs: int
    loader: DataLoader
    losses: Mapping[str, ILoss]
    metrics: Mapping[str, IDatasetMetric]
    eval_loop: EvalLoop

    def __init__(
            self,
            model: TopLevelModule,
            dataset: IDataset,
            losses: Mapping[str, ILoss],
            metrics: Mapping[str, IDatumMetric],
            batch_size: Optional[int] = None,
            explicit_sampler: Sampler = None,
            num_acc_grad: int = 1,
            epochs: int = 100,
            iter_per_epoch: Optional[int] = None,
            epoch_per_eval: int = 1,
            eval_loop: Optional[EvalLoop] = None,
            side_effects: Sequence["ISideEffect"] = [],
        ) -> None:
        """_summary_

        Args:
            model (TopLevelModule): the top level model.
            dataset (IDataset): train split of the dataset.
            losses (Mapping[str, ILoss]): losses to be evaluated.
            metrics (Mapping[str, IDatumMetric]): metrics to be evaluated
            batch_size (Optional[int], optional): train batch size. Defaults to the ``train_batch_size`` static attribute of :class:`IDatum`.
            explicit_sampler (Sampler, optional): a custom sampler instance for the dataloader. Defaults to ``RandomSampler`` or ``DistributedSampler``.
            num_acc_grad (int, optional): step interval to apply gradient descent. Defaults to 1.
            epochs (int, optional): total training epochs. Defaults to 100.
            iter_per_epoch (Optional[int], optional): manually define iteration number of an epoch. Defaults to ``len(dataloader)``.
            epoch_per_eval (int, optional): epoch interval to apply ``eval_loop``. Defaults to 1.
            eval_loop (Optional[EvalLoop], optional): an :class:`EvalLoop` instance. Defaults to None.
            side_effects (Sequence[ISideEffect], optional): Defaults to [].
        """
        self.model:TopLevelModule = model
        self.dataset = dataset
        self.losses = losses
        self.metrics = metrics
        self.batch_size:int = batch_size or self.dataset.datum_type.train_batch_size
        """The normal mini batch size during training. """
        self.num_acc_grad = num_acc_grad
        self.epochs = epochs
        self.iter_per_epoch = iter_per_epoch
        self.epoch_per_eval = epoch_per_eval
        self.eval_loop = eval_loop
        self.handle:_EventHandler = _EventHandler(side_effects)
        self.explicit_sampler = explicit_sampler
        self.model_forward = model.forward
        self.loader:DataLoader = None
        self.ddp_model: DistributedDataParallel = None
    
    def __call__(self):
        """Implement iter_based and epoch_based train_loop with hooks."""
    
        
        if self.model.distributed_devices:
            return mp.start_processes(_subprocess, (self,), nprocs=len(self.model.distributed_devices), join=True, start_method='spawn')
        
        
        for loss in self.losses.values(): loss.to(self.model.device)
        for metric in self.metrics.values(): metric.to(self.model.device)


        if self.model.ddp_enabled:
            self.model_forward = self.ddp_model.forward

            loader_kwargs = dict(
                sampler=self.explicit_sampler or DistributedSampler(self.dataset, shuffle=True, drop_last=True),
            )
        else:
            g = torch.Generator()
            g.manual_seed(torch.initial_seed())
            loader_kwargs = dict(
                shuffle=self.explicit_sampler is None,
                sampler=self.explicit_sampler,
                generator=g,
                drop_last=True,
                worker_init_fn=_seed_worker,
            )

        if isinstance(loader_kwargs['sampler'], DistributedSampler):
            set_epoch = lambda epoch: loader_kwargs['sampler'].set_epoch(epoch)
        else:
            set_epoch = lambda _: None

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dataset.datum_type.num_workers,
            collate_fn=self.dataset.datum_type.collate,
            pin_memory=self.model.device.type!="cpu",
            pin_memory_device=str(self.model.device) if self.model.device.type!="cpu" else "",
            **loader_kwargs
        )

        start_epoch = self.model.training_progress.epoch
        iter_loader = iter(())
        data_size = self.iter_per_epoch or len(self.loader)
        self.handle('on_train_started', self)
        for epoch in range(start_epoch, self.epochs):
            self.handle('on_train_epoch_started', self, epoch)
            for step in range(data_size):
                try: mb = next(iter_loader)
                except StopIteration:
                    set_epoch(self.model.training_progress.step // len(self.loader))
                    iter_loader = iter(self.loader)
                    mb = next(iter_loader)
                self.handle('on_train_step_started', self, step, mb)
                step_return = self.step(mb)
                self.handle('on_train_step_finished', self, step, mb, step_return)
            if len(step_return["losses"]):
                self.model.lr_scheduler.step()
            if self.eval_loop and epoch % self.epoch_per_eval == 0: self.eval_loop()
            self.model.training_progress.inc_epoch()
            self.handle('on_train_epoch_finished', self, epoch)

    def step(self, mb: IMiniBatch):
        """Implement a single train step, with gradiant accumulate."""
        step_dict = dict(losses=dict(), metrics=dict())
        train_loss = self.model_forward(mb, self.losses, self.metrics, step_dict)
        if isinstance(train_loss, Tensor):
            (train_loss / self.num_acc_grad).backward()
            training_progress = self.model.training_progress
            training_progress.inc_step()
            if training_progress.step % self.num_acc_grad == 0:
                self.model.optim.step()
                self.model.optim.zero_grad()
        return step_dict


class _EventHandler:

    def __init__(self, side_effects: Sequence["ISideEffect"] = []) -> None:
        self.side_effects = side_effects

    def __call__(self, event: str, *args, **kwargs):
        for side_effect in self.side_effects:
            if dist.is_initialized():
                abort = 0
                if dist.get_rank() == 0 or side_effect.is_distributed:
                    try: getattr(side_effect, event)(*args, **kwargs)
                    except click.Abort:
                        abort = 1
                object_list = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(object_list, abort)
                if sum(object_list):
                    if dist.get_rank() == 0:
                        click.echo(click.style("Aborted.", fg="red"), color=True)
                    import sys
                    sys.exit()
            else:
                getattr(side_effect, event)(*args, **kwargs)
    

class ISideEffect(Interface):

    is_distributed:bool = False
    """if False, the event will only be invoked on the rank 0 sub-process; otherwise on all sub-processes."""

    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, d):
        self.__dict__ = d

    def on_eval_started(self, loop: "EvalLoop"):
        pass
    
    def on_eval_step_started(self, loop: "EvalLoop", step: int, mb: "IMiniBatch"):
        pass

    def on_eval_step_finished(self, loop: "EvalLoop", step: int, mb:"IMiniBatch"):
        pass

    def on_eval_finished(self, loop: "EvalLoop"):
        pass

    def on_train_started(self, loop: "TrainLoop"):
        pass

    def on_train_epoch_started(self, loop: "TrainLoop", epoch: int):
        pass

    def on_train_step_started(self, loop: "TrainLoop", step: int, mb: "IMiniBatch"):
        pass

    def on_train_step_finished(self, loop: "TrainLoop", step: int, mb: "IMiniBatch", step_return: StepReturnDict):
        pass

    def on_train_epoch_finished(self, loop: "TrainLoop", epoch: int):
        pass

    def on_train_finished(self, loop: "TrainLoop"):
        pass


class TqdmSideEffect(ISideEffect):

    def __init__(self, postfix_keys:Sequence[str]= []) -> None:
        super().__init__()
        from tqdm import tqdm
        self.tqdm = tqdm
        self.postfix_keys = postfix_keys

    def on_eval_started(self, loop: "EvalLoop"):
        self.eval_pbar = self.tqdm(desc=f"Eval {loop.model.training_progress.epoch}", total=len(loop.loader), leave=False, dynamic_ncols=True)
    
    def on_eval_step_finished(self, loop: "EvalLoop", step: int, mb: "IMiniBatch"):
        self.eval_pbar.update()
    
    def on_eval_finished(self, loop: "EvalLoop"):
        self.eval_pbar.close()
        metrics_dict = dict()
        metrics_dict["epoch"] = loop.model.training_progress.epoch
        metrics_dict.update(loop.metric_values)
        self.tqdm.write(repr(metrics_dict))
    
    def on_train_started(self, loop: "TrainLoop"):
        self.train_epoch_pbar = self.tqdm(desc="Epochs", total=loop.epochs, leave=False, dynamic_ncols=True)

    def on_train_epoch_started(self, loop: "TrainLoop", epoch: int):
        self.train_step_pbar = self.tqdm(desc=f"Train {epoch}", total=(loop.iter_per_epoch or len(loop.loader)), leave=False, dynamic_ncols=True)
    
    def on_train_step_finished(self, loop: "TrainLoop", step: int, mb: "IMiniBatch", step_return: StepReturnDict):
        self.train_step_pbar.update()

        step_postfix = dict()
        for key in self.postfix_keys:
            if key in step_return["losses"]:
                loss_reweight, loss_raw_value = step_return["losses"][key]
                step_postfix[key] = f"{loss_reweight}*{loss_raw_value:.5f}"
            elif key in step_return["metrics"]:
                step_postfix[key] = f"{step_return['metrics'][key]:.5f}"
            else:
                pass  # missing key ignored
        self.train_step_pbar.set_postfix(step_postfix)

    def on_train_epoch_finished(self, loop: "TrainLoop", epoch: int):
        self.train_step_pbar.close()
        self.train_epoch_pbar.update()

    def on_train_finished(self, loop: "TrainLoop"):
        self.train_epoch_pbar.close()


class AimSideEffect(ISideEffect):

    def __init__(
            self,
            repo: Union[str, Path],
            experiment: str,
            hparams_dict: Dict[str, Any],
        ) -> None:
        super().__init__()
        self.run_kwargs = dict(repo=repo, experiment=experiment)
        self.hparams_dict = hparams_dict

    def on_train_started(self, loop: TrainLoop):
        import aim
        self.run = aim.Run(**self.run_kwargs)
        self.run['hparams'] = self.hparams_dict

    def on_eval_finished(self, loop: EvalLoop):
        prog = loop.model.training_progress
        for name, value in loop.metric_values.items():
            self.run.track(value, name=f"eval/{name}", step=prog.step, epoch=prog.epoch)

    def on_train_step_finished(self, loop: TrainLoop, step: int, mb: "IMiniBatch", step_return: StepReturnDict):
        prog = loop.model.training_progress
        for name, (loss_reweight, loss_raw_value) in step_return['losses'].items():
            self.run.track(loss_raw_value, name=f"train/loss/{name}/raw", step=prog.step, epoch=prog.epoch)
            self.run.track(loss_reweight * loss_raw_value, name=f"train/loss/{name}/reweighted", step=prog.step, epoch=prog.epoch)
        for name, value in step_return['metrics'].items():
            self.run.track(value, name=f"train/metric/{name}", step=prog.step, epoch=prog.epoch)


class SaveBestModelsSideEffect(ISideEffect):

    def __init__(
            self,
            metric_name: str,
            num_max_keep: int,
            save_path_prefix: str="./",
        ) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.num_max_keep = num_max_keep
        self.save_path_prefix = save_path_prefix
        self.history:List[Dict[Literal["metric_value", "saved_path"], Any]] = []

    def on_train_started(self, loop: TrainLoop):
        assert self.metric_name in loop.eval_loop.metrics.keys(), "metric_name for saving model does not exist in eval_loop"
        
        matched_existing_paths = [path for path in glob(self.save_path_prefix+"*") if self.metric_name in path and ".pt" in path]

        if len(matched_existing_paths):
            import sys
            sys.stdin = os.fdopen(0)
            click.echo(f"{self.__class__.__name__}: found existing checkpoint files matching given prefix:")
            for path in matched_existing_paths:
                click.echo(f'\t* {path}')
            if click.confirm("Should I DELETE them for current running?", default=False, abort=True):
                for path in matched_existing_paths:
                    os.remove(path)

    def on_eval_finished(self, loop: EvalLoop):

        if not (self.num_max_keep > 0): return

        metric_value = loop.metric_values[self.metric_name]
        higher_better = loop.metrics[self.metric_name].datum_metric.higher_better
        better = lambda a, b: (higher_better and a >= b) or (not higher_better and a <= b)

        if self.history:
            sorted_history = sorted(
                self.history,
                key=lambda item: item["metric_value"],
                reverse=higher_better
            )
            topk = sorted_history[:self.num_max_keep][-1]
            if not better(metric_value, topk["metric_value"]):
                return
            if len(sorted_history) >= self.num_max_keep:
                for item in sorted_history[self.num_max_keep-1:]:
                    os.remove(item['saved_path'])
                self.history = sorted_history[:self.num_max_keep-1]
        
        current = dict(
            metric_value=metric_value,
            saved_path=f"{self.save_path_prefix}-ep{loop.model.training_progress.epoch}-step{millify(loop.model.training_progress.step)}-{self.metric_name}={metric_value:.5f}.pt.gz".replace("--ep", "-ep"),
        )
        loop.model.save(current["saved_path"])
        self.history.append(current)