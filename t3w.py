"""
Copyright (C) 2023 Yuyao Huang - All Rights Reserved

You may use, distribute and modify this code under the terms of the Apache 2.0
license, which unfortunately won't be written for another century. You should
have received a copy of the Apache 2.0 license with this file. If not, please
write to: huangyuyao@outlook.com, or visit
https://github.com/tjyuyao/t3w/blob/main/LICENSE:
"""

__version__ = '0.1.0.post2'

import torch, gzip, random, numpy, os, math, weakref
from typing import Any, Sequence, Generic, TypeVar, Optional, Mapping, Type, Hashable, Callable, Dict, Literal, Union, List
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


cli = typer.Typer(
    add_completion=False,
)


def manual_seed(seed, strict=False):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    if strict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def millify(n, names=["", " K", " M", " B", " T"]):
    n = float(n)
    millidx = max(0, min(len(names) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return "{:.1f}{}".format(n / 10 ** (3 * millidx), names[millidx]).replace(".0", "")


if not os.environ.get("T3W_VERBOSE", False):
    """Some monkey-patch for more concise output."""

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


FloatScalarTensor = Float[Tensor, ""]
StepReturnDict = Dict[Literal["losses", "metrics"], Dict[str, FloatScalarTensor]]


class Interface: pass


class IDatum(Interface):

    uuid: Hashable
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 2

    @staticmethod
    def collate(data: Sequence["IDatum"]) -> "IMiniBatch":
        return default_collate(data)
    
    @classmethod
    def from_uuid(cls, uuid: Hashable) -> "IDatum":
        pass


class IMiniBatch(Interface):

    batch_size: int
    padding_mask: Optional[Bool[Tensor, "_"]]
    model:"TopLevelModule"

    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.model = None
    
    def to(self, device):
        for name in self.__dict__:
            attr = getattr(self, name, None)
            if isinstance(attr, Tensor):
                setattr(self, name, attr.to(device))

    @property
    def batch_size(self) -> int:
        return self.padding_mask.int().sum().item()
    
    @property
    def full_batch_size(self) -> int:
        return self._full_batch_size
        
    @batch_size.setter
    def batch_size(self, batch_size):
        self._full_batch_size = batch_size
        self.padding_mask = torch.ones((batch_size,), dtype=torch.bool)


class IDataset(Interface):

    datum_type: Type[IDatum]

    def __init__(self, root:str, split:str=None) -> None:
        self.root = root
        self.split = split

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> IDatum:
        pass


class IDatumMetric(nn.Module, Interface):

    higher_better: bool = True

    def forward(self, mb: IMiniBatch) -> FloatScalarTensor:
        pass


class ILoss(IDatumMetric):
    
    loss_reweight: float = 1.
    higher_better: bool = False


class LearningRate(IDatumMetric):

    def __init__(self, param_group=0) -> None:
        super().__init__()
        self.pg = param_group

    def forward(self, mb: IMiniBatch) -> FloatScalarTensor:
        return torch.tensor(mb.model.optim.param_groups[self.pg]['lr'])


class IDatasetMetric(Interface):

    datum_metric: IDatumMetric

    def __init__(self, datum_metric: IDatumMetric) -> None:
        self.datum_metric = datum_metric
        self.reset()
    
    def eval(self) -> float:
        pass

    def reset(self):
        pass

    def update(self, mb: IMiniBatch) -> None:
        pass

    def synchronize(self) -> None:
        pass


class AverageMetric(IDatasetMetric):

    def eval(self) -> float:
        return self.sum / max(self.cnt, 1)

    def reset(self):
        self.sum = 0.
        self.cnt = 0

    def update(self, mb: IMiniBatch):
        new_val = self.datum_metric.forward(mb).item()
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
    """ https://stackoverflow.com/a/45690594 """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _subprocess(rank, loop: Union["EvalLoop", "TrainLoop"]):

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
class _TrainingProgress:
    step: int = 0
    epoch: int = 0

    def inc_step(self):
        self.step += 1
    
    def inc_epoch(self):
        self.epoch += 1
    

class TopLevelModule(nn.Module):

    def __init__(
            self,
            model:nn.Module,
            optim:Optimizer = None,
            lr_scheduler:_LRScheduler = None,
            regularizer_reweight: float = 1.
        ) -> None:
        super().__init__()
        self.user_model = model
        self.optim = optim
        self.lr_scheduler:_LRScheduler = lr_scheduler
        self.training_progress = _TrainingProgress()
        self.regularizer_reweight = regularizer_reweight
        self.distributed_devices = []
        self.ddp_enabled = False
        self.ddp_rank = 0
        self.ddp_port = 0

    def to(self, device):
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
        return next(self.user_model.parameters()).device
    
    def save(self, path):
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
        with gzip.open(path, 'rb') as ckpt_file:
            ckpt_dict = torch.load(ckpt_file, map_location=str(self.device))
        self.training_progress = _TrainingProgress(**ckpt_dict['training_progress'])
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainLoop:

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
        self.model = model
        self.dataset = dataset
        self.losses = losses
        self.metrics = metrics
        self.batch_size = batch_size or self.dataset.datum_type.train_batch_size
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
                worker_init_fn=seed_worker,
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
            if epoch % self.epoch_per_eval == 0: self.eval_loop()
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