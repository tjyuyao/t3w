import torch.distributed as dist
from random import choices
from typing import Iterator, List
from t3w.core import IMiniBatch, StepReturnDict, TrainLoop
from torch.utils.data.sampler import Sampler
from ..core import ISideEffect, IDataset


class HarderBatchSampler(Sampler[List[int]], ISideEffect):

    def __init__(self, dataset: IDataset, based_on: List[str], batch_size:int = None):
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))
        self.weights = [1. for _ in self.indices]
        self.based_on = based_on
        self.batch_size = batch_size or dataset.datum_type.train_batch_size
        self.batch_indices = []
        if based_on == []:
            raise ValueError("'based_on' argument should not be empty.")

    def on_train_step_finished(self, loop: TrainLoop, step: int, mb: IMiniBatch, step_return: StepReturnDict):
        total_score = 0.
        for key in self.based_on:
            if key in loop.losses:
                loss_obj = loop.losses[key]
                score = loss_obj.loss_reweight * loss_obj.raw_output.detach().flatten()
            elif key in loop.metrics:
                metric_obj = loop.metrics[key]
                score = metric_obj.raw_output.detach().flatten()
            else:
                raise ValueError(f"'based_on' argument has invalid member '{key}'")
            if len(score) != self.batch_size:
                raise ValueError(f"'{key}' metric/loss does not produce a batched output.")
            total_score += score

        # batch_score = (total_score.softmax(0) * self.batch_size).tolist()
        batch_score = (total_score / total_score.sum().clip(min=1e-5) * self.batch_size).tolist()
        for i, s in zip(self.batch_indices, batch_score):
            self.weights[i] = s
        self.batch_indices.clear()

    def __iter__(self) -> Iterator[List[int]]:
        return self

    def __next__(self) -> List[int]:
        self.batch_indices = choices(self.indices, self.weights, k=self.batch_size)
        return self.batch_indices

    def __len__(self):
        if dist.is_initialized():
            return self.num_samples // dist.get_world_size() // self.batch_size
        return self.num_samples
