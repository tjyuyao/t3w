from t3w.core import *
from t3w.core import IMiniBatch, TrainLoop


class WarmUpSideEffect(ISideEffect):

    def __init__(self, num_steps:int) -> None:
        super().__init__()
        self.num_steps = num_steps

    def on_train_started(self, loop: TrainLoop):
        self.base_lrs = loop.model.lr_scheduler.base_lrs
        self.param_groups = loop.model.optim.param_groups

    def on_train_step_started(self, loop: TrainLoop, step: int, mb: IMiniBatch):
        global_steps = loop.model.training_progress.step
        if global_steps <= self.num_steps:
            for i, group in enumerate(self.param_groups):
                group['lr'] = global_steps / self.num_steps * self.base_lrs[i]
