from pathlib import Path
from t3w.core import *
from t3w.core import EvalLoop
from t3w.utils.verbose import millify
from functools import cached_property


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
        Path(save_path_prefix + "-").parent.mkdir(parents=True, exist_ok=True)

    def on_train_started(self, loop: TrainLoop):
        assert self.metric_name in loop.eval_loop.metrics.keys(), "metric_name for saving model does not exist in eval_loop"
        self._guard_duplicated_saving_path

    def on_eval_started(self, loop: EvalLoop):
        self._guard_duplicated_saving_path

    @cached_property
    def _guard_duplicated_saving_path(self):
        matched_existing_paths = [path for path in glob(self.save_path_prefix+"*") if self.metric_name in path and ".pt" in path]

        if len(matched_existing_paths):
            raise FileExistsError(f"{self.__class__.__name__}: found existing checkpoint files matching {self.save_path_prefix}*")

        return True

    def on_eval_finished(self, loop: EvalLoop):

        if not (self.num_max_keep > 0): return

        metric_value = loop.metric_values[self.metric_name]
        higher_better = loop.metrics[self.metric_name].minibatch_metric.higher_better
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

        short_bar = "-" if Path(self.save_path_prefix + "-").name != "-" else ""
        current = dict(
            metric_value=metric_value,
            saved_path=f"{self.save_path_prefix}{short_bar}ep{loop.model.training_progress.epoch}-step{millify(loop.model.training_progress.step)}-{self.metric_name}={metric_value:.5f}.pt.gz".replace("--ep", "-ep"),
        )
        loop.model.save(current["saved_path"])
        self.history.append(current)
