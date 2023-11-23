from t3w.core import *


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
            raise NotImplementedError()
            # import sys
            # sys.stdin = os.fdopen(0)
            # click.echo(f"{self.__class__.__name__}: found existing checkpoint files matching given prefix:")
            # for path in matched_existing_paths:
            #     click.echo(f'\t* {path}')
            # if click.confirm("Should I DELETE them for current running?", default=False, abort=True):
            #     for path in matched_existing_paths:
            #         os.remove(path)

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

        current = dict(
            metric_value=metric_value,
            saved_path=f"{self.save_path_prefix}-ep{loop.model.training_progress.epoch}-step{millify(loop.model.training_progress.step)}-{self.metric_name}={metric_value:.5f}.pt.gz".replace("--ep", "-ep"),
        )
        loop.model.save(current["saved_path"])
        self.history.append(current)
