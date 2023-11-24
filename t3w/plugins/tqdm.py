from t3w.core import *


class TqdmSideEffect(ISideEffect):

    def __init__(self, postfix_keys:Sequence[str]= [], smoothing:float = 0.9) -> None:
        """show a tqdm progress bar for training and evaluation, write eval metrics at the end of eval epoch.

        Args:
            postfix_keys (Sequence[str], optional): control the losses and metrics to be displayed on the bar. Empty sequence will show all.
            smoothing (float, optional): low pass filter value. Defaults to 0.9.
        """
        super().__init__()
        from tqdm import tqdm
        self.tqdm = tqdm
        self.postfix_keys = postfix_keys
        self.tqdm_kwargs = dict(smoothing=smoothing, leave=False, dynamic_ncols=True)

    def on_eval_started(self, loop: "EvalLoop"):
        self.eval_pbar = self.tqdm(desc=f"Eval {loop.model.training_progress.epoch}", total=len(loop.loader), **self.tqdm_kwargs)

    def on_eval_step_finished(self, loop: "EvalLoop", step: int, mb: "IMiniBatch"):
        self.eval_pbar.update()

    def on_eval_finished(self, loop: "EvalLoop"):
        self.eval_pbar.close()
        metrics_dict = dict()
        metrics_dict["epoch"] = loop.model.training_progress.epoch
        metrics_dict.update(loop.metric_values)
        self.tqdm.write(repr(metrics_dict))

    def on_train_started(self, loop: "TrainLoop"):
        self.train_epoch_pbar = self.tqdm(desc="Epochs", total=loop.epochs, **self.tqdm_kwargs)

    def on_train_epoch_started(self, loop: "TrainLoop", epoch: int):
        self.train_step_pbar = self.tqdm(desc=f"Train {epoch}", total=(loop.iter_per_epoch or len(loop.loader)), leave=False, dynamic_ncols=True)

    def on_train_step_finished(self, loop: "TrainLoop", step: int, mb: "IMiniBatch", step_return: StepReturnDict):
        self.train_step_pbar.update()

        step_postfix = dict()
        postfix_keys = self.postfix_keys or (list(step_return["losses"].keys()) + list(step_return["metrics"].keys()))
        for key in postfix_keys:
            if key in step_return["losses"]:
                loss_reweight, loss_raw_value = step_return["losses"][key]
                loss_raw_value = f"{loss_raw_value:.4f}"[:6]
                step_postfix[key] = f"{loss_reweight}*{loss_raw_value}"
            elif key in step_return["metrics"]:
                metric_value = f"{step_return['metrics'][key]:.4f}"[:6]
                step_postfix[key] = f"{metric_value}"
            else:
                pass  # missing key ignored
        self.train_step_pbar.set_postfix(step_postfix)

    def on_train_epoch_finished(self, loop: "TrainLoop", epoch: int):
        self.train_step_pbar.close()
        self.train_epoch_pbar.update()

    def on_train_finished(self, loop: "TrainLoop"):
        self.train_epoch_pbar.close()
