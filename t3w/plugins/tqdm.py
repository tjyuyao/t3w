from t3w.core import *


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
