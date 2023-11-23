from t3w.core import *


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
        from aim.storage.treeutils_non_native import convert_to_native_object

        self.run = aim.Run(**self.run_kwargs)
        self.run['hparams'] = {k:convert_to_native_object(v, strict=False) for k, v in self.hparams_dict.items()}

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
