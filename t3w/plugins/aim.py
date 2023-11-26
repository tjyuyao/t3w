from t3w.core import *
from t3w.core import EvalLoop, IMiniBatch


class AimSideEffect(ISideEffect):

    def __init__(
            self,
            experiment: str,
            hparams_dict: Dict[str, Any],
            run_hash: str = None,
            description: str = None,
            repo: Union[str, Path] = "./",
            track_weights_every_n_steps: int = None,
        ) -> None:
        super().__init__()
        self.run_kwargs = dict(run_hash=run_hash, repo=repo, experiment=experiment)
        self.description = description
        self.hparams_dict = hparams_dict
        self.track_weights_every_n_steps = track_weights_every_n_steps
        self.run = None

    def on_eval_started(self, loop: EvalLoop):
        if self.run is not None:
            return

        import aim
        self.run = aim.Run(**self.run_kwargs)

    def on_train_started(self, loop: TrainLoop):
        import aim
        from aim.storage.treeutils_non_native import convert_to_native_object

        self.run = aim.Run(**self.run_kwargs)
        if self.description:
            self.run.description = self.description
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
        if self.track_weights_every_n_steps and (step + 1) % self.track_weights_every_n_steps == 0:
            from aim.sdk.adapters.pytorch import track_params_dists, track_gradients_dists
            track_params_dists(loop.model, self.run)
            track_gradients_dists(loop.model, self.run)

    def on_eval_step_finished(self, loop: EvalLoop, step: int, mb: IMiniBatch):
        self.handle_medias(step, loop.medias.get_latest_output())

    def handle_medias(self, step:int, media_cache:Sequence[MediaData]):
        from aim import Figure

        media: MediaData
        for media in media_cache:
            if media.media_type in [MediaType.FIGURE_MPL, MediaType.FIGURE_PX]:
                aim_figure = Figure(media.media_data)
                self.run.track(aim_figure, name="figures", step=step, context={"note": media.media_note})
            else:
                raise NotImplementedError(f"Unsupported media type '{media.media_type}'.")
