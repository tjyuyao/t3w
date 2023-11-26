from t3w.core import *
from t3w.core import EvalLoop, IMiniBatch
from .patch import Run
from PIL import Image as PIL_Image
import aim
import matplotlib.figure
from functools import cached_property


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

    @cached_property
    def run(self):
        from aim.storage.treeutils_non_native import convert_to_native_object

        run = Run(**self.run_kwargs)
        if self.description:
            run.description = self.description
        run['hparams'] = {k:convert_to_native_object(v, strict=False)
                          for k, v in self.hparams_dict.items()}

        return run

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

        self.handle_medias(loop.medias.get_latest_output(), step=prog.epoch, training=True)

    def on_eval_step_finished(self, loop: EvalLoop, step: int, mb: IMiniBatch):
        prog = loop.model.training_progress
        self.handle_medias(loop.medias.get_latest_output(), step=prog.epoch, training=False)

    def handle_medias(self, media_cache:Sequence[MediaData], step:NotImplemented, training:bool):
        media: MediaData
        for media in media_cache:
            if media.media_type in [MediaType.FIGURE_PX]:
                aim_figure = aim.Figure(media.media_data)
                self.run.track(aim_figure, name=media.media_type, step=step, context={"note": media.media_note, "training": training})
            elif media.media_type in [MediaType.FIGURE_MPL]:
                aim_image = aim.Image(fig2img(media.media_data))
                self.run.track(aim_image, name=media.media_type, step=step, context={"note": media.media_note, "training": training})
            else:
                raise NotImplementedError(f"Unsupported media type '{media.media_type}'.")


def fig2img(fig:matplotlib.figure.Figure) -> PIL_Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL_Image.open(buf)
    return img
