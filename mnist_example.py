from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from t3w import *


class MNISTDatum(IDatum):

    train_batch_size = 64
    val_batch_size = 1000
    num_workers = 0

    def __init__(self, input, gt, index=None) -> None:
        super().__init__()
        self.input = input
        self.gt = gt
        self.index = index

    @staticmethod
    def collate(data: Sequence["MNISTDatum"]) -> "MNISTMiniBatch":
        batch_size = len(data)
        input = torch.stack([datum.input for datum in data])
        gt = torch.tensor([datum.gt for datum in data])
        index = [datum.index for datum in data]
        return MNISTMiniBatch(batch_size, input, gt, index)


class MNISTMiniBatch(IMiniBatch):

    def __init__(self, batch_size, input, gt, batch_indices=None) -> None:
        self.batch_size = batch_size
        self.input:Shaped[Tensor, "_ C H W"] = input
        self.gt:Shaped[Tensor, "_"] = gt
        self.logits:Float[Tensor, "_ num_class"] = None
        self.pred:Shaped[Tensor, "_"] = None
        self.correct:Shaped[Tensor, "_"] = None
        self.batch_indices:List[int] = batch_indices


class NLLLoss(ILoss):

    def forward(self, mb: MNISTMiniBatch) -> FloatScalarTensor:
        return F.nll_loss(mb.logits, mb.gt)


class MiniBatchAccuracy(IMiniBatchMetric):

    higher_better = True

    def forward(self, mb: MNISTMiniBatch) -> MiniBatchFloats:
        mb.pred = mb.logits.argmax(dim=1)
        mb.correct = mb.pred.eq(mb.gt)
        return mb.correct


class MNISTDataset(IDataset):

    datum_type = MNISTDatum

    def __init__(self, root: str, split: str = None) -> None:
        super().__init__(root, split)
        train  =  split == "train"
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(root, train=train, transform=transform, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> IDatum:
        input, gt = self.dataset[index]
        return MNISTDatum(input, gt, index)


class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, mb: MNISTMiniBatch):
        x = mb.input
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        mb.logits = F.log_softmax(x, dim=1)

    def adadelta_steplr(self, lr=1.0, sched_step=1, sched_gamma=0.7):
        optimizer = optim.Adadelta(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
        return TopLevelModule(
            model=self,
            optim=optimizer,
            lr_scheduler=scheduler,
        )


@cli.command()
def eval(
    ckpt: str = "",
    device:str="cuda:0",
):

    side_effects = [
        TqdmSideEffect(),
    ]

    model = TopLevelModule(
        BasicNet()
    ).to(device)

    if ckpt:
        model.load(ckpt)

    eval_loop = EvalLoop(
        model=model,
        dataset=MNISTDataset("./mnist_data", "val"),
        metrics=dict(
            acc_sumavg=AverageMetric(MiniBatchAccuracy()),
        ),
        side_effects=side_effects,
        batch_size=601,
    )

    eval_loop()


@cli.command()
def train(
    desc:str="",
    lr:float=1.0,
    sched_step:int=1,
    sched_gamma:float=0.7,
    train_batch:int=64,
    eval_batch:int=1000,
    device:str="cuda:0",
    seed:int=42,
):
    manual_seed(seed, strict=True)

    experiment = "mnist-basicnet-adadelta-steplr"
    experiment = f"{experiment}-{desc}" if desc else experiment

    model = BasicNet().adadelta_steplr(
        lr=lr,
        sched_step=sched_step,
        sched_gamma=sched_gamma,
    ).to(device)

    effective_train_batch = train_batch * len(model.distributed_devices)

    side_effects = [
        AimSideEffect(
            repo="./",
            experiment=experiment,
            hparams_dict=dict(
                seed=seed,
                lr=lr,
                sched_step=sched_step,
                sched_gamma=sched_gamma,
                train_batch=effective_train_batch,
            )
        ),
        TqdmSideEffect(
            postfix_keys=['nll', "lr"]
        ),
        SaveBestModelsSideEffect(
            metric_name="accuracy",
            num_max_keep=3,
            save_path_prefix="./basicnet-mnist-"
        ),
    ]

    eval_loop = EvalLoop(
        model=model,
        dataset=MNISTDataset("./mnist_data", "val"),
        batch_size=eval_batch,
        metrics=dict(
            accuracy=AverageMetric(MiniBatchAccuracy())
        ),
        side_effects=side_effects,
    )

    train_loop = TrainLoop(
        model=model,
        dataset=MNISTDataset("./mnist_data", "train"),
        batch_size=train_batch,
        losses=dict(
            nll=NLLLoss()
        ),
        metrics=dict(
            acc=MiniBatchAccuracy(),
            lr=LearningRate(),
        ),
        epochs=5,
        epoch_per_eval=1,
        iter_per_epoch=10,
        eval_loop=eval_loop,
        side_effects=side_effects,
    )

    train_loop()
    model.save("./basicnet_mnist_v0.pt.gz")


if __name__ == "__main__":
    cli()
