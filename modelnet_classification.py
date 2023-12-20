import inspect
from argparse import ArgumentParser

import torch
from torch.nn import CrossEntropyLoss

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor

from torchmetrics.classification import MulticlassAccuracy

from aim.pytorch_lightning import AimLogger

from dgcnn.nn.util import Sequential
from dgcnn.nn import ClassifierHead, DynamicEdgeConv, TorchInfoSummary
from dgcnn.data import ModelNet40DataModule


def invoke_with_matching_kwargs(function, *args, **kwargs):
    parameter_names = list(inspect.signature(function).parameters.keys())
    named_parameters = dict((name, kwargs[name]) for name in parameter_names if name in kwargs)
    return function(*args, **named_parameters)


class DGCNNClassifier(LightningModule):

    def __init__(self, num_classes: int, num_epochs: int, include_normals: bool = False, **kwargs):
        super().__init__()

        self.model = Sequential(
            invoke_with_matching_kwargs(DynamicEdgeConv, in_channels=6 if include_normals else 3, **kwargs),
            ClassifierHead(num_classes=num_classes, embedding_features=2048)
        )

        self.num_epochs = num_epochs

        self.loss = CrossEntropyLoss(label_smoothing=0.2)

        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_accuracy_balanced = MulticlassAccuracy(num_classes=num_classes, average='macro')

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        prediction = self.forward(**batch.to_dict())
        loss = self.loss(prediction, batch.y)
        self.train_accuracy(prediction, batch.y)
        self.log_dict({
            "train_loss": loss,
            "train_acc": self.train_accuracy
        }, batch_size=batch.y.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self.forward(**batch.to_dict())
        loss = self.loss(prediction, batch.y)
        self.val_accuracy(prediction, batch.y)
        self.val_accuracy_balanced(prediction, batch.y)
        self.log_dict({
            "val_loss": loss,
            "val_acc": self.val_accuracy,
            "val_acc_balanced": self.val_accuracy_balanced,
        }, batch_size=batch.y.shape[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs, 1e-4)
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_args(parser):
        DynamicEdgeConv.add_args(parser)


def main(num_epochs, **args):
    # Enables the use of tensor cores
    torch.set_float32_matmul_precision('medium')

    dataset = invoke_with_matching_kwargs(ModelNet40DataModule, **args)

    classifier = DGCNNClassifier(
        num_classes=40,
        num_epochs=num_epochs,
        **args
    )

    logger = AimLogger(
        experiment='dgcnn-replication',
        train_metric_prefix='train_',
        test_metric_prefix='test_',
        val_metric_prefix='val_',
    )
    logger.log_hyperparams(args)

    trainer = Trainer(
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        precision='bf16-mixed',
        callbacks=[
            TorchInfoSummary(depth=6),
            RichProgressBar(),
            LearningRateMonitor(),
        ],
        logger=logger
    )

    trainer.fit(
        classifier,
        dataset.train_dataloader(),
        dataset.val_dataloader()
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=250)
    ModelNet40DataModule.add_args(parser.add_argument_group('Dataset'))
    DGCNNClassifier.add_args(parser.add_argument_group('Model'))

    main(**vars(parser.parse_args()))
