from argparse import ArgumentParser

import torch
from torch.nn import CrossEntropyLoss

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar

from torchmetrics.classification import MulticlassAccuracy

from aim.pytorch_lightning import AimLogger

from dgcnn.nn.util import Sequential
from dgcnn.nn import ClassifierHead
from dgcnn.data import ModelNet40DataModule


class DGCNNClassifier(LightningModule):

    def __init__(self, num_classes, **kwargs):
        super().__init__()

        self.model = Sequential(
            # todo: embedding module
            ClassifierHead(num_classes=num_classes)
        )

        self.loss = CrossEntropyLoss(label_smoothing=0.2)

        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')

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
        self.log_dict({
            "val_loss": loss,
            "val_acc": self.val_accuracy
        }, batch_size=batch.y.shape[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def main(num_epochs, **args):
    # Enables the use of tensor cores
    torch.set_float32_matmul_precision('medium')

    dataset = ModelNet40DataModule(**args)

    classifier = DGCNNClassifier(num_classes=40)

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
            RichProgressBar()
        ],
        logger=logger
    )

    trainer.fit(
        classifier,
        dataset.train_dataloader(),
        dataset.val_dataloader()
    )

    pass


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=250)
    ModelNet40DataModule.add_args(parser.add_argument_group('Dataset'))

    main(**vars(parser.parse_args()))
