import lightning
import torchinfo
from lightning.pytorch import Callback
from lightning.pytorch.utilities.model_summary import DeepSpeedSummary, summarize
from typing import Any, Dict, List, Tuple, Union


class TorchInfoSummary(Callback):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def on_fit_start(self, trainer, pl_module):
        torchinfo.summary(pl_module, **self.kwargs)
