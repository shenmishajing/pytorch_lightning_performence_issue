from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import nn


class MMDetModelAdapter(LightningModule, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(
            self,
            model: nn.Module,
            *args, **kwargs
    ):
        """
        To show a metric in the progressbar a list of tupels can be provided for metrics_keys_to_log_to_prog_bar, the first
        entry has to be the name of the metric to log and the second entry the display name in the progressbar. By default the
        mAP is logged to the progressbar.
        """
        super().__init__(*args, **kwargs)
        self.model = model

    @staticmethod
    def add_prefix(log_dict, prefix = 'train/'):
        return {f'{prefix}{k}': v for k, v in log_dict.items()}

    def log(self, *args, batch_size = None, **kwargs):
        if batch_size is None and hasattr(self, 'batch_size') and self.batch_size is not None:
            batch_size = self.batch_size
        super().log(*args, batch_size = batch_size, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        self.batch_size = batch['img'].shape[0]
        outputs = self.model.train_step(data = batch, optimizer = None)
        return outputs['loss']
