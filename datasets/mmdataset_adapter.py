import copy
import string
from collections.abc import Mapping, Sequence
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from mmcv.parallel.data_container import DataContainer
from mmdet.datasets import build_dataset
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate


class MMDetDataSetAdapter(LightningDataModule):
    def __init__(self, dataset_cfg, split_format_to = 'ann_file', data_loader_config = None):
        super().__init__()
        if data_loader_config is None:
            self.data_loader_config = {}
        else:
            self.data_loader_config = data_loader_config

        self.train_dataset = self.val_dataset = None
        self.dataset_cfg = dataset_cfg
        self.split_format_to = split_format_to if split_format_to is None or isinstance(split_format_to, list) else [split_format_to]

    def setup(self, stage = None):
        if stage in [None, 'fit']:
            self.train_dataset = self._build_data_set('train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = self._build_data_set('val')

    def train_dataloader(self):
        return self._build_data_loader(self.train_dataset, shuffle = True)

    def val_dataloader(self):
        return self._build_data_loader(self.val_dataset)

    def _build_data_loader(self, dataset, shuffle: Optional[bool] = False, collate_fn: Optional[Callable] = None):
        def dataloader(ds, cl_fn) -> DataLoader:
            return DataLoader(ds, shuffle = shuffle and not isinstance(ds, IterableDataset), collate_fn = cl_fn, **self.data_loader_config)

        if collate_fn is None:
            collate_fn = self.collate
        if isinstance(dataset, Mapping):
            return {key: dataloader(ds, cl_fn = collate_fn[key] if isinstance(collate_fn, Mapping) else collate_fn) for key, ds in
                    dataset.items()}
        if isinstance(dataset, Sequence):
            return [dataloader(dataset[i], cl_fn = collate_fn[i] if isinstance(collate_fn, Sequence) else collate_fn) for i in
                    range(len(dataset))]
        return dataloader(dataset, cl_fn = collate_fn)

    @staticmethod
    def collate(batch):
        """Puts each data field into a tensor/DataContainer with outer dimension
        batch size.

        Extend default_collate to add support for
        :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

        1. cpu_only = True, e.g., meta data
        2. cpu_only = False, stack = True, e.g., images tensors
        3. cpu_only = False, stack = False, e.g., gt bboxes
        """
        if not isinstance(batch, Sequence):
            raise TypeError(f'{batch.dtype} is not supported.')
        if isinstance(batch[0], DataContainer):
            if batch[0].stack:
                assert isinstance(batch[0].data, torch.Tensor)

                if batch[0].pad_dims is not None:
                    ndim = batch[0].dim()
                    assert ndim > batch[0].pad_dims
                    max_shape = [0 for _ in range(batch[0].pad_dims)]
                    for dim in range(1, batch[0].pad_dims + 1):
                        max_shape[dim - 1] = batch[0].size(-dim)
                    for sample in batch:
                        for dim in range(0, ndim - batch[0].pad_dims):
                            assert batch[0].size(dim) == sample.size(dim)
                        for dim in range(1, batch[0].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1], sample.size(-dim))
                    padded_samples = []
                    for sample in batch:
                        pad = [0 for _ in range(batch[0].pad_dims * 2)]
                        for dim in range(1, batch[0].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(F.pad(sample.data, pad, value = sample.padding_value))
                    return default_collate(padded_samples)
                elif batch[0].pad_dims is None:
                    return default_collate([sample.data for sample in batch])
                else:
                    raise ValueError('pad_dims should be either None or integers (1-3)')
            return [sample.data for sample in batch]
        elif isinstance(batch[0], Sequence):
            transposed = zip(*batch)
            return [MMDetDataSetAdapter.collate(samples) for samples in transposed]
        elif isinstance(batch[0], Mapping):
            return {key: MMDetDataSetAdapter.collate([d[key] for d in batch]) for key in batch[0]}
        else:
            return default_collate(batch)

    def _build_data_set(self, split):
        cfg = copy.deepcopy(self.dataset_cfg)
        if self.split_format_to is None:
            cfg['split'] = split
        else:
            for s in self.split_format_to:
                cfg[s] = string.Template(cfg[s]).safe_substitute(split = split)
        return build_dataset(cfg)
