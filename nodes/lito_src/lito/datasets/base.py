#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the base data module.

from functools import partial
import os
import random
import typing as T
import warnings

import lightning.pytorch as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import numpy as np
from packaging import version

import torch.utils.data
from torch.utils.data import DataLoader, Dataset, IterableDataset

from lito.script_utils.config_utils import get_obj_from_str, instantiate_from_config

# IMPORTANT: pytorch-lightning's default strategy when using multi-node
# ddp is to use pytorch's distributed_sampler with
# num_replica = num_nodes * num_gpu_per_node.

# Our strategy of global_sharding:
# 1. each node loads all shards, by setting `files_per_node` = -1
# 2. dataset object on each node loads all data locally on each node (so all shards)
# 3. when creating dataloader, use distributed sampler that divides data with world_size (ngpus * num_nodes)

# Our strategy of multinode_local_sharding:
# 1. each node loads a subset of shards, controlled by `files_per_node`
# 2. dataset object on each node loads all data locally on each node
# 3. when creating dataloader, use customized distributed sampler that divides data with ngpus on each node

# Our strategy of full_sharding:
# 1. each process on a node loads a subset of shards
# 2. when creating dataloader, use typical dataloader with typical batching


def _get_dataloader(
    method: str,
    dataset: Dataset,
    local_rank: int,
    global_rank: int,
    num_nodes: int,
    num_processes_per_node: int,
    world_size: int,
    batch_size: int,
    shuffle: bool,
    collate_fn=None,
    num_workers: int = 4,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    persistent_workers: bool = False,
    in_order: bool = True,
    wds_shuffle_buffer_size: int = 0,
    wds_collate_method: str = "list",
    wds_batch_size: int = None,
    prebatched: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Wrap a dataset with a dataloader.
    The sampler strategy is described above.

    Args:
        method:
            'global_sharding':
                num_replica = world_size
            'multinode_local_sharding'
                num_replica = num_processes_per_node
            'full_sharding':
                use typical dataloader
        dataset:
            dataset to be wrapped
        local_rank:
            local rank inside each node
        global_rank:
            global rank of the current process
        num_nodes:
            total number of training nodes
        num_processes_per_node:
            number of processes (gpu) per node
        world_size:
            total number of processes
        batch_size:
            local batch size of each process
        shuffle:
            whether to shuffle the data
        collate_fn:
            collate function, None: use default collate_fn from pytorch.
        num_workers:
            number of preloading workers
        timeout:
            if positive, the timeout value for collecting a batch from workers.
        persistent_workers:
            If True, the data loader will not shut down the worker processes
            after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
        in_order:
            If False, the data loader will not enforce that batches are returned
            in a first-in, first-out order. Only applies when num_workers > 0.
            Only available after torch 2.6.
        wds_shuffle_buffer_size:
            if > 0, we will use webdataset's wrapper of pytorch dataloader
            to unbatch, mix batches from different dataloader workers, and rebatch.
        wds_batch_size:
            if None, same as batch_size.
        prebatched:
            whether the dataset already outputs batch, instead of a single sample

    Returns:
        dataloader
    """
    # set RANK if not set (it seems lightning only sets WORLD_SIZE, LOCAL_RANK, NODE_RANK)
    if os.environ.get("RANK", None) is None:
        os.environ["RANK"] = str(global_rank)

    if os.environ.get("WDS_SHOW_SEED", None) is None:
        os.environ["WDS_SHOW_SEED"] = str(1)

    # a simple and more isolated pathway for wdset
    if method == "wdset":
        import webdataset

        rebatch = wds_shuffle_buffer_size is not None and wds_shuffle_buffer_size > 0

        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            dataloader = webdataset.WebLoader(
                dataset=dataset,
                batch_size=None if (rebatch or prebatched) else batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,  # if rebatch, not called
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=partial(
                    worker_init_function,
                    rank=global_rank,
                    world_size=world_size,
                    num_workers=num_workers,
                ),
                persistent_workers=persistent_workers,
                in_order=False,  # pop batch whenever ready. We want this so we do not wait for a worker to download tar
                **kwargs,
            )
        else:
            dataloader = webdataset.WebLoader(
                dataset=dataset,
                batch_size=None if (rebatch or prebatched) else batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,  # if rebatch, not called
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=partial(
                    worker_init_function,
                    rank=global_rank,
                    world_size=world_size,
                    num_workers=num_workers,
                ),
                persistent_workers=persistent_workers,
                **kwargs,
            )

        # whether to unbatch batches from different workers, shuffle, and the rebatch
        # to improve diversity of samples in batches
        if rebatch:
            # the line mixes samples from different dataloading workers:
            # 1. store each samples from the batch into the a buffer of n number of samples
            # 2. randomly select b samples from the buffer to create new batch
            if wds_collate_method == "list":
                dataloader = (
                    dataloader.unlisted()
                    .shuffle(wds_shuffle_buffer_size)
                    .listed(
                        batchsize=batch_size,
                        partial=not drop_last,
                    )
                )
            else:
                raise NotImplementedError

        return dataloader

    if wds_shuffle_buffer_size is not None and wds_shuffle_buffer_size > 0:
        import webdataset

        dataloader_class = webdataset.WebLoader
    else:
        dataloader_class = torch.utils.data.DataLoader

    is_distributed = world_size > 1
    if not is_distributed or method == "full_sharding":
        # print(f'using full sharding, batch size = {batch_size}')
        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            dataloader = dataloader_class(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=None
                if (dataloader_class == torch.utils.data.DataLoader) and isinstance(dataset, IterableDataset)
                else shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                persistent_workers=persistent_workers if num_workers > 0 else None,
                multiprocessing_context="spawn" if num_workers > 0 else None,
                worker_init_fn=partial(worker_init_function, rank=global_rank),
                in_order=in_order,
                **kwargs,
            )
        else:
            if not in_order:
                warnings.warn("in_order = False is only available after torch 2.6.0")
            dataloader = dataloader_class(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=None
                if (dataloader_class == torch.utils.data.DataLoader) and isinstance(dataset, IterableDataset)
                else shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                persistent_workers=persistent_workers if num_workers > 0 else None,
                multiprocessing_context="spawn" if num_workers > 0 else None,
                worker_init_fn=partial(worker_init_function, rank=global_rank),
                **kwargs,
            )
        # print(f'total number of batches: {len(dataloader)}')
        # return dataloader
    else:
        if method == "global_sharding":
            # print(f'using global sharding, batch size = {batch_size}')
            sampler = torch.utils.data.DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=shuffle,
                seed=0,
                drop_last=drop_last,
            )
        elif method == "multinode_local_sharding":
            sampler = torch.utils.data.DistributedSampler(
                dataset=dataset,
                num_replicas=num_processes_per_node,
                rank=local_rank,
                shuffle=shuffle,
                seed=0,
                drop_last=drop_last,
            )
        else:
            raise NotImplementedError

        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            dataloader = dataloader_class(
                batch_size=batch_size,
                dataset=dataset,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                persistent_workers=persistent_workers,
                multiprocessing_context="spawn" if num_workers > 0 else None,
                worker_init_fn=partial(worker_init_function, rank=global_rank),
                in_order=in_order,
                **kwargs,
            )
        else:
            if not in_order:
                warnings.warn("in_order = False is only available after torch 2.6.0")
            dataloader = dataloader_class(
                batch_size=batch_size,
                dataset=dataset,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                persistent_workers=persistent_workers,
                multiprocessing_context="spawn" if num_workers > 0 else None,
                worker_init_fn=partial(worker_init_function, rank=global_rank),
                **kwargs,
            )

    if wds_shuffle_buffer_size is not None and wds_shuffle_buffer_size > 0:
        import webdataset

        assert isinstance(dataloader, webdataset.WebLoader)

        if wds_batch_size is None:
            wds_batch_size = batch_size
        assert wds_batch_size is not None

        # the line mixes samples from different dataloading workers:
        # 1. store each samples from the batch into the a buffer of n number of samples
        # 2. randomly select b samples from the buffer to create new batch
        if wds_collate_method == "list":
            dataloader = (
                dataloader.unlisted()
                .shuffle(wds_shuffle_buffer_size)
                .listed(
                    batchsize=wds_batch_size,
                    partial=not drop_last,
                )
            )
        else:
            raise NotImplementedError

    return dataloader


def worker_init_function(
    worker_id: int,
    rank: T.Optional[int] = None,
    world_size: int = None,
    num_workers: int = None,
) -> None:
    """The worker_init_fn for dataloader to create workers.
    This is called after dataset is copied (ie, unpickled).

    The init function makes sure:
    - every worker on every node gets different random seed
    - every worker gets different random seed for each epoch
    - reproducible random seeds

    See: https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.pl_worker_init_function
    """
    # get global rank
    global_rank = rank if rank is not None else rank_zero_only.rank

    os.environ["WORKER"] = str(worker_id)
    os.environ["NUM_WORKERS"] = str(num_workers)

    # limit opencv using multithreading and avoid cpu contention
    import cv2

    cv2.setNumThreads(0)
    # print(f'(rank {global_rank}, worker {worker_id}), original max_num_threads = {torch.get_num_threads()}')
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if "EPOCH" in os.environ:
        epoch = int(os.environ["EPOCH"])
    else:
        epoch = 0

    # modified from pytorch lightning
    process_seed = torch.initial_seed()
    # see: https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
    # base_seed is already different every epoch
    base_seed = process_seed - worker_id

    ss = np.random.SeedSequence([base_seed, worker_id, global_rank, epoch])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss, dset_ss = ss.spawn(3)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)

    try:
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        dset_seed = dset_ss.generate_state(1, dtype=np.uint64)[0]
        print(f"seeding dataset on global rank {global_rank} worker id {worker_id} with {dset_seed}")
        # dataset.set_seed(dset_seed)
    except:
        traceback.print_exc()


class BaseDataModule(L.LightningDataModule):
    """ """

    def __init__(
        self,
        method: str,
        train_dataset_config: T.Dict[str, T.Any],
        valid_dataset_config: T.Optional[T.Dict[str, T.Any]],
        test_dataset_config: T.Optional[T.Dict[str, T.Any]],
        predict_dataset_config: T.Optional[T.Dict[str, T.Any]],
        train_dataloader_config: T.Dict[str, T.Any],
        valid_dataloader_config: T.Optional[T.Dict[str, T.Any]],
        test_dataloader_config: T.Optional[T.Dict[str, T.Any]],
        predict_dataloader_config: T.Optional[T.Dict[str, T.Any]],
    ):
        super().__init__()

        # make sure `prepare_data` function is run on local_rank = 0 of each node
        self.prepare_data_per_node = True

        self.method = method

        if isinstance(valid_dataset_config, (list, tuple)):
            assert isinstance(valid_dataloader_config, (list, tuple))
            assert len(valid_dataloader_config) == len(valid_dataset_config)

        self.train_dataset_config = train_dataset_config  # default(train_dataset_config, dict())
        self.valid_dataset_config = valid_dataset_config  # default(valid_dataset_config, dict())
        self.test_dataset_config = test_dataset_config  # default(test_dataset_config, dict())
        self.predict_dataset_config = predict_dataset_config  # default(predict_dataset_config, dict())

        self.train_dataloader_config = train_dataloader_config  # default(train_dataloader_config, dict())
        self.valid_dataloader_config = valid_dataloader_config  # default(valid_dataloader_config, dict())
        self.test_dataloader_config = test_dataloader_config  # default(test_dataloader_config, dict())
        self.predict_dataloader_config = predict_dataloader_config  # default(predict_dataloader_config, dict())

        self._get_collate_fn()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def _get_collate_fn(self):
        for dloader_config in [
            self.train_dataloader_config,
            self.valid_dataloader_config,
            self.test_dataloader_config,
            self.predict_dataloader_config,
        ]:
            if dloader_config is None:
                continue
            if isinstance(dloader_config, dict):
                dloader_configs = [dloader_config]
            elif isinstance(dloader_config, (list, tuple)):
                dloader_configs = dloader_config
            else:
                raise NotImplementedError

            for i in range(len(dloader_configs)):
                if dloader_configs[i].get("collate_fn", None) is None:
                    continue

                collate_fn = dloader_configs[i]["collate_fn"]
                if isinstance(collate_fn, str):
                    # pass collate_fn as a function
                    collate_fn = get_obj_from_str(collate_fn)
                elif isinstance(collate_fn, dict):
                    # pass collate_fn as a callable object
                    assert "target" in collate_fn, f"{collate_fn=}"
                    collate_fn = instantiate_from_config(collate_fn)
                else:
                    raise NotImplementedError

                dloader_configs[i]["collate_fn"] = collate_fn

    def prepare_data(self):
        """
        Only called by processes with local_rank = 0.
        Override the function to download dataset.
        """
        node_rank = self.trainer.node_rank
        num_nodes = self.trainer.num_nodes

    def setup(self, stage: str):
        """
        Create dataset, can save to self
        """

        local_rank = self.trainer.local_rank
        global_rank = self.trainer.global_rank
        node_rank = self.trainer.node_rank
        num_nodes = self.trainer.num_nodes
        num_devices = self.trainer.num_devices

        # # set RANK if not set (it seems lightning only sets WORLD_SIZE, LOCAL_RANK, NODE_RANK)
        # if os.environ.get("RANK", None) is None:
        #     os.environ["RANK"] = str(global_rank)

    def _get_env_info(self):
        if self.trainer is not None:
            return dict(
                local_rank=self.trainer.local_rank,
                global_rank=self.trainer.global_rank,
                num_nodes=self.trainer.num_nodes,
                num_processes_per_node=self.trainer.num_devices,
                world_size=self.trainer.world_size,
            )
        else:
            return dict(
                local_rank=0,
                global_rank=0,
                num_nodes=1,
                num_processes_per_node=1,
                world_size=1,
            )

    def train_dataloader(self):
        """Wrap train_dataset with dataloader and return it."""
        assert self.train_dataset is not None
        return _get_dataloader(
            method=self.method,
            dataset=self.train_dataset,
            **self._get_env_info(),
            **self.train_dataloader_config,
        )

    def val_dataloader(self):
        """Wrap val_dataset with dataloader and return it."""
        assert self.val_dataset is not None
        if isinstance(self.val_dataset, (tuple, list)):
            assert len(self.val_dataset) == len(self.valid_dataloader_config)
            dataloaders = []
            for i in range(len(self.val_dataset)):
                dset = self.val_dataset[i]
                dloader_config = self.valid_dataloader_config[i]
                dloader = _get_dataloader(
                    method=self.method,
                    dataset=dset,
                    **self._get_env_info(),
                    **dloader_config,
                )
                dataloaders.append(dloader)

            return dataloaders
        else:
            return _get_dataloader(
                method=self.method,
                dataset=self.val_dataset,
                **self._get_env_info(),
                **self.valid_dataloader_config,
            )

    def test_dataloader(self):
        """Wrap test_dataset with dataloader and return it.
        Should not be called if test_dataset is None.
        """
        assert self.test_dataset is not None
        return _get_dataloader(
            method=self.method,
            dataset=self.test_dataset,
            **self._get_env_info(),
            **self.test_dataloader_config,
        )

    def predict_dataloader(self):
        """Wrap predict_dataset with dataloader and return it.
        Should not be called if predict_dataset is None.
        """
        assert self.predict_dataset is not None
        return _get_dataloader(
            method=self.method,
            dataset=self.predict_dataset,
            **self._get_env_info(),
            **self.predict_dataloader_config,
        )


def list_collate(batch: T.List[T.Any]) -> T.List[T.Any]:
    return batch
