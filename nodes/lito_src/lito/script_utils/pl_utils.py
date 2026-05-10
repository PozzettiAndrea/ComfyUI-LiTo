#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the utils to use pytorch-lightning.
import copy
from datetime import datetime
import os
import shlex
import subprocess
import traceback
import typing as T

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from tqdm import tqdm

import torch.optim

from lito.datasets import base
from lito.script_utils.config_utils import instantiate_from_config

# IMPORTANT: pytorch-lightning's default strategy when using multi-node
# ddp is to use pytorch's distributed_sampler with
# num_replica = num_nodes * num_gpu_per_node.


def launch_tensorboard(log_dir=None, port=None):
    """
    Launches a tensorboard process in the background.

    Args:
        log_dir:
            where the tensorboard logs to be loaded. uses ARTIFACT_DIR if None
        port:
            which port to open tensorboard. uses BOLT_TAB_TENSORBOARD env var if None
    """
    if port is None:
        if "TENSORBOARD_PORT" in os.environ:
            port = os.environ["TENSORBOARD_PORT"]
        else:
            port = "7788"
            print("TENSORBOARD_PORT not set, use 7788")

    if log_dir is None:
        log_dir = "artifacts"

    command = f"tensorboard --logdir {log_dir} --port {port} --bind_all --samples_per_plugin=images=100 "
    print("launching tensorboard via: ", command)
    subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


def get_wandb_api_key() -> str:
    """Get wandb api key stored in the secrets file."""
    if not os.path.exists("secrets.txt"):
        raise ValueError("You need to create a file named secrets.txt with your wandb key in the root directory.")

    with open("secrets.txt") as f:
        wandb_api_key = f.read().strip()
        os.environ["WANDB_API_KEY"] = wandb_api_key
        return wandb_api_key


def get_environment_info(config: T.Union[T.Dict[str, T.Any]]):
    """
    Gather information about the cluster environment.

    Notes:
        We will use the role name `trainer` to indicate a
        iris node to be a training job.

    Returns:
        is_bolt:
            whether we are on a noninteractive bolt environment
        is_iris:
            whether we use multinode with iris controller
        node_rank:
            int,  node rank of the current process (multiple gpus on same node share same node_rank)
        master_ip:
            str, master node ip address
        master_port:
            int, master node port
        num_trainers:
            int, total number of trainers
        num_nodes:
            int, total number of nodes when we call the function
        local_artifact_dir:
            artifact dir locally on each node
        checkpoint_dir:
            checkpoint dir to save the model, tensorboard, etc.
        parent_bolt_id:
            None or bolt_id to the parent node
        parent_artifact_dir:
            None or parent's artifact_dir
        num_retries:
            number of retries on bolt
        recovery_dir:
            dir used for job recovery (things saved here will be
            restored when job retries)
    """
    is_bolt = False
    is_iris = False
    node_rank = 0
    master_ip = "127.0.0.1"
    master_port = 29001
    num_trainers = 1
    num_nodes = config["num_nodes"]
    devices = config.get("devices", "auto")
    parent_bolt_id = None
    parent_artifact_dir = None
    num_retries = 0
    is_interactive = True

    # candidate of dir if not set in config
    _local_artifact_dir = "artifacts"
    _checkpoint_dir = _local_artifact_dir  # where to save the
    _recovery_dir = "checkpoints"

    if config["artifact_dir"] is None:
        local_artifact_dir = _local_artifact_dir
    else:
        local_artifact_dir = config["artifact_dir"]

    if config["checkpoint_dir"] is None:
        checkpoint_dir = _checkpoint_dir
    else:
        checkpoint_dir = config["checkpoint_dir"]

    if config.get("recovery_dir", None) is None:
        recovery_dir = _recovery_dir
    else:
        recovery_dir = config["recovery_dir"]

    return dict(
        is_bolt=is_bolt,
        is_iris=is_iris,
        is_interactive=is_interactive,
        node_rank=int(node_rank),
        master_ip=str(master_ip),
        master_port=int(master_port),
        num_trainers=int(num_trainers),
        num_nodes=int(num_nodes),
        devices=devices,
        local_artifact_dir=local_artifact_dir,
        checkpoint_dir=checkpoint_dir,
        parent_bolt_id=str(parent_bolt_id),
        parent_artifact_dir=str(parent_artifact_dir),
        num_retries=num_retries,
        recovery_dir=recovery_dir,
    )


class CustomClusterEnvironment(LightningEnvironment):
    """
    See https://lightning.ai/docs/pytorch/stable/clouds/cluster_expert.html

    If "LOCAL_RANK" not in the environment variable,
    pytorch lightning will assume it should manage processes and launch them.
    Since iris controller creates `IRISCTL_LOCAL_RANK` and bolt does not create `LOCAL_RANK`,
    we let pytorch lightning control the process.

    """

    def __init__(
        self,
        node_rank: int,
        main_address: str,
        main_port: int,
    ):
        super().__init__()
        self._node_rank = int(node_rank)
        self._main_address = str(main_address)
        self._main_port = int(main_port)

    @property
    def main_address(self) -> str:
        """The main address through which all processes connect and communicate."""
        return self._main_address

    @property
    def main_port(self) -> int:
        """An open and configured port in the main node for processes communication."""
        return self._main_port

    def node_rank(self) -> int:
        """The rank (index) of the node on which the current process runs."""
        return self._node_rank


def setup_loggers(
    config: T.Dict[str, T.Any], env_info: T.Dict[str, T.Any], is_interactive: bool
) -> T.List[T.Union[TensorBoardLogger, WandbLogger]]:
    """Set up tensorboard and wandb and return a list of loggers."""

    # create tensorboard logger and run tensorboard on all nodes
    # tensorboard log are saved to local artifact dir (for speed)
    tensorboard_dir = env_info["local_artifact_dir"]
    os.makedirs(tensorboard_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name=config.get("name", "lightning_logs"),
    )
    launch_tensorboard(tensorboard_dir)
    loggers = [tb_logger]

    # wandb
    # batch job on bolt (iris or not)
    # create wandb on node 0 on non-interactive job only
    if env_info["node_rank"] == 0 and config["wandb_project"] is not None and not is_interactive:
        # I choose to output error if cannot get wandb key
        get_wandb_api_key()

        try:
            wandb_name = config["wandb_name"]
            wandb_logger = WandbLogger(
                name=wandb_name,
                project=config["wandb_project"],
                entity=config["wandb_entity"],
                tags=[env_info["parent_bolt_id"]],
            )
            print(f"wandb name: {config['wandb_name']}")
        except:
            # try again with different name
            print(f"Failed to use wandb name: {config['wandb_name']}")
            now = datetime.now().strftime("%m-%d_%H:%M:%S")
            if config["wandb_name"] is None:
                config["wandb_name"] = str(now)
            else:
                config["wandb_name"] = f"{config['wandb_name']}_{str(now)}"

            wandb_name = config["wandb_name"]
            wandb_logger = WandbLogger(
                name=config["wandb_name"],
                project=config["wandb_project"],
                entity=config["wandb_entity"],
                tags=[env_info["parent_bolt_id"]],
            )
            print(f"Successfully used wandb name: {config['wandb_name']}")

        # save config to wandn
        wandb_logger.log_hyperparams(config)
        loggers.append(wandb_logger)

    return loggers


def setup_plugins(config: T.Dict[str, T.Any], env_info: T.Dict[str, T.Any]) -> T.List[T.Any]:
    cluster_env = CustomClusterEnvironment(
        node_rank=env_info["node_rank"],
        main_address=env_info["master_ip"],
        main_port=env_info["master_port"],
    )
    return [cluster_env]


def setup_callbacks(config: T.Dict[str, T.Any], env_info: T.Dict[str, T.Any]):
    """
    Prepare callback function to perform tasks during the course of training.
    E.g., checkpoint, progress bar, etc.
    """
    callbacks = []
    # checkpoint
    save_freq = config["plm_config"]["params"]["optim_config"]["val_check_interval"]
    if config["fast_dev_run"] > 0:
        save_freq = min(save_freq, config["fast_dev_run"])
    print(
        f"node rank {env_info['node_rank']} will "
        f"save model to {env_info['checkpoint_dir']} every {save_freq} training iters."
    )

    if config.get("model_saving_monitor_loss", None) is not None:
        monitor_loss = config["model_saving_monitor_loss"]
    else:
        monitor_loss = None  # "loss/total_loss"

    os.makedirs(env_info["checkpoint_dir"], exist_ok=True)
    save_top_k = config.get("save_top_k", 50)
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_loss,
        save_last=True,
        save_top_k=save_top_k,
        dirpath=env_info["checkpoint_dir"],
        filename=f"model-best-iter{{step:08d}}-loss{{{monitor_loss}:.8f}}",
        auto_insert_metric_name=False,
        mode="min",
        every_n_train_steps=max(1, save_freq),
    )
    callbacks.append(checkpoint_callback)

    # learning rate
    lr_callback = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_callback)

    # progress bar
    progress_bar = TQDMProgressBar(refresh_rate=1)
    callbacks.append(progress_bar)

    return callbacks


def get_datamodule(config) -> base.BaseDataModule:
    data_config = config["data_config"]
    # create data module specific to the dataset
    data_module = instantiate_from_config(data_config)
    return data_module


def scale_lr_by_batch_size(
    lr_256: float,
    batch_size: int,
    world_size: int,
) -> float:
    """
    Adjust learning rate based on the linear scaling rule from Goyal et al.

    Args:
        lr_256:
            defines the lr when global_batch_size = 256
        batch_size:
            the local batch size on each gpu
        world_size:
            the total number of processes (e.g., num_gpus * num_nodes)
    """
    scaled_lr = lr_256 * batch_size * world_size / 256.0
    return scaled_lr


def plot_lr_schedule(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_iters: int,
):
    """
    Plot the lr schedule in console and return the learning rates.
    """
    optimizer = copy.deepcopy(optimizer)
    scheduler = copy.deepcopy(scheduler)
    scheduler.optimizer = optimizer

    lrs = []
    iters = []

    for i in tqdm(range(num_iters)):
        try:
            optimizer.step()  # important
            scheduler.step()

            # Store current learning rate
            # current_lr = scheduler.get_last_lr()[0]
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            iters.append(i + 1)
        except:
            traceback.print_exc()
            break

    print(f"finished computing lr, plotting", flush=True)
    import plotext as plt

    # don't want to draw too many points
    num_jumps = max(1, len(iters) // 1000)
    sub_iters = iters[::num_jumps]
    sub_lrs = lrs[::num_jumps]

    plt.plot(sub_iters, sub_lrs)
    plt.xscale("log")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()

    return iters, lrs
