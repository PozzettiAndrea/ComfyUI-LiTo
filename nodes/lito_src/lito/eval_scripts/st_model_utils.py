#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# The script computes shape tokens from point clouds.


import copy
import os
import typing as T
from urllib.parse import urlparse

import torch


def update_config(
    *,
    ori_config: T.Dict[str, T.Any],
    new_config: T.Dict[str, T.Any],
    allow_new_key: bool,
):
    for tmp_k, tmp_v in new_config.items():
        if not allow_new_key:
            assert tmp_k in ori_config, f"{tmp_k=}"
        if isinstance(tmp_v, dict):
            if tmp_k in ori_config:
                ori_config[tmp_k] = update_config(
                    ori_config=ori_config[tmp_k],
                    new_config=tmp_v,
                    allow_new_key=allow_new_key,
                )
            else:
                ori_config[tmp_k] = tmp_v
        else:
            ori_config[tmp_k] = tmp_v
    return ori_config


def is_http_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def download_checkpoint(url: str, download_dir_root: str, overwrite: bool) -> str:
    """Download a checkpoint from a URL into ``download_dir_root`` and return the local path.

    Skips the download if the destination already exists (unless ``overwrite=True``).
    Streams to a ``.tmp`` file and renames on success so an interrupted download
    cannot leave a partially-written file in the cache.
    """
    import requests
    from tqdm import tqdm

    os.makedirs(download_dir_root, exist_ok=True)
    filename = os.path.basename(urlparse(url).path) or "checkpoint.ckpt"
    local_path = os.path.join(download_dir_root, filename)

    if os.path.exists(local_path) and not overwrite:
        print(f"using cached checkpoint at {local_path}")
        return local_path

    print(f"downloading checkpoint from {url} -> {local_path}")
    tmp_path = local_path + ".tmp"
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with (
            open(tmp_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=filename) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    os.replace(tmp_path, local_path)
    return local_path


def load_model(
    checkpoint_url: str,
    download_dir_root: str = "artifacts",
    overwrite: bool = True,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu"),
    eval: bool = True,
    freeze: bool = True,
    load_params: bool = True,
    config_overwrite: T.Optional[T.Dict[str, T.Any]] = None,
    config_overwrite_allow_new_key: bool = False,
) -> T.Dict[str, T.Any]:
    """
    Download and load the model
    Args:
        checkpoint_url:
            local path to the checkpoint, or an ``http(s)://`` URL. URLs are
            downloaded into ``download_dir_root`` and reused on subsequent calls.
        download_dir_root:
            where the model will be saved
        overwrite:
            whether to overwrite the model
        config_overwrite:
            a dict that contains key-values to be overwritten in the originall config
        config_overwrite_allow_new_key:
            if True, we allow new keys that do not exist in the original config.
    Returns:
        model:
    """
    if is_http_url(checkpoint_url):
        checkpoint_filename = download_checkpoint(
            url=checkpoint_url,
            download_dir_root=download_dir_root,
            overwrite=overwrite,
        )
    else:
        assert os.path.exists(checkpoint_url), f"{checkpoint_url} does not exist"
        checkpoint_filename = checkpoint_url
    print(f"loading model from {checkpoint_filename}")

    # load checkpoint
    checkpoint = torch.load(checkpoint_filename, map_location=torch.device("cpu"))
    raw_ori_config = checkpoint["config"]
    global_step = checkpoint["global_step"]

    if config_overwrite is None:
        ori_config = raw_ori_config
    else:
        ori_config = update_config(
            ori_config=copy.deepcopy(raw_ori_config),
            new_config=config_overwrite,
            allow_new_key=config_overwrite_allow_new_key,
        )

    model = None
    if load_params:
        # Lightning's `load_from_checkpoint` is gone — BaseTrainer is now a
        # plain nn.Module. Construct the trainer with the params from
        # `ori_config["plm_config"]["params"]` (same kwargs we used to pass
        # to load_from_checkpoint), then load weights manually from the
        # Lightning .ckpt's `state_dict` key. We use strict=False because
        # checkpoints include lpips weights we may not have installed.
        target = ori_config["plm_config"]["target"]
        params = ori_config["plm_config"]["params"]
        if target == "lito.trainers.lito_trainer.LightTokenizationTrainer":
            from lito.trainers import lito_trainer
            model = lito_trainer.LightTokenizationTrainer(**params)
        elif target == "lito.trainers.lito_dit_trainer.LiToDiTTrainer":
            from lito.trainers import lito_dit_trainer
            model = lito_dit_trainer.LiToDiTTrainer(**params)
        else:
            raise NotImplementedError(target)

        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)

        # Preserve what BaseTrainer.on_load_checkpoint used to do:
        # stash the config blob on the model for downstream consumers.
        try:
            model.config = checkpoint.get("config")
        except Exception:
            pass

        model.to(device=device, dtype=dtype)
        if eval:
            model.eval()
        if freeze:
            model.freeze()

    print(f"loaded model at iter {global_step}", flush=True)
    return dict(
        model=model,
        config=ori_config,
        ori_config=raw_ori_config,
        global_step=global_step,
        checkpoint_filename=checkpoint_filename,
    )
