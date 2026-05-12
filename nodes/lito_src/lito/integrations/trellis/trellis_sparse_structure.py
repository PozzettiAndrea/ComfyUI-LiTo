import typing as T

import trellis.models

import torch


def get_trellis_sparse_structure_pipeline(
    checkpoint_dir: str = None,
) -> "TrellisSparseStructurePipeline":
    """
    Load or download huggingface checkpoint and return TrellisSparseStructurePipeline.

    Args:
        checkpoint_dir:
            str, if not None, points to the local dir of the checkpoint, containing pipeline.json.

    Returns:
        pipeline in eval mode, on cpu
    """
    if checkpoint_dir is None:
        checkpoint_dir = "microsoft/TRELLIS-image-large"  # download from huggingface
    pipeline = TrellisSparseStructurePipeline.from_pretrained(checkpoint_dir)
    return pipeline


class TrellisSparseStructurePipeline:
    """
    Pipeline for testing sparse structure vae.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """

    def __init__(
        self,
        models: T.Dict[str, torch.nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
        self.sparse_structure_info = None

    @staticmethod
    def from_pretrained(path: str) -> "TrellisSparseStructurePipeline":
        """
        Load a pretrained model.
        """
        config_dict = {
            "name": "TrellisSparseStructurePipeline",
            "args": {
                "models": {
                    "sparse_structure_encoder": "ckpts/ss_enc_conv3d_16l8_fp16",
                    "sparse_structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
                },
                "sparse_structure_info": {"res_io": 64, "res_latent": 16, "dim_latent": 8},
            },
        }
        args = config_dict["args"]

        _models = {}
        for k, v in args["models"].items():
            try:
                _models[k] = trellis.models.from_pretrained(f"{path}/{v}")
            except:
                _models[k] = trellis.models.from_pretrained(v)

        new_pipeline = TrellisSparseStructurePipeline(_models)
        new_pipeline._pretrained_args = args
        new_pipeline.sparse_structure_info = args["sparse_structure_info"]
        return new_pipeline

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = dict(),
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution  # 16
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples  # (b, d=8, reso_k=16, reso_j=16, reso_i=16)

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()  # (num, 5b1kji) -> (num, 4bkji)

        return coords  # (num_total_occupied, 4bkji)

    def encode_highres_grid(self, dense_grid: torch.Tensor):
        """
        Encode a dense highres grid (0, 1) into low-res latent
        Args:
            dense_grid:
                (b, 1, res_highres_k, res_highres_j, res_highres_i)  [0, 1]

        Returns:
            latent:
                (b, d=8, res_lowres_k, res_lowres_j, res_lowres_i)  float
        """
        encoder = self.models["sparse_structure_encoder"]
        lowres_latent = encoder(dense_grid)  # (b, d=8, res_lowres_k, res_lowres_j, res_lowres_i)
        return lowres_latent

    def decode_lowres_latent_to_logits(self, lowres_latent: torch.Tensor):
        """
        Decode a dense lowres latent into high-res dense grid logits

        Args:
            lowres_latent:
                (b, d=8, res_lowres_k, res_lowres_j, res_lowres_i)

        Returns:
            highres_logits:
                (b, 1, res_highres_k, res_highres_j, res_highres_i)  float.  > 0 if occupied
        """
        decoder = self.models["sparse_structure_decoder"]
        dense_logits = decoder(lowres_latent)  # (b, 1, res_highres_k, res_highres_j, res_highres_i)
        return dense_logits

    @torch.no_grad()
    def decode_lowres_latent_to_occ(self, lowres_latent: torch.Tensor):
        """
        Decode a dense lowres latent into high-res dense grid logits

        Args:
            lowres_latent:
                (b, d=8, res_lowres_k, res_lowres_j, res_lowres_i)

        Returns:
            occ:
                (b, 1, res_highres_k, res_highres_j, res_highres_i)  bool
        """
        logits = self.decode_lowres_latent_to_logits(
            lowres_latent=lowres_latent
        )  # (b, 1, res_highres_k, res_highres_j, res_highres_i)
        occ = logits > 0
        return occ

    @torch.no_grad()
    def run(
        self,
        dense_grid: torch.Tensor,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            dense_grid:
                (b, 1, res_highres_k, res_highres_j, res_highres_i)  [0, 1]  1 means occupied
        """

        latent = self.encode_highres_grid(dense_grid=dense_grid)  # (b, 1, low_res_k, low_res_j, low_res_i)
        rec = self.decode_lowres_latent_to_occ(lowres_latent=latent)  # (b, 1, high_res_k, high_res_j, high_res_i) bool
        return dict(
            latent=latent,
            reconstruct=rec,
        )

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, "device"):
                return model.device
        for model in self.models.values():
            if hasattr(model, "parameters"):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        import comfy.model_management
        self.to(comfy.model_management.get_torch_device())

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
