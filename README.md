> [!WARNING]
> Warning, uses experimental package `comfy-env` to attempt a one click isolated install. Will download and use pixi package manager.

https://github.com/user-attachments/assets/c3ff490b-e0d4-47cc-953b-a2236673e8bf

# ComfyUI-LiTo

ComfyUI wrapper for [LiTo](https://apple.github.io/ml-lito/) (Surface Light Field Tokenization), Apple Research's image-to-3D Gaussian Splat generator from ICLR 2026.

Single RGBA image → 3D Gaussians (~524K splats) capturing geometry + view-dependent appearance (specular highlights, Fresnel reflections). ~4.7s on H100 after compilation.

## Nodes

| Node | Description |
|------|-------------|
| **(Down)Load LiTo Model** | Downloads `lito_dit_rgba.ckpt` (~3GB) from Apple CDN to `ComfyUI/models/lito/` |
| **LiTo Preprocess Image** | Background removal (rembg) + crop/pad to 518×518 RGBA |
| **LiTo Image to 3D** | DiT flow-matching sampling + Gaussian decoding |
| **LiTo Export PLY** | Save Gaussians to standard PLY |
| **LiTo Preview Point Cloud** | Browser-based 3D preview |

## Workflow

```
LoadImage → LiToPreprocess → LiToImageTo3D → LiToExportPLY → LiToPreviewPointCloud
                                  ↑
                            LiToLoadModel
```

## Installation

This pack uses [comfy-env](https://github.com/PozzettiAndrea/comfy-env) for process isolation — LiTo's heavy dependencies (pytorch3d, gsplat, nvdiffrast, flash_attn, spconv, xformers) run in their own subprocess environment, separate from ComfyUI's host Python.

CUDA wheels resolved automatically from [cuda-wheels](https://pozzettiandrea.github.io/cuda-wheels).

## License

The wrapper code follows Apple's [LICENSE](./LICENSE). The model weights are governed by [LICENSE_MODEL](./LICENSE_MODEL) (Apple ML Research model license — non-commercial research use).

## Credits

- [LiTo](https://github.com/apple/ml-lito) by Jen-Hao Rick Chang, Xiaoming Zhao, Dorian Chan, Oncel Tuzel (Apple, ICLR 2026)
