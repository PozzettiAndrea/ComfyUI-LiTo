# Vendored TRELLIS v1 — package marker only.
#
# Upstream this file eagerly imports every subpackage:
#
#     from . import models, modules, pipelines, renderers, representations, utils
#
# LiTo only uses:
#   - trellis.modules.sparse
#   - trellis.modules.utils
#   - trellis.models.structured_latent_vae.{base, decoder_mesh}
#   - trellis.models (lazy via __getattr__) — used by the sparse-structure pipeline
#
# Importing `pipelines` transitively pulls in `rembg` -> `pymatting` -> `numba`,
# and numba crashes inside comfy-env workers because comfy-env replaces
# builtins.print with a wrapper whose __name__ is "_forwarded_print" but doesn't
# expose that name on __main__. Keeping this file empty avoids the eager
# pull-in entirely; submodules LiTo needs still resolve via direct dotted import.
