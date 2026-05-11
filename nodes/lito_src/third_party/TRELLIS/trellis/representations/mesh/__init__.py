# Vendored TRELLIS package marker — eager submodule import removed.
#
# Upstream this file did:
#     from .cube2mesh import SparseFeatures2Mesh, MeshExtractResult
#
# That cube2mesh needs the flexicubes git submodule (uninitialized here). LiTo
# bundles its own cube2mesh under
# `lito.integrations.trellis.representations.mesh.cube2mesh` with a vendored
# FlexiCubes sibling, so we re-export from there. TRELLIS's own decoder_mesh
# does `from ...representations.mesh import SparseFeatures2Mesh`, which lands
# here.
from lito.integrations.trellis.representations.mesh.cube2mesh import (
    MeshExtractResult,
    SparseFeatures2Mesh,
)
