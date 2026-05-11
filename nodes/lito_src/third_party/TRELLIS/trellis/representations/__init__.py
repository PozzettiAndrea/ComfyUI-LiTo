# Vendored TRELLIS package marker — most eager submodule imports removed.
#
# Upstream this file did:
#     from .radiance_field import Strivec
#     from .octree import DfsOctree as Octree
#     from .gaussian import Gaussian
#     from .mesh import MeshExtractResult
#
# .gaussian needs utils3d and .mesh needs the flexicubes git submodule (which
# TRELLIS requires you to initialize separately). LiTo never uses Gaussian /
# Strivec / Octree, but TRELLIS's own decoder_mesh imports MeshExtractResult
# from this namespace, so we re-export it (and SparseFeatures2Mesh) from LiTo's
# vendored cube2mesh which uses our locally-vendored FlexiCubes.
from lito.integrations.trellis.representations.mesh.cube2mesh import (
    MeshExtractResult,
    SparseFeatures2Mesh,
)
