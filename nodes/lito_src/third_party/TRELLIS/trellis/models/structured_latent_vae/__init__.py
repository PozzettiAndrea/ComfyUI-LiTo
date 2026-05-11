# Vendored TRELLIS package marker — eager submodule imports removed.
#
# Upstream this file did:
#     from .encoder import SLatEncoder, ElasticSLatEncoder
#     from .decoder_gs import SLatGaussianDecoder, ElasticSLatGaussianDecoder
#     from .decoder_rf import SLatRadianceFieldDecoder, ElasticSLatRadianceFieldDecoder
#     from .decoder_mesh import SLatMeshDecoder, ElasticSLatMeshDecoder
#
# decoder_gs imports `from ...representations import Gaussian` which would drag
# in utils3d + flexicubes via the parent representations/__init__.py. LiTo only
# needs `base.SparseTransformerBase` and `decoder_mesh` from this package; both
# are imported by absolute path so the empty parent __init__ doesn't matter.
