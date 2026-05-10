import sys
import pathlib

# Add vendored source to path
_NODES_DIR = pathlib.Path(__file__).resolve().parent
_LITO_SRC = _NODES_DIR / "lito_src"
if str(_LITO_SRC) not in sys.path:
    sys.path.insert(0, str(_LITO_SRC))

from .load_model import LiToLoadModel
from .preprocess import LiToPreprocess
from .inference import LiToImageTo3D
from .export_ply import LiToExportPLY
from .preview_nodes import LiToPreviewPointCloud

NODE_CLASS_MAPPINGS = {
    "LiToLoadModel": LiToLoadModel,
    "LiToPreprocess": LiToPreprocess,
    "LiToImageTo3D": LiToImageTo3D,
    "LiToExportPLY": LiToExportPLY,
    "LiToPreviewPointCloud": LiToPreviewPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LiToLoadModel": "(Down)Load LiTo Model",
    "LiToPreprocess": "LiTo Preprocess Image",
    "LiToImageTo3D": "LiTo Image to 3D",
    "LiToExportPLY": "LiTo Export PLY",
    "LiToPreviewPointCloud": "LiTo Preview Point Cloud",
}
