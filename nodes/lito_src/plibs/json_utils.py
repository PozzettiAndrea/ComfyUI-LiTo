import json

import numpy as np


class PureNumpyJsonEncoder(json.JSONEncoder):
    """Custom encoder for saving numpy data types into a json file.

    Json only supports saving native python types like int, list, dict, etc.
    Therefore it raises exceptions if passed a numpy-typed number or array.
    The encoder translates numpy objects to python native objects automatically.

    Usage:
        When dumping a dict to json, use:
        json.dump(some_dict, filename, cls=NumpyJsonEncoder)
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


class NumpyJsonEncoder(PureNumpyJsonEncoder):
    def default(self, obj):
        # lazy import
        import torch

        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()

        return super().default(obj)
