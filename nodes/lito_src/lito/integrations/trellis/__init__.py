#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
import platform
import sys
import types

from lito.integrations.trellis.trellis_utils import add_trellis_to_sys_path

add_trellis_to_sys_path()


# kaolin is Linux+CUDA-only; on macOS the PyPI placeholder wheel raises
# ImportError. TRELLIS's flexicubes module does
# ``from kaolin.utils.testing import check_tensor`` at module top, which would
# otherwise prevent ``import trellis`` (and therefore the sparse-structure
# pipeline used by the voxel decoder) from loading at all on macOS.
#
# ``check_tensor`` is only used as a debug/validation helper, so on macOS we
# install a stub ``kaolin.utils.testing`` module exposing a no-op
# ``check_tensor``. This must run before any ``import trellis`` so that the
# vendored TRELLIS submodule resolves the symbol against the stub. We only
# touch ``sys.modules`` on Darwin so Linux behaviour is unchanged.
# kaolin shim — needed on every platform now that we vendor FlexiCubes locally
# (the upstream FlexiCubes file does `from kaolin.utils.testing import check_tensor`,
# which is just a debug helper). We don't carry the heavy kaolin dep just for
# that, so install a no-op stub if real kaolin isn't importable. Originally this
# was Darwin-only because TRELLIS's bundled flexicubes worked on Linux+CUDA via
# kaolin from conda; with our vendoring we need the shim everywhere.
try:
    from kaolin.utils.testing import check_tensor as _kaolin_check_tensor  # noqa: F401
except ImportError:
    for _stale in ("kaolin", "kaolin.utils", "kaolin.utils.testing"):
        sys.modules.pop(_stale, None)
    _kaolin_pkg = types.ModuleType("kaolin")
    _kaolin_utils = types.ModuleType("kaolin.utils")
    _kaolin_testing = types.ModuleType("kaolin.utils.testing")
    _kaolin_testing.check_tensor = lambda *args, **kwargs: True
    _kaolin_utils.testing = _kaolin_testing
    _kaolin_pkg.utils = _kaolin_utils
    sys.modules["kaolin"] = _kaolin_pkg
    sys.modules["kaolin.utils"] = _kaolin_utils
    sys.modules["kaolin.utils.testing"] = _kaolin_testing
