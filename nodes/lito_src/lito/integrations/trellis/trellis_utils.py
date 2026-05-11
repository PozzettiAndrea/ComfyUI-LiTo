#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#


import os
import sys

# In the upstream Apple layout this file lives at src/lito/integrations/trellis/
# and the repo root is 5 levels up. In our ComfyUI wrapper we vendor lito as
# nodes/lito_src/lito/..., so the equivalent "repo root" (the directory that
# contains third_party/) is one level shallower.
REPO_ROOT = os.path.normpath(os.path.join(__file__, "..", "..", "..", ".."))


def add_trellis_to_sys_path(
    trellis_repo_root: str = None,
):
    """
    Add TRELLIS repo path so we can import from it.

    Args:
        trellis_repo_dir:
            str, dir containing trellis/.
            If None, it first checks if "TRELLIS_REPO_DIR" exists in ENVIRONMENT VARIABLES;
            if not, it uses 'os.path.join(REPO_ROOT, "third_party", "TRELLIS")'
    """

    if trellis_repo_root is None:
        trellis_repo_root = os.environ.get("TRELLIS_REPO_DIR", os.path.join(REPO_ROOT, "third_party", "TRELLIS"))

    assert os.path.exists(trellis_repo_root), f"{trellis_repo_root} does not exist"

    if trellis_repo_root not in sys.path:
        sys.path.append(trellis_repo_root)

    return trellis_repo_root
