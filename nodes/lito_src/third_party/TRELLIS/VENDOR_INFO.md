# Vendored TRELLIS v1

Source: https://github.com/microsoft/TRELLIS
Pinned commit: `6b0d64751ad54d9c32d7b05fec482eb29178f56f`
License: MIT (see `LICENSE`)

Only the `trellis/` Python package is vendored — the rest of the TRELLIS repo
(dataset toolkits, app.py, blender scripts, etc.) is not used by LiTo.

`trellis/__init__.py` has been blanked — see the comment in that file for
rationale (avoids eager rembg/numba pull-in via the unused `pipelines`
subpackage). LiTo's submodule imports (`trellis.modules.sparse`,
`trellis.models.structured_latent_vae.*`, `trellis.modules.utils`) still resolve
via direct dotted import.

## DO NOT re-add as a git submodule

This directory was previously added accidentally as a git submodule (mode
160000 gitlink) which caused git to wipe the working tree on every checkout.
If you see `git rm --cached` complaints about a submodule here, that's likely
the same problem recurring; follow the same fix:

    git rm --cached nodes/lito_src/third_party/TRELLIS
    # then re-vendor and `git add` as normal files
