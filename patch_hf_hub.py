"""Patch huggingface_hub to restore cached_download (removed in >=0.26).

This patches the installed __init__.py so the shim persists at import time
(needed by modelscope and other libraries that import cached_download).
"""
import huggingface_hub
import inspect
import os

try:
    from huggingface_hub import cached_download
    print("cached_download already exists, no patch needed")
except ImportError:
    init_path = os.path.join(
        os.path.dirname(inspect.getfile(huggingface_hub)), "__init__.py"
    )
    with open(init_path, "a") as f:
        f.write(
            "\n# Compatibility shim: cached_download removed in 0.26\n"
            "from huggingface_hub import hf_hub_download as cached_download\n"
        )
    print(f"Patched {init_path}: cached_download = hf_hub_download")

    # Verify
    import importlib
    importlib.reload(huggingface_hub)
    from huggingface_hub import cached_download  # noqa: F811
    print("Verification OK: cached_download importable")
