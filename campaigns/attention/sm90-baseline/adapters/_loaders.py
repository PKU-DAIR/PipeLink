"""Load the pinned standalone public kernels without importing a serving engine."""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
PINS = {
    "flash-attention": "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820",
    "flashinfer": "f89d15f2c5a5d0036e6b9cafb2b67496106b9ad6",
    "flash-msa": "5834d91e43935209408c3a1e974bf0b36de76802",
    "vllm": "3137ff07739f0cd19e06119506c6432a4a7e77ad",
    "phi3-small": "e95b185a59a58450c9b1dd7347a9dee78f18113e",
}


def clones_root() -> Path:
    return Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "clones"


def load_fa3():
    """Return flash_attn_interface from the single shipped bf16 d128 build."""
    output = Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "sm90-baseline" / "fa3"
    if not list((output / "flash_attn_3").glob("_C*.so")):
        raise FileNotFoundError(f"FA3 extension missing in {output}; run {PROJECT / 'build_fa3.sh'}")
    if str(output) not in sys.path:
        sys.path.insert(0, str(output))
    return importlib.import_module("flash_attn_interface")


def _module_from_file(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_vllm_m3():
    """Return the unmodified sparse_attn.py module; only its two imports are supplied."""
    import torch
    import triton
    import triton.language as tl

    platform = ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(
        is_arch_support_pdl=lambda: torch.cuda.get_device_capability()[0] >= 9
    )
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.triton, triton_utils.tl = triton, tl
    shims = {"vllm.platforms": platform, "vllm.triton_utils": triton_utils}
    previous = {name: sys.modules.get(name) for name in shims}
    sys.modules.update(shims)
    try:
        return _module_from_file(
            "pipelink_vllm_m3_sparse_attn",
            clones_root() / "vllm-main/vllm/models/minimax_m3/common/ops/sparse_attn.py",
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_phi3():
    """Return Phi's shipped Triton kernel module (including native mask builder)."""
    return _module_from_file(
        "pipelink_phi3_triton_flash_blocksparse_attn",
        clones_root() / "phi3-small" / PINS["phi3-small"] / "triton_flash_blocksparse_attn.py",
    )


def load_flash_msa():
    """Return (forward module, reverse-index module); FA3 must already be built."""
    load_fa3()
    return (
        importlib.import_module("flash_msa.msa_forward_cutedsl"),
        importlib.import_module("flash_msa.reverse_index_cuda"),
    )
