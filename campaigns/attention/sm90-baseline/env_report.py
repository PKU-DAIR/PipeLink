#!/usr/bin/env python3
"""Report pinned public sm90 implementations and isolated bf16 d128 correctness smokes."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

from adapters._loaders import PINS, clones_root, load_fa3, load_flash_msa, load_phi3, load_vllm_m3

IMPLEMENTATIONS = ("fa3", "fa4", "flashinfer-fa3", "flashinfer-variable-fa3", "flex-triton", "flex-flash", "vllm-m3", "flash-msa", "phi3-small")
PACKAGES = ("torch", "triton", "flash-attn-4", "nvidia-cutlass-dsl", "quack-kernels", "flashinfer-python", "flash-msa", "apache-tvm-ffi", "cuda-python")


def fetch_sources():
    """Fetch pinned file-loaded kernels; installed backends come from uv.lock."""
    root = clones_root()
    root.mkdir(parents=True, exist_ok=True)
    repos = (
        ("vllm-main", "vllm", "https://github.com/vllm-project/vllm.git", "vllm/models/minimax_m3/common/ops"),
    )
    for directory, pin, url, sparse in repos:
        destination = root / directory
        if not (destination / ".git").exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination)], check=True)
        if sparse:
            subprocess.run(["git", "-C", str(destination), "sparse-checkout", "set", sparse], check=True)
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", "origin", PINS[pin]], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", PINS[pin]], check=True)
    destination = root / "phi3-small" / PINS["phi3-small"]
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("triton_flash_blocksparse_attn.py", "triton_blocksparse_attention_layer.py", "config.json", "LICENSE"):
        target = destination / filename
        if not target.exists():
            url = f"https://huggingface.co/microsoft/Phi-3-small-128k-instruct/resolve/{PINS['phi3-small']}/{filename}"
            urllib.request.urlretrieve(url, target)


def configure_caches():
    root = Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "sm90-public-cache"
    for key, leaf in (("TORCH_EXTENSIONS_DIR", "torch-extensions"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"), ("FLASHINFER_WORKSPACE_BASE", "flashinfer"), ("CUTE_DSL_CACHE_DIR", "cute"), ("CUDA_CACHE_PATH", "cuda")):
        os.environ.setdefault(key, str(root / leaf))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_HOME", os.environ["PIPELINK_CUDA_HOME"])
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("MAX_JOBS", "4")
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")


def reference(q, k, v, phi=False):
    """All output elements, fp32 math with TF32 disabled; independent native Phi equation."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    s, heads, dim = q.shape[1:]
    positions = torch.arange(s, device=q.device)
    qp, kp = positions[:, None], positions[None, :]
    causal = kp <= qp
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for h in range(heads):
        kh = h // (heads // k.shape[2])
        scores = q[0, :, h].float() @ k[0, :, kh].float().T / math.sqrt(dim)
        mask = causal
        if phi:
            qb, kb = qp // 64, kp // 64
            mask = mask & ((qb - kb < 16) | ((kb + h + 1) % 8 == 0))
        scores.masked_fill_(~mask, -math.inf)
        output[0, :, h] = scores.softmax(-1) @ v[0, :, kh].float()
    return output


def run_implementation(name, q, k, v):
    import torch

    scale, seq, hq, hkv = 128 ** -0.5, q.shape[1], q.shape[2], k.shape[2]
    facts = {}
    if name == "fa3":
        module = load_fa3()
        out, lse = module.flash_attn_func(q, k, v, causal=True, softmax_scale=scale, return_attn_probs=True)
        facts = {"api": "flash_attn_func(..., causal=True, return_attn_probs=True)", "layout": "BSHD", "lse_base": "e", "lse_shape": list(lse.shape)}
        facts["build_flags"] = __import__("flash_attn_config").CONFIG["build_flags"]
    elif name == "fa4":
        from flash_attn.cute.interface import flash_attn_func
        out, lse = flash_attn_func(q, k, v, causal=True, softmax_scale=scale, num_splits=1, return_lse=True)
        facts = {"api": "flash_attn_func(..., num_splits=1, return_lse=True)", "layout": "BSHD", "lse_base": "e", "lse_shape": list(lse.shape)}
    elif name == "flashinfer-fa3":
        import flashinfer
        out, lse = flashinfer.single_prefill_with_kv_cache(q[0], k[0], v[0], causal=True, backend="fa3", sm_scale=scale, return_lse=True)
        out = out[None]
        facts = {"api": "single_prefill_with_kv_cache(..., backend='fa3', return_lse=True)", "layout": "SHD", "lse_base": 2, "lse_shape": list(lse.shape)}
    elif name == "flashinfer-variable-fa3":
        from flashinfer import VariableBlockSparseAttentionWrapper
        wrapper = VariableBlockSparseAttentionWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device), backend="fa3")
        nb = seq // 128
        block_mask = torch.ones((hkv, nb, nb), dtype=torch.bool, device=q.device).tril()
        sizes = torch.full((hkv, nb), 128, dtype=torch.int32, device=q.device)
        wrapper.plan(block_mask, sizes, sizes, hq, hkv, 128, causal=True, sm_scale=scale, q_data_type=q.dtype, kv_data_type=k.dtype)
        out, lse = wrapper.run(q[0].transpose(0, 1).contiguous(), k[0].transpose(0, 1).contiguous(), v[0].transpose(0, 1).contiguous(), return_lse=True)
        facts = {"api": "VariableBlockSparseAttentionWrapper(backend='fa3').plan/run", "input_layout": "HSD", "native_output_shape": list(out.shape), "lse_base": 2, "lse_shape": list(lse.shape), "block": 128}
        out = out.transpose(0, 1)[None]
    elif name.startswith("flex-"):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        def causal(b, h, qi, ki):
            return qi >= ki
        mask = create_block_mask(causal, B=None, H=None, Q_LEN=seq, KV_LEN=seq, device=q.device)
        backend = "TRITON" if name == "flex-triton" else "FLASH"
        def attention(qh, kh, vh):
            return flex_attention(qh, kh, vh, block_mask=mask, scale=scale, enable_gqa=True, return_lse=True, kernel_options={"BACKEND": backend})
        compiled = torch.compile(attention, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
        out, lse = compiled(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        out = out.transpose(1, 2)
        facts = {"api": "torch.compile(flex_attention), kernel_options BACKEND=" + backend, "layout": "BHSD", "lse_base": "e", "lse_shape": list(lse.shape), "cuda_graphs": False}
    elif name == "vllm-m3":
        module = load_vllm_m3()
        blocks = seq // 128
        kv = torch.cat((k[0].reshape(blocks, 128, hkv, 128), v[0].reshape(blocks, 128, hkv, 128)), dim=-1).permute(0, 2, 1, 3).contiguous()
        selections = torch.arange(blocks, dtype=torch.int32, device=q.device).expand(hkv, seq, blocks).contiguous()
        selections.masked_fill_(selections > (torch.arange(seq, device=q.device)[None, :, None] // 128), -1)
        out = torch.empty_like(q)
        module.minimax_m3_sparse_attn(q[0], kv, selections, torch.arange(blocks, dtype=torch.int32, device=q.device)[None], torch.tensor([0, seq], dtype=torch.int32, device=q.device), torch.tensor([seq], dtype=torch.int32, device=q.device), torch.zeros(1, dtype=torch.int32, device=q.device), seq, hkv, scale, out[0])
        facts = {"api": "minimax_m3_sparse_attn(..., output=out)", "layout": "Q:SHD; KV:pages,Hkv,128,2D; selection:Hkv,S,topk", "lse_base": None, "block": 128, "selection": "full causal prefix; sorted; -1 suffix"}
    elif name == "flash-msa":
        module, metadata_module = load_flash_msa()
        blocks = seq // 128
        selections = torch.arange(blocks, dtype=torch.int32, device=q.device).expand(1, hkv, seq, blocks).contiguous()
        selections.masked_fill_(selections > (torch.arange(seq, device=q.device)[None, None, :, None] // 128), -1)
        metadata = metadata_module.build_sparse_attention_metadata_cuda(selections, backward_query_chunk=32)
        out, lse, _ = module.run_main_forward(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), scale=scale, metadata=metadata)
        out = out.transpose(1, 2)
        facts = {"api": "build_sparse_attention_metadata_cuda + run_main_forward", "layout": "BHSD", "lse_base": "e", "lse_shape": list(lse.shape), "block": 128, "selection": "full prefix; local included exactly once", "num_remote_tasks": metadata.num_remote_tasks}
        from flash_msa._flash_attn_compat import _fa4_varlen_func
        facts["backend"] = "FA3 local varlen + " + ("FA4 remote paged-varlen" if _fa4_varlen_func() is not None else "FA3 remote varlen")
    elif name == "phi3-small":
        module = load_phi3()
        layout, _, _ = module._get_sparse_attn_mask(hq, seq, seq, q.dtype, q.device, BLOCK=64, local_blocks=16, vert_stride=8, homo_head=False)
        out = module.blocksparse_flash_attn_padded_fwd(q, k, v, scale, layout, block_size=64, max_seqlen=seq)
        facts = {"api": "_get_sparse_attn_mask + blocksparse_flash_attn_padded_fwd", "layout": "BSHD", "lse_base": None, "block": 64, "mask": "causal & (qblock-kblock<16 | (kblock+h+1)%8==0)"}
    return out, facts


def smoke(name):
    import torch

    torch.manual_seed(20260927)
    q = torch.randn((1, 2048, 32, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((1, 2048, 8, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    result = {"implementation": name, "passed": False}
    started = time.perf_counter()
    try:
        out, facts = run_implementation(name, q, k, v)
        torch.cuda.synchronize()
        result["first_call_seconds"] = time.perf_counter() - started
        result["api_facts"] = facts
        ref = reference(q, k, v, phi=name == "phi3-small")
        error = (out.float() - ref).abs().max().item()
        result.update(max_abs_error_o=error, passed=math.isfinite(error) and error <= 0.02, output_dtype=str(out.dtype))
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), attempted="pinned upstream import and native d128 bf16 forward", first_call_seconds=time.perf_counter() - started)
    return result


def markdown(report):
    """Public projection: no device ordinals, UUIDs, diagnostics or local paths."""
    lines = ["# sm90 public-baseline environment", "",
             f"Generated by `env_report.py` from `{report['schema']}` at {report['timestamp']}.", "",
             f"GPU: {report['gpu']} (sm{''.join(map(str, report['capability']))}); driver {report['driver']}; "
             f"torch CUDA {report['torch_cuda']}; Python {report['python'].split()[0]}.", "",
             "| Package | Version |", "|---|---|"]
    lines += [f"| {name} | {version or 'not installed'} |" for name, version in report["versions"].items()]
    lines += ["", "| Pinned source | Commit |", "|---|---|"]
    lines += [f"| {name} | `{pin}` |" for name, pin in report["pins"].items()]
    shape = report["shape"]
    lines += ["", *(f"Override: {name}: {value}." for name, value in report["overrides"].items()), "",
              f"Correctness smoke: batch {shape['batch']}, sequence {shape['seqlen']}, {shape['heads_q']} query / "
              f"{shape['heads_kv']} KV heads, head dim {shape['head_dim']}, {shape['dtype']}; fp32 reference, max |error| <= "
              f"{report['reference']['max_abs_error_threshold']}. {report['timing_note']}", "",
              "| Implementation | Passed | Max abs error | API | Error |", "|---|---|---:|---|---|"]
    for row in report["implementations"]:
        error = (row.get("error") or "").splitlines()[0] if row.get("error") else ""
        api = (row.get("api_facts") or {}).get("api", "")
        lines.append(f"| {row['implementation']} | {'yes' if row['passed'] else 'no'} | "
                     f"{row.get('max_abs_error_o', '')} | `{api}` | {error} |")
    lines += ["", f"All passed: {'yes' if report['passed'] else 'no'}.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-sources", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device ordinal in this process")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--only", nargs="+", choices=IMPLEMENTATIONS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--render", type=Path, help="CPU-only: write the Markdown report for an existing JSON record")
    args = parser.parse_args()
    if args.render:
        args.render.with_suffix(".md").write_text(markdown(json.loads(args.render.read_text())))
        return 0
    configure_caches()
    if args.fetch_sources or args.fetch_only:
        with contextlib.redirect_stdout(sys.stderr):
            fetch_sources()
    if args.fetch_only:
        return 0
    import torch
    torch.cuda.set_device(args.gpu)
    if args.implementation:
        with contextlib.redirect_stdout(sys.stderr):
            report = smoke(args.implementation)
    else:
        versions = {}
        for name in PACKAGES:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        report = {
            "schema": "pipelink-sm90-public-env-v1", "timestamp": datetime.now(timezone.utc).isoformat(),
            "pins": PINS, "versions": versions, "python": sys.version,
            "gpu": torch.cuda.get_device_name(args.gpu), "capability": list(torch.cuda.get_device_capability(args.gpu)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_cuda": torch.version.cuda,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0],
            "shape": {"batch": 1, "seqlen": 2048, "heads_q": 32, "heads_kv": 8, "head_dim": 128, "dtype": "bf16"},
            "reference": {"dtype": "fp32", "tf32": False, "all_output_elements": True, "max_abs_error_threshold": 0.02},
            "timing_note": "First-call wall time includes import, layout/plan and JIT; not an attention benchmark.",
            "overrides": {"flash-msa torch requirement": "upstream 2.12.0 overridden with plan pin 2.14.0"},
            "implementations": [],
        }
        with tempfile.TemporaryDirectory(prefix="sm90-env-", dir=os.environ.get("TMPDIR")) as directory:
            for name in args.only or IMPLEMENTATIONS:
                destination = Path(directory) / f"{name}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--implementation", name,
                           "--gpu", str(args.gpu), "--out", str(destination)]
                print(f"Smoke {name}", file=sys.stderr, flush=True)
                process = subprocess.run(command, capture_output=True, text=True)
                if destination.exists():
                    row = json.loads(destination.read_text())
                else:
                    row = {"implementation": name, "passed": False, "error": f"child exited {process.returncode}", "attempted": command}
                row["exit_code"] = process.returncode
                if process.stderr:
                    row["diagnostics"] = process.stderr
                report["implementations"].append(row)
        report["passed"] = all(row["passed"] for row in report["implementations"])
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        if not args.implementation:
            args.out.with_suffix(".md").write_text(markdown(report))
    print(text, end="")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
