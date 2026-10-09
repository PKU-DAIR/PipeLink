#!/usr/bin/env python3
"""Build a seeded random Llama-3.2-1B and export a PipeLink host bundle."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig

HOST_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOST_DIR.parents[2]
MEGAKERNELS_ROOT = REPO_ROOT / "third_party" / "Megakernels"
sys.path.insert(0, str(MEGAKERNELS_ROOT))

import megakernels.demos.latency.scheduler as latency_scheduler  # noqa: E402
import megakernels.instructions as mk_instructions  # noqa: E402
import megakernels.utils as mk_utils  # noqa: E402
from megakernels.demos.latency.instructions import (  # noqa: E402
    AttentionReduction,
    PartialAttention,
)
from megakernels.demos.latency.python_vm import (  # noqa: E402
    INSTRUCTION_TO_SOLVER,
)
from megakernels.llama import LlamaForCausalLM, RMSNorm  # noqa: E402
from megakernels.model_types import BatchState, ExtraModelConfig  # noqa: E402
from megakernels.python_vm import PyVM_Interpreter  # noqa: E402
from megakernels.scheduler import assign_to_sms, tensorize_instructions  # noqa: E402

NUM_SMS = 148
NUM_LAYERS = 16
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256
BAR_ROWS = 10
LAYER_OPCODES = frozenset(range(1, 7))
OPCODE_NAMES = (
    "NoOp",
    "LayerNorm_QKV_MatVecRopeAppend",
    "PartialAttention",
    "AttentionReduction",
    "O_ProjResidual",
    "LayerNormDoubleMatVecSiLU",
    "DownProjResidual",
    "RMS_LM_Head",
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompt-len", required=True, type=int)
    parser.add_argument("--ntok", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument(
        "--attention-partitions",
        type=int,
        default=1,
        help="KV partitions per head; values above one schedule terminal attention reductions",
    )
    parser.add_argument(
        "--layer-limit",
        type=int,
        default=None,
        help="schedule only the first N layers; by default schedule all layers and lm_head",
    )
    args = parser.parse_args()
    if args.prompt_len < 0:
        parser.error("--prompt-len must be non-negative")
    if args.ntok <= 0:
        parser.error("--ntok must be positive")
    if args.max_seq_len < args.prompt_len + args.ntok:
        parser.error("--max-seq-len must cover prompt-len + ntok")
    if args.layer_limit is not None and not 1 <= args.layer_limit <= NUM_LAYERS:
        parser.error(f"--layer-limit must be between 1 and {NUM_LAYERS}")
    if not 1 <= args.attention_partitions <= 24:
        parser.error("--attention-partitions must be between 1 and 24 (instruction width)")
    return args


def make_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB_SIZE,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        max_position_embeddings=131072,
        tie_word_embeddings=True,
        torch_dtype=torch.bfloat16,
        attention_bias=False,
        mlp_bias=False,
    )


def configure_sm_count(device: torch.device) -> bool:
    """Return whether the 148-SM compatibility patch was installed."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        actual = torch.cuda.get_device_properties(device).multi_processor_count
        if actual != NUM_SMS:
            raise RuntimeError(
                f"the CUDA reference device has {actual} SMs; the port requires {NUM_SMS}"
            )
        return False

    def fixed_sm_count(_device: object) -> int:
        return NUM_SMS

    # Both scheduler.py modules imported get_sm_count by value. Patch those aliases as
    # well as the defining module and the method used by scheduling/tensorization.
    mk_utils.get_sm_count = fixed_sm_count
    mk_instructions.get_sm_count = fixed_sm_count
    latency_scheduler.get_sm_count = fixed_sm_count
    mk_instructions.BaseGlobals.sm_count = lambda self: NUM_SMS
    return True


def initialize_model(model: LlamaForCausalLM) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                module.weight.fill_(1.0)

    if model.config.tie_word_embeddings:
        model.lm_head.lm_head.weight = model.model.embed_tokens.embed_tokens.weight


def little_endian_array(tensor: torch.Tensor, dtype_name: str) -> np.ndarray:
    cpu = tensor.detach().contiguous().cpu()
    if dtype_name == "bf16":
        if cpu.dtype != torch.bfloat16:
            raise TypeError(f"expected bf16 tensor, got {cpu.dtype}")
        return cpu.view(torch.uint16).numpy().astype("<u2", copy=False)
    if dtype_name == "f32":
        if cpu.dtype != torch.float32:
            raise TypeError(f"expected f32 tensor, got {cpu.dtype}")
        return cpu.numpy().astype("<f4", copy=False)
    if dtype_name == "i32":
        if cpu.dtype != torch.int32:
            raise TypeError(f"expected i32 tensor, got {cpu.dtype}")
        return cpu.numpy().astype("<i4", copy=False)
    if dtype_name == "u32":
        if cpu.dtype != torch.uint32:
            raise TypeError(f"expected u32 tensor, got {cpu.dtype}")
        return cpu.numpy().astype("<u4", copy=False)
    raise ValueError(f"unsupported bundle dtype: {dtype_name}")


def attention_instructions(schedule, partitions: int) -> list[mk_instructions.Instruction]:
    """Expand upstream's single-partial schedule without changing other operators."""
    instructions = schedule.get_linear_instructions()
    if partitions == 1:
        return instructions
    globs = schedule.globs
    globs.skip_attn_reduction = False
    gqa_ratio = globs.num_attention_heads // globs.num_kv_heads
    expanded = []
    for instruction in instructions:
        if not isinstance(instruction, PartialAttention):
            expanded.append(instruction)
            continue
        expanded.extend(
            replace(instruction, num_partials=partitions, partial_idx=partial)
            for partial in range(partitions)
        )
        # The linear schedule is topological; each terminal reduction follows
        # all of its partials and precedes the layer's O-projection.
        for head in range(
            instruction.kv_head_idx * gqa_ratio,
            (instruction.kv_head_idx + 1) * gqa_ratio,
            globs.attn_reduction_size,
        ):
            expanded.append(
                AttentionReduction(
                    layer_idx=instruction.layer_idx,
                    head_start_idx=head,
                    num_partials=partitions,
                    is_terminal=True,
                    reduction_list=list(range(partitions)),
                )
            )
    return expanded


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    sm_count_patched = configure_sm_count(device)

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)

    print(f"building seeded model on {device}...", flush=True)
    config = make_config()
    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_len_override=args.max_seq_len,
        max_batch_size=1,
    )
    model = LlamaForCausalLM(config, extra_config)
    initialize_model(model)
    model.to(device=device)
    model.requires_grad_(False)

    with torch.no_grad():
        model.model.interleave_rope()
        model.stack_params()
    model.setup_caches()

    if model.device != device:
        raise RuntimeError(f"model device bookkeeping is {model.device}, expected {device}")
    if model.model.rope_cos.device != device:
        raise RuntimeError("model buffers did not end up on the requested device")

    layer_description = (
        "all layers + lm_head"
        if args.layer_limit is None or args.layer_limit == NUM_LAYERS
        else f"layers [0,{args.layer_limit}) without lm_head"
    )
    print(
        f"building and tensorizing the 148-SM latency schedule ({layer_description})...",
        flush=True,
    )
    schedule = latency_scheduler.LatencyScheduleBuilder.build(
        model, layer_limit=args.layer_limit
    )
    globs = schedule.globs
    if globs.sm_count() != NUM_SMS:
        raise RuntimeError(f"schedule has {globs.sm_count()} SMs, expected {NUM_SMS}")
    for pos_id in range(args.prompt_len, args.prompt_len + args.ntok):
        blocks = math.ceil((pos_id + 1) / globs.attn_kv_block_size)
        blocks_per_partial = math.ceil(blocks / args.attention_partitions)
        if (args.attention_partitions - 1) * blocks_per_partial >= blocks:
            raise ValueError(f"attention partition is empty at pos_id={pos_id}")
    linear_instructions = attention_instructions(schedule, args.attention_partitions)
    assigned = assign_to_sms("rr", instructions=linear_instructions, sm_count=NUM_SMS)
    tensorize_instructions(globs, assigned)
    if globs.instructions is None:
        raise RuntimeError("tensorize_instructions did not populate globs.instructions")
    num_instructions = int(globs.instructions.shape[1])
    opcodes = globs.instructions[..., 0]
    opcode_values, opcode_frequencies = torch.unique(opcodes, return_counts=True)
    opcode_histogram = {
        int(opcode): int(frequency)
        for opcode, frequency in zip(opcode_values.tolist(), opcode_frequencies.tolist())
    }
    layer_words = globs.instructions[..., 1][
        torch.isin(opcodes, torch.tensor(tuple(LAYER_OPCODES), device=device))
    ]
    layer_values, layer_frequencies = torch.unique(layer_words, return_counts=True)
    layer_histogram = {
        int(layer): int(frequency)
        for layer, frequency in zip(layer_values.tolist(), layer_frequencies.tolist())
    }
    if args.layer_limit == 1:
        unexpected_opcodes = sorted(set(opcode_histogram) - ({0} | LAYER_OPCODES))
        if unexpected_opcodes:
            raise RuntimeError(
                f"one-layer schedule contains unexpected opcodes: {unexpected_opcodes}"
            )
        if set(layer_histogram) != {0}:
            raise RuntimeError(
                "one-layer schedule contains nonzero layer indices: "
                f"{layer_histogram}"
            )
    histogram_text = ", ".join(
        f"{OPCODE_NAMES[opcode] if opcode < len(OPCODE_NAMES) else 'unknown'}"
        f"({opcode})={count}"
        for opcode, count in sorted(opcode_histogram.items())
    )
    print(f"instruction opcode histogram: {histogram_text}", flush=True)
    print(
        "layer-index histogram for layer-carrying opcodes: "
        + ", ".join(f"{layer}={count}" for layer, count in sorted(layer_histogram.items())),
        flush=True,
    )

    seq_len = args.prompt_len + args.ntok
    with torch.no_grad():
        globs.k_cache[:, :, :seq_len].normal_()
        globs.v_cache[:, :, :seq_len].normal_()
        globs.k_cache[:, :, seq_len:].normal_(std=100.0)
        globs.v_cache[:, :, seq_len:].normal_(std=100.0)

        token_ids = torch.randint(
            VOCAB_SIZE, (args.ntok,), dtype=torch.long, device=device
        )
        embedding_rows = []
        for token_id in token_ids:
            state = BatchState(input_ids=token_id.reshape(1, 1))
            embedded = model.model.embed_tokens(state).hidden_states
            if embedded is None:
                raise RuntimeError("embedding module produced no hidden states")
            embedding_rows.append(embedded.reshape(HIDDEN_SIZE))
        embeddings = torch.stack(embedding_rows)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []

    def emit(name: str, tensor: torch.Tensor, dtype_name: str) -> None:
        path = out / f"{name}.bin"
        array = little_endian_array(tensor, dtype_name)
        array.tofile(path)
        expected_bytes = math.prod(tensor.shape) * array.dtype.itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"short write for {name}: {actual_bytes} bytes, expected {expected_bytes}"
            )
        dims = " ".join(str(int(dim)) for dim in tensor.shape)
        manifest_lines.append(
            f"{name} {dtype_name} {tensor.ndim} {dims} {path.name}"
        )
        print(f"wrote {name}: {actual_bytes} bytes", flush=True)

    # Write caches before the VM mutates the decode rows: these files are the initial state.
    common_tensors = [
        ("qkv_weights", globs.qkv_proj_weights, "bf16"),
        ("attn_norm_weights", globs.attn_ln_weights, "bf16"),
        ("o_weights", globs.o_proj_weights, "bf16"),
        ("mlp_norm_weights", globs.mlp_ln_weights, "bf16"),
        ("up_weights", globs.up_proj_weights, "bf16"),
        ("gate_weights", globs.gate_proj_weights, "bf16"),
        ("down_weights", globs.down_proj_weights, "bf16"),
        ("lm_head_norm_weights", globs.lm_head_norm_weights, "bf16"),
        ("lm_head_weights", globs.lm_head_weights, "bf16"),
        ("k_cache", globs.k_cache[:, 0], "bf16"),
        ("v_cache", globs.v_cache[:, 0], "bf16"),
        ("rope_cos", globs.rope_cos[: args.max_seq_len], "f32"),
        ("rope_sin", globs.rope_sin[: args.max_seq_len], "f32"),
        ("instructions", globs.instructions, "i32"),
        ("embeddings", embeddings, "bf16"),
    ]
    for name, tensor, dtype_name in common_tensors:
        emit(name, tensor, dtype_name)

    interpreter = PyVM_Interpreter(INSTRUCTION_TO_SOLVER)
    print(
        f"running PyVM reference: {args.ntok} tokens, "
        f"{len(linear_instructions)} linear instructions/token...",
        flush=True,
    )
    for t in range(args.ntok):
        pos_id = args.prompt_len + t
        with torch.no_grad():
            globs.hidden_states[:] = embeddings[t]
            globs.barriers.zero_()
            globs.pos_id = pos_id
            interpreter.interpret(globs, linear_instructions)

        references = [
            ("hidden_states", globs.hidden_states, "bf16"),
            ("logits", globs.logits, "bf16"),
            ("q_post_rope", globs.post_ln_rope_q, "bf16"),
            ("attn_out", globs.attn_out, "bf16"),
            ("silu_out", globs.silu_out, "bf16"),
            ("k_row", globs.k_cache[:, 0, pos_id], "bf16"),
            ("v_row", globs.v_cache[:, 0, pos_id], "bf16"),
            ("attn_out_p0", globs.attn_out_intermediates[:, 0, :], "f32"),
            ("attn_lse_p0", globs.attn_lse_intermediates[:, 0], "f32"),
        ]
        for name, tensor, dtype_name in references:
            emit(f"ref_tok{t}_{name}", tensor, dtype_name)
        print(f"completed reference token {t} at pos_id {pos_id}", flush=True)

    rope_rows = min(args.max_seq_len, int(globs.rope_cos.shape[0]))
    scalars = {
        "num_sms": NUM_SMS,
        "num_instructions": num_instructions,
        "layer_limit": (
            0
            if args.layer_limit is None or args.layer_limit == NUM_LAYERS
            else args.layer_limit
        ),
        "instruction_opcode_histogram": ",".join(
            f"{opcode}:{count}" for opcode, count in sorted(opcode_histogram.items())
        ),
        "instruction_layer_histogram": ",".join(
            f"{layer}:{count}" for layer, count in sorted(layer_histogram.items())
        ),
        "max_seq_len": args.max_seq_len,
        "rope_rows": rope_rows,
        "num_layers": NUM_LAYERS,
        "bar_rows": BAR_ROWS,
        "vocab_size": VOCAB_SIZE,
        "attn_scale": globs.attn_scale,
        "rms_norm_eps": globs.rms_norm_eps,
        "skip_attn_reduction": int(globs.skip_attn_reduction),
        "attention_partitions": args.attention_partitions,
        "prompt_len": args.prompt_len,
        "ntok": args.ntok,
        "pos_id_first": args.prompt_len,
        "seed": args.seed,
        "reference_device": str(device),
    }
    scalar_text = "".join(f"{key} {value}\n" for key, value in scalars.items())
    (out / "scalars.txt").write_text(scalar_text, encoding="utf-8")

    manifest_text = "\n".join(manifest_lines) + "\n"
    manifest_tmp = out / f".manifest.txt.tmp-{os.getpid()}"
    manifest_tmp.write_text(manifest_text, encoding="utf-8")
    manifest_tmp.replace(out / "manifest.txt")

    elapsed = time.monotonic() - started
    bundle_bytes = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    print(
        f"bundle complete: out={out} reference_device={device} "
        f"elapsed_seconds={elapsed:.3f} bytes={bundle_bytes} "
        f"sm_count_patched={str(sm_count_patched).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
