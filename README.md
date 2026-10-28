<h1 align="center"><img src="docs/assets/pipelink-logo.png" alt="" width="56" align="absmiddle">PipeLink</h1>

<p align="center">
  English | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <b>A linker for partial GPU kernels.</b><br>
  Package CUDA pipeline stages separately, then link them into one fused kernel —<br>
  with contracts checked and resources bound at link time.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="CUDA 13.3" src="https://img.shields.io/badge/CUDA-13.3-76b900">
  <img alt="GPUs: Hopper and Blackwell" src="https://img.shields.io/badge/GPU-Hopper%20%7C%20Blackwell-555">
  <a href="https://arxiv.org/abs/2610.37200"><img alt="arXiv: 2610.37200" src="https://img.shields.io/badge/arXiv-2610.37200-b31b1b"></a>
</p>

<p align="center">
  <img src="docs/assets/pipelink-overview.svg" alt="Stages written with CUDA, CUTLASS or TileLang are packed into .plink packages; pipec2 link checks their contracts, solves resources and roles, generates adapters and emits one fused kernel" width="100%">
</p>

A fused GPU kernel is usually one translation unit written by one team. With PipeLink,
its stages (a load stage, an MMA mainloop, an epilogue, a stage that exchanges data with
other GPUs) can come from different owners and toolchains, even as binary-only packages,
and still run as **one kernel** at the speed of the same code compiled together.

- 🧩 **Compose stages you did not write.** Each stage declares what it provides and needs;
  the linker checks every seam and rejects a mismatch with a named diagnostic.
- 🔀 **Mix languages and libraries.** Hand-written CUDA, CUTLASS/CuTe, ThunderKittens and
  TileLang stages link into one kernel: a TileLang epilogue consumes a CUTLASS
  accumulator as easily as its own, and a CuTe attention op runs inside a ThunderKittens
  megakernel.
- 🔒 **Keep your source.** Ship a stage as a `.plink` package (LTO IR + manifest), plus a
  provider library and a C header if it needs host-side setup. Users compose it without
  seeing the code.
- ⚙️ **Let the linker do the plumbing.** The linker solves warp roles, shared memory, TMEM,
  registers, barriers and ring depths, and generates tensor hand-offs between TMEM, shared
  memory and registers. Across GPUs, it checks the readiness counters between stages and
  binds the peer memory the caller provides.
- 🚀 **Pay nothing for the seam.** Across the showcases below, linked kernels are at most
  0.2% slower than the same sources compiled as one unit.

> Paper: [PipeLink: Contract-Guided Linking of GPU Kernel Components](https://arxiv.org/abs/2610.37200)

## What you can build

| | Capability | Learn |
|---|---|---|
| 🔌 | **Callable seams** — call device functions exported by other packages | [T1](docs/tutorials/T1-hello-callable.md) |
| 🔁 | **Pipelines** — connect stages through mbarrier rings whose depth is set at link time | [T2](docs/tutorials/T2-ring-pipeline.md) |
| 🛡️ | **Contract checking** — swap a stage; if its contracts do not match, the link fails with `L-PROTO`, `L-HANDLE`, `L-PHASE`… instead of the kernel hanging | [T3](docs/tutorials/T3-contract-gallery.md) |
| 🧮 | **Generated tensor transport** — import an accumulator from TMEM or registers without knowing where it lives | [T4](docs/tutorials/T4-tensor-import.md) |
| 📦 | **Binary-only stages** — ship a package and a C header, with persistent sessions and CUDA graph capture | [T5](docs/tutorials/T5-host-abi.md) |
| 🐍 | **TileLang stages** — author in TileLang, link against opaque CUDA | [T6](docs/tutorials/T6-tilelang.md) |
| 🌐 | **Multi-GPU overlap** — link another owner's communication stage into a compute kernel; NVLink transfers overlap compute inside one kernel per GPU | [T7](docs/tutorials/T7-comm-overlap.md) |

## Showcases

### A whole-model megakernel that mixes ThunderKittens and CUTLASS

The 16-layer Llama-1B decode megakernel from
[Megakernels](https://github.com/HazyResearch/Megakernels), written with ThunderKittens,
is split into nine packages (an instruction interpreter plus eight ops) and relinked into
one persistent kernel. On sm103 with the SM clock locked at **1155 MHz**, it reaches
**0.9996×** the throughput of the same sources compiled as one unit
(1.149 ms/token, 1024-token reduced-KV fixture), with zero memcheck, racecheck and synccheck findings.

**Then CUTLASS joins the kernel.** Swapping only the attention package for a CUTLASS/CuTe
version leaves the other eight ThunderKittens packages, the instruction schedule and the
host code untouched. At a 4096-token context the mixed kernel decodes at
**1.630 ms/token**, **2.16×** the ThunderKittens-only throughput, and stays within 0.02%
of its own one-unit build.
[Tour →](campaigns/megakernel/TOUR.md)

### An expert-parallel MoE training megakernel, owned by four teams

[Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens), a deterministic
expert-parallel MoE training megakernel, is split into a framework interpreter and three
op packages (communication, GEMM and activation), each owning its descriptors, shared
memory and synchronization. One set of op packages links into all 24 kernels (forward,
backward and recompute; BF16 and MXFP8; one or four GPUs; clamped or not). The 12 timed
four-GPU kernels run at 0.9988–1.0007× the same sources compiled as one unit and at
0.989–1.014× upstream MoK's whole step. A communication-only tile change relinks without
touching the other owners, and the activation owner swaps in a GPT-OSS SwiGLU without the
other owners' sources.
[Case study →](campaigns/mok/README.md)

### One SwiGLU consumer, three GPU families

[CODA](campaigns/coda/README.md) fuses Qwen3-32B's MLP GEMM with its SwiGLU activation.
Each GPU family has its own separately packaged GEMM producer, which keeps the accumulator
where its hardware puts it:

- **Datacenter Blackwell:** a 2-SM tcgen05 GEMM with the accumulator in TMEM.
- **Hopper:** a three-warpgroup WGMMA GEMM with the accumulator in registers.
- **Consumer Blackwell:** an `mma.sync` GEMM with the accumulator in registers.

Every producer links against one hand-written CUDA SwiGLU consumer and one TileLang
consumer. The linker generates each hand-off, so neither consumer learns where the
accumulator lives. The TileLang consumer can also specialize per producer, much like
profile-guided optimization: it adopts the producer's layout from that producer's
manifest, which turns the register hand-offs into plain register copies. Across the 12
timed kernels (two consumers and two batch sizes on each family), none is more than 0.13%
slower than the same sources compiled as one unit. Against cuBLAS followed by a separate
SwiGLU, the linked kernels are 1.20–1.21× faster when that SwiGLU comes from
`torch.compile` and 1.34–1.37× faster when it comes from Liger on datacenter Blackwell,
1.03–1.08× and 1.12–1.18× on Hopper, and
1.01–1.03× and 1.05–1.08× on consumer Blackwell; against hand-fused kernels they range
from 0.94× to 1.01×. [Case study →](campaigns/coda/README.md)

### One attention core, seven variants

One optimized dense attention core, derived from FlashAttention-3 on Hopper, serves very
different attention kernels: each links a few new stages around the same unchanged core
package. A sliding window is a 19-line mask, and softcapping is a 20-line score function.
Even kernels that schedule their work very differently, such as MiniMax-M3 sparse
attention and attention split across GPUs, need only two new stages each, and they stay
competitive:

- **MiniMax-M3 sparse attention (MSA)** runs at 0.94–38× the speed of FA4, 1.5–38× that
  of FlexAttention and 1.6–4.5× that of
  [vLLM's dedicated Triton kernel](https://github.com/vllm-project/vllm/blob/3137ff07739f0cd19e06119506c6432a4a7e77ad/vllm/models/minimax_m3/common/ops/sparse_attn.py#L514).
  A new load stage visits only the selected key blocks; a new epilogue writes partial
  results that a small kernel combines. FA4 and FlexAttention apply the selection as a mask
  over query tiles, so each tile computes every block that any of its queries selected:
  FA4 is 6% faster when the selected blocks form a prefix and up to 38× slower when they
  are scattered.
- **Context-parallel attention with in-kernel communication** speeds up causal prefill
  over 32K tokens by 1.95× on two GPUs and 3.67× on four versus one GPU, and runs
  1.20–1.31× faster than Transformer Engine's context parallelism at 8K and 32K tokens on
  the same GPUs. A communication owner's stage pushes key/value shards to the other GPUs
  over NVLink, and the load stage waits for each remote block just before loading it, so
  no separate gather or output-correction pass is needed.

<p align="center">
  <img src="docs/assets/attention-reuse.svg" alt="Seven Hopper attention variants share one unchanged core package; each row swaps only small load, mask, score, epilogue or extra-warp stages, from a 19-line mask for sliding window to a selected-block load and partial epilogue for MiniMax-M3 sparse attention and a communication owner's KV-gather stage for context-parallel attention" width="100%">
</p>

On dense and sliding-window attention the linked kernels reach 84–99% of FA3 and match
it on chunked attention; on Phi-3 block-sparse attention they are 2.3–2.7× faster than
the fastest public kernel we measured. The remaining gaps to FA3 come from our core, which
every variant inherits; linking itself measures 1.000× against the same sources compiled
as one unit.
[Case study →](campaigns/attention/README.md)

## Get started

Requirements: Linux x86_64, CUDA 13.3 (`nvcc`, nvJitLink), a C++17 host compiler, Python 3
and [`uv`](https://docs.astral.sh/uv/). Packing and linking need no GPU. Running needs a
Hopper, datacenter Blackwell or consumer Blackwell GPU; T7 needs two to four Hopper GPUs
with peer access.

```sh
git clone --recursive https://github.com/alumkal/PipeLink.git && cd PipeLink
cp .env.example .env            # set PIPELINK_CUDA_HOME, PIPELINK_SCRATCH_ROOT and PIPELINK_GPUS
source tools/env.sh
uv run --project pipec --locked pipec2 --help
```

### Your first composition

A provider exports a device function; a caller uses it without seeing its source.

```cpp
// provider.cu
extern "C" __device__ float hello_scale(float value) { return value * 2.0f; }

PL_STAGE("org.hello.provider", "1.0.0");
PL_NO_PARAMS();
PL_EXPORT(scale, hello_scale);
PL_CALL(scale, hello_scale, pl::consume, "scale");
```

```cpp
// caller.cu (simplified)
extern "C" __device__ float pipelink_call_scale(float value);   // resolved at link time
extern "C" __device__ void hello_call(const void* raw, unsigned) {
    const auto* p = static_cast<const HelloCallerParams*>(raw);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < p->count; i += blockDim.x * gridDim.x)
        p->output[i] = pipelink_call_scale(p->input[i]);
}

PL_STAGE("org.hello.caller", "1.0.0");
PL_PARAMS(HelloCallerParams, input, output, count);
PL_ROLE(call, hello_call, 4);                                    // four warps
PL_CALL(scale, pipelink_call_scale, pl::produce, "call");
```

```sh
# Pick your GPU's profile: sm_90a, sm_100a, sm_103a, sm_120a or sm_121a.
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_120a   # pipec2 pack ×2, pipec2 link
uv run --project pipec --locked python examples/hello-callable/run.py   --profile sm_120a   # launch, check, time
```

`pipec2 link` derives the launch geometry, the params layout and the call trampoline, so the
host needs no hand-written launcher. Walk through it in [T1](docs/tutorials/T1-hello-callable.md).

### Tutorials

| Tutorial | GPU to run it |
|---|---|
| [T0](docs/tutorials/T0-pack-link.md) · Pack, inspect and link | None |
| [T1](docs/tutorials/T1-hello-callable.md) · Hello, callable seam | Any supported GPU |
| [T2](docs/tutorials/T2-ring-pipeline.md) · A pipeline through an mbarrier ring | Any supported GPU |
| [T3](docs/tutorials/T3-contract-gallery.md) · Swap a stage, break a contract | None |
| [T4](docs/tutorials/T4-tensor-import.md) · Generated tensor transport | Any supported GPU |
| [T5](docs/tutorials/T5-host-abi.md) · Ship a stage without source | Hopper for the full walkthrough |
| [T6](docs/tutorials/T6-tilelang.md) · Author a stage in TileLang | Any supported GPU |
| [T7](docs/tutorials/T7-comm-overlap.md) · Overlap communication with compute across GPUs | Two to four Hopper GPUs |
| [T8](docs/tutorials/T8-case-studies.md) · Case studies | None to read |

T4 and T6 need the [patched TileLang backend](frontends/tilelang/env/README.md).
The [documentation index](docs/index.md) covers what each profile supports.

## Scope

Generated code moves data exactly within one CTA and performs no arithmetic. Traffic
between CTAs or GPUs is written in stage code; the linker checks and binds its counters,
signals and peer or multicast memory over caller-named communication domains
([T7](docs/tutorials/T7-comm-overlap.md), [multimem-domains](examples/multimem-domains/README.md)). A tensor
import hands over one generation of its export at a time; to keep several in flight,
write a multi-slot ring in stage code ([T2](docs/tutorials/T2-ring-pipeline.md)). The
linker checks declarations, not the opaque code behind them, so every composition still
needs numerical checks and sanitizers.

## Repository

| Path | Contents |
|---|---|
| [`pipec/`](pipec/README.md) | Compiler, package format and `pipec2` CLI |
| [`frontends/tilelang/`](frontends/tilelang/README.md) | TileLang authoring frontend |
| [`examples/`](examples/) | Teaching compositions |
| [`campaigns/`](campaigns/) | Attention, megakernel and MoE training case studies |
| [`evaluator/`](evaluator/) | Parity and evidence harness |
| [`docs/`](docs/index.md) | Tutorials and references |

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). PipeLink is licensed
under [Apache-2.0](LICENSE); third-party portions keep their own licenses ([NOTICE](NOTICE)).

## Citation

If you use PipeLink in your research, please cite:

```bibtex
@misc{guo2026pipelink,
  title         = {PipeLink: Contract-Guided Linking of GPU Kernel Components},
  author        = {Jize Guo and Xupeng Miao},
  year          = {2026},
  eprint        = {2610.37200},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```
