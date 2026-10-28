# PipeLink documentation

A **stage** is an independently packaged CUDA translation unit: an opaque LTO
object plus a manifest that declares entry roles, parameters, resources and contracts.
A stage need not own scheduled work; a callable provider runs on its caller's threads.
Owners declare what their code promises, and the compiler derives layouts, symbols and
bindings.

PipeLink trusts what `pack` attests. `pack` checks the object's symbols against the
declarations and records an attestation; `link` relies on it while checking
declarations, compatibility and resources. A tampered or inaccurate manifest therefore
goes undetected at link time. `inspect` can verify the payload digest on request, but
that check is optional and adds no guarantee beyond the attestation.

Start with [T0](tutorials/T0-pack-link.md) and the [compiler trust model](reference/pack-and-link.md#trust-model).

## Prerequisites

- **Linux x86_64**, Python 3, `uv`, a C++17 host compiler (`g++`), and GNU `nm`.
- **CUDA 13.3**, including `nvcc`, **nvJitLink** and `cuobjdump`. Packing, linking,
  inspection and the contract-rejection gallery do not require a GPU.
- A compatible CUDA driver and a **Hopper/sm90**, **Blackwell/sm103** or
  **consumer Blackwell/sm120** GPU for native tutorial execution; Compute Sanitizer
  for the documented racecheck and synccheck runs. The registered profiles are
  `sm_90a`, `sm_100a`, `sm_103a`, `sm_120a` and `sm_121a`; a package targets one profile.
  `sm_100a` and `sm_121a` mirror `sm_103a` and `sm_120a` for compute capability 10.0
  and 12.1 parts. They are tested through pack and link only; native results come
  from the other three.
- A configured machine-local `.env`, starting from [`.env.example`](../.env.example),
  and a writable `PIPELINK_SCRATCH_ROOT`. Follow [T0's setup](tutorials/T0-pack-link.md#goal-and-prerequisites);
  if you already have a configuration, keep it. The commands run in per-project uv
  environments (`uv run --project ...`), so no global uv environment is needed.
- **T4 and T6:** the pinned, patched **TileLang 0.1.14 wheel**, its separate
  frontend uv environment, and the CUTLASS and mixture-of-kittens submodules. The
  lock file points to a local
  wheel; if you do not have it, build it with the
  [Linux x86_64 source-build recipe](../frontends/tilelang/env/README.md).
  A stock TileLang wheel lacks the patches and cannot stand in for it. T4 needs this
  backend for non-alias tensor adapters even with a hand-written CUDA consumer.
  Their CODA acceptance run creates its own public-baseline environment on first use;
  the attention case study needs a separate one.

Run tutorial commands from the repository root. Each tutorial states its own
build inputs, environment setup and native execution requirements.

## Tutorial ladder

| Tutorial | Goal | GPU requirement |
|---|---|---|
| [T0 — Install, pack, link, read `link.json`](tutorials/T0-pack-link.md) | Package two stages, inspect their manifests, link a cubin and read launch requirements. | None; CUDA toolkit only. |
| [T1 — Hello, callable seam](tutorials/T1-hello-callable.md) | Follow `PL_EXPORT` / `PL_CALL`, call-traffic direction, parameter blobs and host-less launch. | Hopper/sm90, Blackwell/sm103 or sm120 to run. |
| [T2 — A pipeline through an mbarrier ring](tutorials/T2-ring-pipeline.md) | TMA → shared-memory ring → compute, role placement, resource claims and link-bound depth. | Hopper/sm90, Blackwell/sm103 or sm120 to run. |
| [T3 — Swap a stage; break a contract](tutorials/T3-contract-gallery.md) | Reuse a producer, swap a consumer and explain six named contract rejections. | None; compilation and link checks only. |
| [T4 — Let the linker generate the data path](tutorials/T4-tensor-import.md) | One hand-written SwiGLU consumer imports GEMM accumulators from TMEM or registers ([CODA](../campaigns/coda/README.md)). | Hopper/sm90, Blackwell/sm103 or sm120 to run; build/link are GPU-free. |
| [T5 — Ship a stage without source: host ABI v3](tutorials/T5-host-abi.md) | Owner provider `.so`, public C arguments, source-withheld consumer, persistent session and CUDA graph. | Hopper/sm90 for the full walkthrough; Blackwell/sm103 and sm120 have the separate provider-launch route. |
| [T6 — Author a stage in TileLang](tutorials/T6-tilelang.md) | Derive a contract-authored consumer from partner manifests, without provider headers or hand-written macros. | Hopper/sm90, Blackwell/sm103 or sm120 to run; lowering and build/link are GPU-free. |
| [T7 — Overlap communication with compute across GPUs](tutorials/T7-comm-overlap.md) | Link a communication owner's stage into T5's GEMM through a system-scope counter, a CTA read-drain signal and caller-owned peer memory. | Two to four Hopper/sm90 GPUs with peer access to run; build/link are GPU-free. |
| [T8 — Case studies](tutorials/T8-case-studies.md) | Read attention reuse under absolute-performance comparisons, context-parallel attention across GPUs and whole-model megakernel composition. | None to read; reproducing a case study needs the GPU it names. |

## Mechanism × profile coverage

Each cell names the example that exercises a mechanism on a profile. One linked
kernel uses one profile; a shared source is packed separately for each target.
Native results and their conditions are in the linked tutorials and case-study
reports.

| Seam mechanism | Hopper/sm90 (`sm_90a`) | Blackwell/sm103 (`sm_103a`) | Consumer Blackwell/sm120 (`sm_120a`) |
|---|---|---|---|
| A. Callable: `PL_EXPORT` / `PL_CALL`, by-value trampolines | [T1: hello-callable](tutorials/T1-hello-callable.md) | [T1: the same source pair](tutorials/T1-hello-callable.md) | [T1: the same source pair](tutorials/T1-hello-callable.md) |
| B. mbarrier channel: TMA → shared-memory ring → compute; depth bound at link | [T2: ring-pipeline](tutorials/T2-ring-pipeline.md) | [T2: the same WGMMA-free source](tutorials/T2-ring-pipeline.md), packed with `--profile sm_103a` | [T2: the same source](tutorials/T2-ring-pipeline.md), packed with `--profile sm_120a` |
| C. Stage swap and negative diagnostics | [T3: contract-gallery](tutorials/T3-contract-gallery.md), GPU-free | [T3: contract-gallery](tutorials/T3-contract-gallery.md), GPU-free | [T3: contract-gallery](tutorials/T3-contract-gallery.md), GPU-free |
| D. Tensor import: linker-generated adapter | [T4: CODA hand SwiGLU consumer](tutorials/T4-tensor-import.md), WGMMA registers, staged through shared memory | [T4: the same consumer source](tutorials/T4-tensor-import.md), TMEM receipts | [T4: the same consumer source](tutorials/T4-tensor-import.md), `mma.sync` registers, staged through shared memory |
| E. Host ABI v3: owner `.so`, source withholding, session and graph | [T5: hopper-gemm-split](tutorials/T5-host-abi.md), including source-withheld session and graph replay | [T5: profile-consumer provider launch](tutorials/T5-host-abi.md); the source-withheld session and graph walkthrough is sm90-only. | [T5: profile-consumer provider launch](tutorials/T5-host-abi.md), as on sm103 |
| F. Frontend-authored stage | [T6: CODA contract-authored TileLang SwiGLU consumer](tutorials/T6-tilelang.md), `import_tensor(Partner.load(...))`, specialized to a direct register copy | [T6: the same recipe](tutorials/T6-tilelang.md), re-lowered against the TMEM producer for `sm_103a` | [T6: the same recipe](tutorials/T6-tilelang.md), re-lowered for `sm_120a` |
| G. Interpreter / megakernel-lite: descriptors, phase-exclusive pages, grid counter | [profile-consumer](../examples/profile-consumer/README.md) | [profile-consumer](../examples/profile-consumer/README.md) | [profile-consumer](../examples/profile-consumer/README.md) |
| H. Multi-GPU: system-scope counter channels, named communication domains with peer and multicast host views, multi-device sessions | [T7: comm-overlap](tutorials/T7-comm-overlap.md), AllGather→GEMM and GEMM→ReduceScatter; [attention `cp`](../campaigns/attention/README.md#context-parallel-causal-attention-cp) | [multimem-domains](../examples/multimem-domains/README.md): a multicast `tp` domain and a peer `dp` domain in one kernel (needs NVLink multicast); the comm-overlap GEMM is sm90-only. | **No tutorial on this profile**; the pipec vocabulary is profile-independent. |
| Case study: attention reuse and absolute placement | [Attention sm90 tour](../campaigns/attention/TOUR-sm90.md): six compositions reuse one core | **Hopper-only case study.** [T1: callable](tutorials/T1-hello-callable.md) and [T4: one consumer, three producers](tutorials/T4-tensor-import.md) show its seams on this profile. | **Hopper-only case study**, as on sm103. |
| Case study: whole-model megakernel | **Rejected on this profile:** the interpreter claims TMEM (`L-HANDLE`); the port targets Blackwell and has no Hopper implementation. | [Megakernel tour](../campaigns/megakernel/TOUR.md) | **Rejected on this profile:** no TMEM (`L-HANDLE`). |

### Profile-specific boundaries

- [`hopper-gemm-split`](../examples/hopper-gemm-split/README.md) is **sm90-only**.
  Its `mma.cu` uses Hopper WGMMA and requires `CUTE_ARCH_MMA_SM90A_ENABLED`, so
  packing for `sm_103a` fails with **`P-COMPILE`**. The limit comes from this GEMM,
  not from the host ABI.
- TMEM claims are **rejected on `sm_90a` and `sm_120a` (`L-HANDLE`)**. There, T4's producers
  export registers instead.
- CLC scheduling (`clc-sm100-v1`) and `cta_group: 2` are **rejected on `sm_90a` and
  `sm_120a` (`L-TOPO`)**. Choosing a profile does not translate architecture-specific
  CUDA instructions or port an owner's kernel.

On every profile, a tensor import hands over one generation of its export at a time.
T2's mbarrier ring is a different mechanism: when stages need several generations in
flight, write a multi-slot ring like it in stage code over a channel endpoint
([T2](tutorials/T2-ring-pipeline.md)).

## Testing

[Tutorial acceptance tests](tutorials/test_tutorials.py) run T0 and T3 for every
registered profile without a GPU:

```sh
uv run --project pipec --locked python -m pytest docs/tutorials/test_tutorials.py -q
```

[CI](../.github/workflows/ci.yml) also runs the compiler and evaluator tests and
packs each example for the profiles it supports.
GPU tests skip unless `PIPELINK_TUTORIAL_GPU` names a GPU; no CI runner has one.
To cover them, the maintainers run
[`bash tools/nightly.sh GPU_INDEX`](../tools/nightly.sh) by hand, which builds and runs
T1, T2, the CODA cells of T4/T6 and the matching T5 route on one Hopper/sm90 or
Blackwell/sm103 GPU. The CODA test requires `PIPELINK_TILELANG_PYTHON` and
`TILELANG_INCLUDE_DIR` as in
[T4's environment setup](tutorials/T4-tensor-import.md#3-build-the-cells-of-a-profile);
without them it fails with a prerequisite message.

## Reference

Start from the [compiler README](../pipec/README.md) for installation, the
command-line workflow and the trust model.

- [Device annotations](reference/annotations.md): params, roles, calls, endpoints
  and protocols, claims and lifetimes, topology and variants.
- [Host ABI](reference/host-abi.md): owner providers, host annotations, scheduler
  schemas and transfer reports.
- [Tensor imports](reference/tensor-imports.md): authoring, manifest contracts,
  planning and the link flow of generated data paths.
- [Pack and link](reference/pack-and-link.md): pack recipe, manifest provenance,
  link stages and the resource solver.
- [Launching](reference/launching.md): host-less and host-provider launches,
  sessions, paired control images and multi-device sessions.
- [Clause catalogue](reference/clauses.md): IDs, primary checking phases and
  descriptions from the live compiler.
- [Profiles](reference/profiles.md): target spellings, caps, resource domains,
  scheduling vocabulary, layout atoms and adapter compile requirements.

The clause catalogue and profiles pages are generated. Regenerate them from the
repository root without a GPU:

```sh
uv run --project pipec --locked python docs/reference/generate.py
```

## Scope

Generated transport is exact and stays within one CTA; generated code performs
no arithmetic and leaves the scheduling of opaque stages alone. Traffic between
CTAs or GPUs is stage code; the linker checks and binds its counters and peer memory
([T7](tutorials/T7-comm-overlap.md)). Tensor imports carry one generation per
export; write multi-slot rings in stage code over a channel endpoint, as in
[T2](tutorials/T2-ring-pipeline.md). Opaque stages are trusted to honor their
declarations, so native numerical checks and sanitizers decide acceptance. See
[Contributing](../CONTRIBUTING.md) for the required checks.

The performance gate compares linked execution with the **same stage sources in
one translation unit**, which isolates the cost of linking. Comparisons against
public implementations are reported in full, including losses, but do not gate.
SASS comparisons help explain differences and stand in for none of the numerical,
sanitizer or timing evidence.
