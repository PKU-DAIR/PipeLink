# Launching linked kernels

`pipec2 link` writes `pipeline.cubin` and `link.json`. The launcher reads the link
and picks one of two routes:

- **Host-less**: links without a `host` section. The caller supplies the grid,
  buffers and each stage's params blob.
- **Host ABI v3**: links with a `host` section. Owner providers build the params,
  and the scheduler authority's `query` supplies the geometry.

Both routes use the same `CudaBackend`, aligned params image and
`cuLaunchKernelEx` argument passing. The block is always `(link.threads, 1, 1)`,
dynamic shared memory comes from the link, and `--cluster X,Y,Z` selects one of
the link's admitted clusters. `--device N` selects a visible CUDA ordinal.

## Host-less launch

```sh
uv run --project pipec pipec2 launch output/link.json \
  --grid 4,1,1 --buffer input=input.npy --buffer output=1024:float32 \
  --params params.py:build --output output=result.npy
```

| Option | Meaning |
|---|---|
| `--buffer NAME=FILE.npy` | Upload a C-order NumPy array (pickle loading is disabled). |
| `--buffer NAME=SHAPE:DTYPE` | Allocate a zeroed buffer, for example `out=1024:float32` or `matrix=64x64:uint16`. Use `uint16` for raw bf16 bit patterns; there is no implicit bf16 conversion. |
| `--params PY_FILE:FUNCTION` | Run a trusted Python builder that maps `{buffer: device_address}` to `{stage_id: bytes}`. It runs after every buffer is allocated and initialized. |
| `--blob STAGE_ID=PATH` | Supply an already-built params blob. |
| `--output NAME=FILE.npy` | Save a buffer after the launch. |

Pack each blob from the owner's published params header, including C padding;
the launcher has no field metadata and does not interpret pointers:

```python
import ctypes as C

class ScaleParams(C.Structure):
    _fields_ = [("input", C.c_uint64), ("output", C.c_uint64),
                ("count", C.c_uint32), ("scale", C.c_float)]

def build(addresses):
    return {"example.scale": bytes(ScaleParams(
        addresses["input"], addresses["output"], 1024, 2.0))}
```

Every stage with nonzero params in `link.params.entries` needs exactly one blob of
the right length; zero-param stages may be omitted or given `b""`. Unknown stages
and wrong lengths fail with `L-PARAMS`. Blobs are copied to their linked offsets
without per-field validation.

### Python interface

```python
import numpy as np
from pipec2.launcher import open_hostless, run_hostless
from params import build

buffers = {"input": np.arange(1024, dtype=np.float32),
           "output": ((1024,), np.float32)}
outputs = run_hostless("output/link.json", buffers=buffers, params=build,
                       grid=(4, 1, 1), outputs=["output"])

with open_hostless("output/link.json", buffers=buffers, params=build,
                   grid=(4, 1, 1)) as session:
    session.launch()
    batch_ms = session.time(20)        # 20 back-to-back launches, one event pair
    output = session.read("output")
```

- `run_hostless(..., outputs=None)` reads back every buffer.
- `session.addresses` maps buffer names to device addresses; `launch()` submits
  one launch on the session stream; `time(n)` returns the total milliseconds of `n`
  launches and synchronizes; `read(name)` synchronizes and returns a copy with the
  original shape and dtype.
- Allocations and params stay fixed until close. Buffers are initialized once and
  not reset between launches.
- Both functions accept `cluster=` and `cubin=`. Without `cubin`, the image is the
  link's `pipeline.cubin`; a `cubin=Path(...)` runs another image, such as a
  one-TU control, with the link's symbol, threads, shared memory and params layout.

For paired linked/control measurements, view the control image through the same
session instead of opening a second one:

```python
with open_hostless("output/link.json", buffers=buffers, params=build,
                   grid=(4, 1, 1)) as linked:
    with linked.with_image("output/one_tu.cubin") as control:
        linked_ms = linked.time(20)
        control_ms = control.time(20)
```

The view reuses the session's allocations, params and stream without calling the
builder, so allocation and descriptor differences cannot bias the ratio. Launches
of both images serialize on that stream. The session must outlive its views; read
each image's output before the next launch if both write the same buffer. Use the
shared parity harness (`evaluator/harness/parity/`) for warmup, calibration and
interleaved ordering rather than the sequence above.

`open_hostless` refuses links with a host section.

## Host ABI v3 launch

```sh
uv run --project pipec pipec2 launch output/link.json \
  --provider pipeline.owner.params=owner.so \
  --provider-args pipeline.owner.params=owner-args.bin \
  --host-inputs invocation.json
```

`--provider GROUP=SO` and `--provider-args GROUP=PATH` are keyed by params group,
not stage. Stages without host state but with nonzero params take `--blob`.
There is no grid override: geometry comes only from the scheduler authority.
The CLI opens one session, launches once, synchronizes and reads back the
declared outputs.

`invocation.json` holds exactly two objects, `scalars` and `resources`. Each
resource declares `data` (an input file or null), `allocation_bytes`, `extents`,
`bit_strides`, `dtype_id`, `alignment`, `origin_byte_offset`, `origin_bit_offset`
and an optional `output` file for readback. Paths are relative to the JSON file.
Inputs are explicit caller fixtures; providers never generate them.

### Sessions

`pipec2.host.broker.open_host` returns an immutable `HostSession`:

- **Open** loads each provider once, checks every ABI report before any query,
  builds the shared scalar image, resolves demands, selects geometry, allocates
  and zero-fills declared workspaces, and runs each owner's `build` into the
  pinned params image. Transfer reports are checked
  ([Host ABI](host-abi.md#transfer-reports)) before the context synchronizes.
  Host-less blobs in a mixed link are copied to their offsets.
- **`launch(stream)`** issues only stream work: zero-fills for `reset: launch`
  workspaces, then `cuLaunchKernelEx`. No allocation, encoding, synchronization,
  readback, driver query or provider call happens, so launch is repeatable and
  safe to capture in a CUDA graph. Changing params, resources, scalars or geometry
  means closing and reopening. `launch(stream, image=control_backend)` launches a
  second `CudaBackend` in the same context (for example the one-TU control) with
  this session's params, geometry and resets.
- **Close** drains the context, releases owner builds in reverse order, frees
  workspace and unloads providers. `reset: open` workspaces persist across
  launches; `reset: launch` requires `initializer: zero`.

```python
from pipec2.host.broker import open_host
from pipec2.host.cuda import CudaBackend
from pipec2.host.fixture import Fixture

with CudaBackend(cubin, symbol, dynamic_smem_bytes=smem, cluster=cluster,
                 block=(threads, 1, 1)) as backend, Fixture(backend) as fixture:
    a = fixture.tensor("a", data=a_bytes, allocation_bytes=len(a_bytes), ...)
    d = fixture.tensor("d", data=None, allocation_bytes=d_bytes, ..., output=True)
    stream = backend.stream_token
    with open_host(result, providers, args, fixture.resources, scalars, backend, cluster) as session:
        session.launch(stream)
        graph = backend.capture(stream, lambda: session.launch(stream))
        graph.launch(stream)
        backend.synchronize()
        outputs = fixture.readback()
```

Callers own inputs, their readiness on the launch stream and output readback.
`open_host(..., blobs={stage_id: bytes})` accepts the same mixed-link blobs as the
CLI; blobs never replace a host group's residue. `launcher.launch()` accepts
`host_resources` (immutable `pipec2.host.fixture.Resource` loans, each keeping its
lender alive), `host_domains` and `host_scalars`.

## Multi-device sessions

A session serves one rank. Its caller names every communication domain an owner
declared (`PL_HOST_DOMAIN`) and states this rank's place in it:
`open_host(..., domains={"tp": Domain(rank, world), "ep": Domain(...)})`. Domain
identity is the name, never inferred from equal ranks or worlds, so one rank can
belong to several domains (TP x EP). Each owner's `query` and `build` see the
domains it declared in `plk_common_v2.domains` (`plk_find_domain(common, id)`).

A channel with a domain is lent by the caller as one symmetric allocation per rank
of that domain (identical size, alignment and origin); pipec exchanges no handles
and owns no communication runtime. Its `Resource` carries `domain`, and the views
the channels over it declare: `peers` (one base per domain rank, `peers[rank] ==
address`) and `multicast` (this allocation's base in the domain's multicast VA
space). Local resources keep all three empty. At open the broker:

- requires every declared domain, `0 <= rank < world`, the resource's domain to
  equal the channel's, `len(peers) == world` and every declared view present;
- checks every peer and multicast address (non-null, aligned, no overflow), the
  demanded origin alignment on every view, and alias offsets on every view;
- gives `build` `plk_device_resource_v2.domain_id`, `peer_addresses[world]` for
  peer views (NULL otherwise) and `multicast_address` for multicast views (0
  otherwise), valid for the session's lifetime.

A multicast resource is a subview of backing memory the caller bound at the
multicast granularity; the broker sees only the subview. Bytes written through
one view and read through another (multicast stores read through the local
unicast address) are virtual aliases: the owners place `fence.proxy.alias` on the
synchronization path. An owner waiting in more than one domain relies on every
rank waiting in the same global order; the linker does not prove liveness.

`CudaBackend` never leaves its context current: every method, and each phase of
`open_host`, runs inside `backend.current()`. It borrows a current context only on
the requested device and otherwise retains that device's primary context. Wrap
raw driver calls beside a backend in `with backend.current():`. One thread can
therefore drive one backend per device:

```python
from contextlib import ExitStack
from pipec2.host.fixture import Domain, symmetric

with ExitStack() as stack:
    backends = [stack.enter_context(CudaBackend(cubin, symbol, device=d, ...)) for d in range(world)]
    CudaBackend.enable_peer_access(backends)
    fixtures = [stack.enter_context(Fixture(b)) for b in backends]
    shards = symmetric(fixtures, "shard", domain="tp", data=per_rank_bytes, allocation_bytes=n, ...)
    sessions = [stack.enter_context(open_host(result, providers, args,
                    {**fixtures[r].resources, "shard": shards[r]}, scalars, backends[r], cluster,
                    domains={"tp": Domain(r, world)}))
                for r in range(world)]
    for r, session in enumerate(sessions):      # every rank issues every launch
        session.launch(backends[r].stream_token)
    for backend in backends:
        backend.synchronize()
```

`pipec2.host.cuda.MulticastGroup(backends, size)` fabricates in-process multicast
memory for such a group (a VMM backing per device bound to one multicast object);
`group.views(offset, nbytes, domain=..., ...)` returns one `Resource` per rank with
both views. Callers using a communication library build the same `Resource`s from
its pointer queries.

`ExitStack` closes every session, draining its rank, before any fixture frees
memory that peers read. Cross-rank readiness of peer inputs and lockstep launches
are the caller's obligations, and peer-signalled counters must not rely on a
per-launch zero fill. [T7](../tutorials/T7-comm-overlap.md) applies this to
AllGather→GEMM and GEMM→ReduceScatter; [multimem-domains](../../examples/multimem-domains/README.md)
combines a multicast domain and a peer domain in one kernel.
