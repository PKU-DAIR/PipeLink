#!/usr/bin/env python3
"""In-process multi-device runner for the comm-overlap compositions.

One process drives one CudaBackend per rank (device), opens one host session per rank,
and issues every launch to every rank in lockstep. Modes:

  check     correctness at TP2/TP3/TP4 on a small shape, both images (CTA read-drain coverage)
  sanitize  racecheck + synccheck on both images of both compositions at TP2
  time      seam-parity linked/one-TU per-step ratio with the all-rank boundary, the large-shape
            oracle check, and descriptive placement against the sequential baselines
  sass      normalized SASS diagnostic (GPU-free)

--artifacts selects a build root containing index.json without replacing out/PROFILE.
RS outputs are rank-local, including distinct saved pipelined-generation outputs;
wrap coverage includes W-1 parity counters and the u64 epoch at TP2/TP3/TP4.
Timing reports retain scored rank-event arrays and host wall durations for both
images and the descriptive sequential baseline. A failed requested clock-lock
health check makes the overall timing result FAIL.
An RS build declaring rs_completion launches its owner-local completion kernel
on the same rank stream after either push image; graphs and timings cover both.
Follow-on timing additionally scores balanced push-only event intervals in a
separate instrumented trial; both step and push linked/control ratios must be
at most 1.03. The uninstrumented end-to-end trial remains the step metric.
"""
from __future__ import annotations

import argparse
import ctypes as C
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEMM = REPO / "examples/hopper-gemm-split"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(GEMM))

RTOL = ATOL = 1e-3
CAP = 512  # ready-counter slots (the comm owner's gmem_counter claim)
LAUNCHES = 6  # synchronized direct launches per image (>= 5)
REPLAYS = 3
FREE = 6  # pipelined generations enqueued without host synchronization or cross-rank joins
# TP3 presets: AG retains its original ready-counter/epoch coverage; RS counters
# now receive W-1 arrivals per parity use. TP2/TP4 also exercise odd contributor
# counts at the u32 generation boundary, where narrowing before shifting is wrong.
WRAP = {"ag": (2**31 - 3, 2**32 - 3), "rs": (2**32 - 3, 2 * (2**32 // 2) - 4)}


def small(composition, world):
    # AG: global M (gathered), per-rank N, K. RS: M, N, per-rank K.
    return (256 * world, 384, 512)


def large(composition, world):
    return (8192, 8192 // world, 4096) if composition == "ag" else (8192, 8192, 4096 // world)


def seed(launch, rank, tensor):
    return (launch * 64 + rank) * 4 + tensor + 1


class Driver:
    """Raw driver calls beside one backend (always under its context)."""

    def __init__(self, backend, helpers_cubin):
        self.backend = backend
        self.module = C.c_void_p()
        self.functions = {}
        with backend.current():
            backend._call("cuModuleLoad", [C.POINTER(C.c_void_p), C.c_char_p], C.byref(self.module),
                          str(helpers_cubin).encode())
            for name in ("comm_fill_bf16", "comm_spin", "comm_reduce"):
                function = C.c_void_p()
                backend._call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                              C.byref(function), self.module, name.encode())
                self.functions[name] = function
            # A side stream for concurrent caller work that contends for this rank's SMs.
            self.side = C.c_void_p()
            backend._call("cuStreamCreate", [C.POINTER(C.c_void_p), C.c_uint], C.byref(self.side), 1)
        self.fork, self.joined = backend.event_create(), backend.event_create()

    def kernel(self, name, args: bytes, grid, block, stream):
        buffer = C.create_string_buffer(args)
        params = (C.c_void_p * 1)(C.cast(buffer, C.c_void_p))
        with self.backend.current():
            self.backend._call("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.c_void_p, C.c_void_p],
                               self.functions[name], grid, 1, 1, block, 1, 1, 0, stream, params, None)

    def fill(self, address, rows, cols, ld, seed_value, stream):
        self.kernel("comm_fill_bf16", struct.pack("<QQQQII", address, rows, cols, ld, seed_value, 0),
                    min(4096, (rows * cols + 255) // 256), 256, stream)

    def spin(self, nanoseconds, stream, blocks=1, threads=32):
        self.kernel("comm_spin", struct.pack("<Q", nanoseconds), blocks, threads, stream)

    def contend(self, nanoseconds, stream):
        """Occupy all but 16 SMs from a side stream, concurrently with `stream`'s next work."""
        self.backend.event_record(self.fork, stream)
        self.wait(self.fork, self.side.value)
        self.spin(nanoseconds, self.side.value, self.backend.observations["observed_sm_count"] - 16, 1024)
        self.backend.event_record(self.joined, self.side.value)

    def join(self, stream):
        self.wait(self.joined, stream)

    def reduce(self, sources, out, vectors, stream):
        padded = list(sources) + [0] * (8 - len(sources))
        self.kernel("comm_reduce", struct.pack("<8QQQII", *padded, out, vectors, len(sources), 0),
                    min(4096, (vectors + 255) // 256), 256, stream)

    def poison(self, address, words, stream):  # f32 quiet NaN
        with self.backend.current():
            self.backend._call("cuMemsetD32Async", [C.c_uint64, C.c_uint, C.c_size_t, C.c_void_p],
                               address, 0x7FC00000, words, stream)

    def copy(self, destination, source, size, stream):
        with self.backend.current():
            self.backend._call("cuMemcpyDtoDAsync_v2", [C.c_uint64, C.c_uint64, C.c_size_t, C.c_void_p],
                               destination, source, size, stream)

    def wait(self, event, stream):
        with self.backend.current():
            self.backend._call("cuStreamWaitEvent", [C.c_void_p, C.c_void_p, C.c_uint], stream, event, 0)

    def close(self):
        self.backend.synchronize()
        for event in (self.fork, self.joined):
            self.backend.event_destroy(event)
        with self.backend.current():
            self.backend._call("cuStreamDestroy_v2", [C.c_void_p], self.side)
            self.backend._call("cuModuleUnload", [C.c_void_p], self.module)


class Adapter:
    """oracle.Cublas expects .addresses and .synchronize()."""

    def __init__(self, backend, addresses):
        self.addresses = addresses
        self.synchronize = backend.synchronize


class World:
    """W ranks of one composition: backends, fixtures, sessions and both images."""

    def __init__(self, stack, out, index, composition, world, shape, *, control=True):
        from pipec2.host.broker import open_host
        from pipec2.host.cuda import CudaBackend
        from pipec2.host.fixture import Domain, Fixture, symmetric
        from pipec2.host.profile import DTYPES
        from pipec2.model import LinkResult

        self.composition, self.world, self.shape = composition, world, shape
        entry = index["compositions"][composition]
        self.result = LinkResult.from_json_obj(json.loads((out / entry["link"]).read_text()))
        cluster = tuple(self.result.cluster)
        geometry = dict(dynamic_smem_bytes=self.result.dyn_smem_bytes, cluster=cluster,
                        block=(self.result.threads, 1, 1))
        symbol = self.result.kernel_symbol
        self.backends = [stack.enter_context(CudaBackend(out / entry["linked"], symbol, device=r, **geometry))
                         for r in range(world)]
        CudaBackend.enable_peer_access(self.backends)
        self.controls = [stack.enter_context(CudaBackend(out / entry["one_tu"], symbol, device=r, **geometry))
                         for r in range(world)] if control else []
        self.drivers = [Driver(b, out / "helpers.cubin") for b in self.backends]
        stack.callback(lambda: [d.close() for d in self.drivers])
        fixtures = [stack.enter_context(Fixture(b)) for b in self.backends]
        self.streams = [b.stream_token for b in self.backends]
        bf16, u32, u64, f32 = (DTYPES[name]["id"] for name in ("bf16", "u32", "u64", "f32"))
        M, N, K = shape
        W = world
        self.completion_backends = []
        self.completion_geometries = []
        self.completion_params = {}
        self.completion_geometry = []
        completion = entry.get("rs_completion") if composition == "rs" else None
        self.rs_route = entry.get("rs_route", "in-grid") if composition == "rs" else None
        if completion is not None:
            threads = completion["threads"]
            owner_tiles = (M // W // 128) * (N // 128)
            for r, backend in enumerate(self.backends):
                with backend.current():
                    reducer = stack.enter_context(CudaBackend(
                        out / completion["cubin"], completion["symbol"], device=r,
                        dynamic_smem_bytes=0, cluster=(1, 1, 1), block=(threads, 1, 1)))
                observed = reducer.observations
                # Unit clusters: this is the driver's device-wide simultaneous
                # CTA capacity, not just SM count. Use the fully resident grid.
                blocks = min(owner_tiles, observed["max_active_clusters"])
                if blocks < 1:
                    raise RuntimeError("RS completion has no resident launch capacity")
                dims = lambda x: SimpleNamespace(x=x, y=1, z=1)
                self.completion_backends.append(reducer)
                self.completion_geometries.append(SimpleNamespace(
                    grid=dims(blocks), block=dims(threads), cluster=dims(1), dynamic_smem_bytes=0))
                self.completion_geometry.append({
                    "rank": r, "grid": [blocks, 1, 1], "block": [threads, 1, 1],
                    "cluster": [1, 1, 1], "dynamic_smem_bytes": 0, "owner_tiles": owner_tiles,
                    "observed_sm_count": observed["observed_sm_count"],
                    "max_active_clusters": observed["max_active_clusters"]})

        def local(r, name, size, extents, strides, dtype, data=None):
            return fixtures[r].tensor(name, data=data, allocation_bytes=size, extents=extents,
                                      bit_strides=strides, dtype_id=dtype, alignment=256)

        def zeros_u32(name, count):
            return symmetric(fixtures, name, domain="tp", data=[bytes(4 * count)] * W, allocation_bytes=4 * count,
                             extents=(count,), bit_strides=(32,), dtype_id=u32, alignment=256)

        resources = [{} for _ in range(W)]
        blobs = [{} for _ in range(W)]
        if composition == "ag":
            gather = symmetric(fixtures, "gather", domain="tp", data=[None] * W, allocation_bytes=M * K * 2, extents=(M, K),
                               bit_strides=(K * 16, 16), dtype_id=bf16, alignment=256)
            ready, released = zeros_u32("ready", CAP), zeros_u32("released", W)
            for r in range(W):
                resources[r] = {"gather": gather[r], "ready": ready[r], "released": released[r],
                                "gen": local(r, "gen", 4, (1,), (32,), u32, bytes(4)),
                                "b": local(r, "b", N * K * 2, (N, K), (K * 16, 16), bf16)}
            self.a = [g.address for g in gather]
            self.out_rows = M
            self.state = {"ready": ready, "released": released, "gen": [res["gen"] for res in resources]}
        elif composition == "rs":
            slots = symmetric(fixtures, "slots", domain="tp", data=[None] * W, allocation_bytes=2 * M * N * 4,
                              extents=(2 * M * N,), bit_strides=(32,), dtype_id=f32, alignment=256)
            counters = zeros_u32("counters", 2 * (M // W // 128) * (N // 128))
            released = zeros_u32("released", W)
            for r in range(W):
                resources[r] = {"slots": slots[r], "counters": counters[r],
                                "released": released[r], "gen": local(r, "gen", 8, (1,), (64,), u64, bytes(8)),
                                "a": local(r, "a", M * K * 2, (M, K), (K * 16, 16), bf16),
                                "b": local(r, "b", N * K * 2, (N, K), (K * 16, 16), bf16)}
            self.a = [res["a"].address for res in resources]
            self.out_rows = M // W
            self.state = {"counters": counters, "released": released, "gen": [res["gen"] for res in resources]}
        else:  # base: one independent hopper-gemm-split GEMM per rank
            for r in range(W):
                resources[r] = {"a": local(r, "a", M * K * 2, (M, K), (K * 16, 16), bf16),
                                "b": local(r, "b", N * K * 2, (N, K), (K * 16, 16), bf16)}
            self.a = [res["a"].address for res in resources]
            self.out_rows = M
        self.b = [res["b"].address for res in resources]
        providers = {group: (out / path).resolve() for group, path in entry["providers"].items()}
        scalars = {"M": M, "N": N, "K": K}

        def sessions(tag):
            """One session per rank with its own output storage; all other resources shared."""
            if composition == "rs":
                outputs = [local(r, "output" + tag, M // W * N * 4, (M // W, N),
                                 (N * 32, 32), f32) for r in range(W)]
                own = [{"output": resource} for resource in outputs]
                extra = [{} for _ in range(W)]
            else:
                outputs = [local(r, "output" + tag, M * N * 4, (M, N), (N * 32, 32), f32) for r in range(W)]
                own = [{} for _ in range(W)]
                extra = [{"gemm.epilogue.store": struct.pack("<QQQii", o.address, 0, 0, M, N)} for o in outputs]
            opened = [stack.enter_context(open_host(self.result, providers, {}, {**resources[r], **own[r]}, scalars,
                                                    self.backends[r], cluster, blobs={**blobs[r], **extra[r]},
                                                    domains={"tp": Domain(r, W)}))
                      for r in range(W)]
            if completion is not None:
                for session in opened:
                    residue = next(owner.group["residue"] for owner in session.owners
                                   if owner.group["params_group"] == "comm.owner.rs")
                    self.completion_params[id(session)] = SimpleNamespace(
                        address=session.params.address + residue["offset"], size=residue["bytes"])
            return [o.address for o in outputs], opened

        self._sessions = sessions
        self.output, self.sessions = sessions("")
        self.generations = []  # per-generation (outputs, sessions), opened on demand
        self.geometry = self.sessions[0].geometry
        self.fixtures = fixtures
        self.events = [b.event_create() for b in self.backends]
        stack.callback(lambda: [b.event_destroy(e) for b, e in zip(self.backends, self.events)])

    def open_generations(self, count):
        while len(self.generations) < count:
            self.generations.append(self._sessions(f"-g{len(self.generations)}"))
        return self.generations[:count]

    def preset(self, generation):
        """Protocol state as if `generation` launches had completed on every rank (wrap tests)."""
        M, N, K = self.shape
        W = self.world
        low = generation % (1 << 32)
        words = lambda count, value: struct.pack(f"<{count}I", *([value % (1 << 32)] * count))
        self.synchronize()
        for r, backend in enumerate(self.backends):
            gen = self.state["gen"][r]
            backend.upload(gen.address, struct.pack("<Q" if gen.allocation_bytes == 8 else "<I",
                                                    generation if gen.allocation_bytes == 8 else low))
            backend.upload(self.state["released"][r].address, words(W, low))
            if self.composition == "ag":
                backend.upload(self.state["ready"][r].address, words(CAP, generation * (K // 256)))
            else:
                tiles = self.state["counters"][r].allocation_bytes // 8
                # Only W-1 remote contributors increment each parity's counters.
                backend.upload(self.state["counters"][r].address,
                               words(tiles, (generation + 1) // 2 * (W - 1))
                               + words(tiles, generation // 2 * (W - 1)))

    # -- per-rank stream work ------------------------------------------------------------
    def launch(self, r, image="linked", sessions=None, *, push_events=None):
        session = (sessions or self.sessions)[r]
        if image == "linked" and push_events is None:
            session.launch(self.streams[r])
        else:
            for address, size in session.resets:
                self.backends[r].zero(address, size, self.streams[r])
            if push_events is not None:
                self.backends[r].event_record(push_events[0], self.streams[r])
            backend = self.backends[r] if image == "linked" else self.controls[r]
            backend.launch(session.params, session._geometry, self.streams[r])
            if push_events is not None:
                self.backends[r].event_record(push_events[1], self.streams[r])
        if self.completion_backends:
            self.completion_backends[r].launch(self.completion_params[id(session)],
                                               self.completion_geometries[r], self.streams[r])

    def fill(self, r, launch):
        M, N, K = self.shape
        rows = M // self.world if self.composition == "ag" else M
        offset = r * rows * K * 2 if self.composition == "ag" else 0
        self.drivers[r].fill(self.a[r] + offset, rows, K, K, seed(launch, r, 0), self.streams[r])
        self.drivers[r].fill(self.b[r], N, K, K, seed(launch, r, 1), self.streams[r])

    def poison(self, r, address=None):
        self.drivers[r].poison(self.output[r] if address is None else address, self.out_rows * self.shape[1],
                               self.streams[r])

    def synchronize(self):
        for backend in self.backends:
            backend.synchronize()

    def read(self, r, address=None):
        import numpy as np
        size = self.out_rows * self.shape[1] * 4
        data = self.backends[r].readback(self.output[r] if address is None else address, size)
        return np.frombuffer(data, dtype=np.float32).reshape(self.out_rows, self.shape[1])

    def outputs(self):
        return [self.read(r) for r in range(self.world)]


class Oracle:
    """cuBLAS of the gathered (AG) or K-concatenated (RS) problem from independently filled inputs."""

    def __init__(self, stack, world: World, cuda_home):
        from oracle import Cublas

        self.world = world
        M, N, K = world.shape
        W = world.world
        self.cache = {}
        self.blas = []
        if world.composition == "ag":
            # One cuBLAS per rank: A gathered from every rank's seeds, B from that rank's seed.
            for r, backend in enumerate(world.backends):
                a = backend.allocate(M * K * 2, 256)
                b = backend.allocate(N * K * 2, 256)
                c = backend.allocate(M * N * 4, 256)
                for address in (c, b, a):
                    stack.callback(backend.free, address)
                with backend.current():
                    blas = stack.enter_context(Cublas(Adapter(backend, {"a": a, "b": b, "oracle": c}),
                                                      (M, N, K), cuda_home))
                self.blas.append((backend, blas, a, c, b))
        else:
            backend = world.backends[0]
            a = backend.allocate(M * W * K * 2, 256)
            b = backend.allocate(N * W * K * 2, 256)
            c = backend.allocate(M * N * 4, 256)
            for address in (c, b, a):
                stack.callback(backend.free, address)
            with backend.current():
                blas = stack.enter_context(Cublas(Adapter(backend, {"a": a, "b": b, "oracle": c}),
                                                  (M, N, W * K), cuda_home))
            self.blas.append((backend, blas, a, c, b))
            self.rs_b = b

    def expected(self, launch):
        import numpy as np
        if launch in self.cache:
            return self.cache[launch]
        world = self.world
        M, N, K = world.shape
        W = world.world
        result = []
        if world.composition == "ag":
            rows = M // W
            for r, (backend, blas, a, c, b) in enumerate(self.blas):
                driver = world.drivers[r]
                for s in range(W):
                    driver.fill(a + s * rows * K * 2, rows, K, K, seed(launch, s, 0), 0)
                driver.fill(b, N, K, K, seed(launch, r, 1), 0)
                with backend.current():
                    backend.synchronize()
                    blas.launch()
                    backend.synchronize()
                result.append(np.frombuffer(backend.readback(c, M * N * 4), dtype=np.float32).reshape(M, N))
        else:
            backend, blas, a, c, _ = self.blas[0]
            driver = world.drivers[0]
            for s in range(W):
                driver.fill(a + s * K * 2, M, K, W * K, seed(launch, s, 0), 0)
                driver.fill(self.rs_b + s * K * 2, N, K, W * K, seed(launch, s, 1), 0)
            with backend.current():
                backend.synchronize()
                blas.launch()
                backend.synchronize()
            full = np.frombuffer(backend.readback(c, M * N * 4), dtype=np.float32).reshape(M, N)
            result = [full[r * M // W:(r + 1) * M // W].copy() for r in range(W)]
        self.cache[launch] = result
        return result


def compare_all(actual, expected, label):
    from evaluator.harness.parity.example import compare
    ranks = [compare(a, e, rtol=RTOL, atol=ATOL) for a, e in zip(actual, expected, strict=True)]
    return {"label": label, "ranks": ranks, "status": "PASS" if all(r["allclose"] for r in ranks) else "FAIL"}


def bitwise(left, right):
    return all(a.tobytes() == b.tobytes() for a, b in zip(left, right, strict=True))


def coverage(world: World, oracle: Oracle, image: str, launches=LAUNCHES, replays=REPLAYS, free=FREE):
    """CTA read-drain coverage for one image. Every checked output is compared with the oracle."""
    W = world.world
    record = {"image": image, "direct": [], "graph": []}
    outputs = {}

    def checked(label, launch, actual):
        result = compare_all(actual, oracle.expected(launch), label)
        outputs[label] = actual
        return result

    # 1. Synchronized lockstep launches, changing payloads, a skewed rank on some launches.
    for e in range(launches):
        skew = e % 2 == 1
        for r in range(W):
            world.fill(r, e)
            world.poison(r)
            if skew and r == e % W:
                world.drivers[r].spin(3_000_000, world.streams[r])  # 3 ms late start
        for r in range(W):
            world.launch(r, image)
        world.synchronize()
        result = checked(f"direct-{e}", e, world.outputs())
        result["skewed_rank"] = e % W if skew else None
        record["direct"].append(result)

    # 2. Per-rank graphs replayed in lockstep (payload of launch 0, which a direct launch saw).
    graphs = []
    for r in range(W):
        world.fill(r, 0)
    world.synchronize()
    for r in range(W):
        graphs.append(world.backends[r].capture(world.streams[r], lambda r=r: world.launch(r, image)))
    for i in range(replays):
        for r in range(W):
            world.poison(r)
        for r in range(W):
            graphs[r].launch(world.streams[r])
        world.synchronize()
        result = checked(f"graph-{i}", 0, world.outputs())
        result["bitwise_vs_direct_0"] = bitwise(outputs[f"graph-{i}"], outputs["direct-0"])
        record["graph"].append(result)

    # 3. Pipelined generations: `free` changing-payload launches per rank enqueued with no host
    #    synchronization and no cross-rank joins on the launch streams. Generations 1 and 3 start
    #    one rank late; during generations 2 and 4 one rank's SMs are mostly occupied by concurrent
    #    caller work (a side stream), so its peers run ahead into the next generation mid-launch.
    #    Every generation has its own session and output storage (shared counters, slots,
    #    released flags and generation), so each output is kept and checked after the batch.
    #    The same sequence is then captured as one graph per rank and replayed in lockstep.
    generations = world.open_generations(free)

    def enqueue(r, first):
        contended = False
        for e, (_, sessions) in enumerate(generations):
            world.fill(r, first + e)
            if e in (1, 3) and r == e % W:
                world.drivers[r].spin(3_000_000, world.streams[r])  # 3 ms late start
            if e in (2, 4) and r == (e // 2) % W:
                world.drivers[r].contend(2_000_000, world.streams[r])  # slowed for ~2 ms
                contended = True
            world.launch(r, image, sessions)
        if contended:  # rejoin the side stream (required to end a graph capture)
            world.drivers[r].join(world.streams[r])

    def poison_all():
        for outputs, _ in generations:
            for r in range(W):
                world.poison(r, outputs[r])
        world.synchronize()

    def check_batch(prefix, first):
        world.synchronize()
        return [checked(f"{prefix}-{e}", first + e, [world.read(r, outputs[r]) for r in range(W)])
                for e, (outputs, _) in enumerate(generations)]

    poison_all()
    for r in range(W):
        enqueue(r, 100)
    record["pipelined"] = check_batch("pipelined", 100)
    graphs = [world.backends[r].capture(world.streams[r], lambda r=r: enqueue(r, 200)) for r in range(W)]
    record["pipelined_graph"] = []
    for i in range(replays):
        poison_all()
        for r in range(W):
            graphs[r].launch(world.streams[r])
        batch = check_batch(f"pipelined-graph{i}", 200)
        for e, result in enumerate(batch):
            result["bitwise_vs_replay_0"] = bitwise(outputs[f"pipelined-graph{i}-{e}"], outputs[f"pipelined-graph0-{e}"])
        record["pipelined_graph"].extend(batch)
    for graph in graphs:
        graph.close()

    checks = [*record["direct"], *record["graph"], *record["pipelined"], *record["pipelined_graph"]]
    passed = all(c["status"] == "PASS" for c in checks) and all(
        c["bitwise_vs_direct_0"] for c in record["graph"]) and all(
        c["bitwise_vs_replay_0"] for c in record["pipelined_graph"])
    record["status"] = "PASS" if passed else "FAIL"
    return record, outputs


def check_world(out, index, composition, world_size, shape, cuda_home, images=("linked", "one-tu"),
                launches=LAUNCHES, replays=REPLAYS, free=FREE, preset=None):
    with ExitStack() as stack:
        world = World(stack, out, index, composition, world_size, shape, control="one-tu" in images)
        oracle = Oracle(stack, world, cuda_home)
        if preset is not None:
            world.preset(preset)
        report = {"composition": composition, "world": world_size, "shape": list(shape),
                  "initial_generation": preset or 0, "geometry": world.geometry, "images": {}}
        if composition == "rs":
            report.update(rs_route=world.rs_route, completion_geometry=world.completion_geometry)
        per_image = {}
        for image in images:
            report["images"][image], per_image[image] = coverage(world, oracle, image, launches, replays, free)
        passed = all(r["status"] == "PASS" for r in report["images"].values())
        if len(images) == 2:
            same = {label: bitwise(per_image["linked"][label], per_image["one-tu"][label])
                    for label in per_image["linked"]}
            report["linked_vs_one_tu_bitwise"] = all(same.values())
            report["linked_vs_one_tu_mismatched_labels"] = [k for k, v in same.items() if not v]
            passed &= report["linked_vs_one_tu_bitwise"]
        report["status"] = "PASS" if passed else "FAIL"
        return report


# -- timing ---------------------------------------------------------------------------------
class AllRank:
    """time(n): start events on every rank, n lockstep steps issued asynchronously to every
    rank, end events, then synchronize all ranks. Step time = max_r(elapsed_r) / n."""

    def __init__(self, backends, issue):
        self.backends, self.issue = backends, issue
        self.start = [b.event_create() for b in backends]
        self.end = [b.event_create() for b in backends]
        self.records = []

    def launch(self):
        self.issue()

    def time(self, n):
        wall = time.perf_counter()
        for b, e in zip(self.backends, self.start):
            b.event_record(e, b.stream_token)
        for _ in range(n):
            self.issue()
        for b, e in zip(self.backends, self.end):
            b.event_record(e, b.stream_token)
        for b in self.backends:
            b.synchronize()
        wall = (time.perf_counter() - wall) * 1e3
        elapsed = [b.event_elapsed_time(s, e) for b, s, e in zip(self.backends, self.start, self.end)]
        self.records.append({"n": n, "per_rank_ms": elapsed, "wall_ms": wall})
        return max(elapsed)

    def close(self):
        for b, s, e in zip(self.backends, self.start, self.end):
            b.event_destroy(s)
            b.event_destroy(e)


class PushPhase(AllRank):
    """Separate instrumented trial; score max_r(sum of that rank's push intervals).

    Resets and the completion kernel are outside each interval. Keep whole-step
    event totals and wall time too; ordinary seam-parity trials remain uninstrumented.
    """

    def __init__(self, world, image):
        self.world, self.image = world, image
        self.pairs = [[] for _ in world.backends]
        self.position = 0
        super().__init__(world.backends, self.issue_step)

    def issue_step(self):
        for r in range(self.world.world):
            self.world.launch(r, self.image, push_events=self.pairs[r][self.position])
        self.position += 1

    def time(self, n):
        # Allocate/reuse event pools before the timed batch, never in World.launch.
        for backend, pairs in zip(self.backends, self.pairs):
            while len(pairs) < n:
                pairs.append((backend.event_create(), backend.event_create()))
        self.position = 0
        super().time(n)  # all ranks issue every step before any host synchronization
        intervals = [[backend.event_elapsed_time(start, end) for start, end in pairs[:n]]
                     for backend, pairs in zip(self.backends, self.pairs)]
        record = self.records[-1]
        record["step_per_rank_ms"] = record["per_rank_ms"]
        record["push_per_rank_step_ms"] = intervals
        record["per_rank_ms"] = [sum(rank) for rank in intervals]
        return max(record["per_rank_ms"])

    def close(self):
        for backend, pairs in zip(self.backends, self.pairs):
            for start, end in pairs:
                backend.event_destroy(start)
                backend.event_destroy(end)
        super().close()


def summarize(adapter, scored_batches=16):
    records = adapter.records[-scored_batches:]  # exclude warmup and calibration
    return {"scored_batches": records,
            "mean_step_ms": sum(max(r["per_rank_ms"]) / r["n"] for r in records) / len(records),
            "mean_wall_step_ms": sum(r["wall_ms"] / r["n"] for r in records) / len(records),
            "max_rank_over_min_rank_ms": max(max(r["per_rank_ms"]) / min(r["per_rank_ms"]) for r in records),
            "wall_over_event_max": [r["wall_ms"] / max(r["per_rank_ms"]) for r in records]}


def clocks(gpus):
    completed = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.max.sm", "--format=csv,noheader",
                                "-i", ",".join(map(str, gpus))], capture_output=True, text=True)
    return completed.stdout.strip().splitlines()


def time_world(out, index, composition, world_size, cuda_home, gpus):
    from evaluator.harness.parity.example import time_pair
    shape = large(composition, world_size)
    M, N, K = shape
    W = world_size
    with ExitStack() as stack:
        world = World(stack, out, index, composition, W, shape)
        oracle = Oracle(stack, world, cuda_home)
        base = World(stack, out, index, "base", W, (M, N, K), control=False)
        report = {"composition": composition, "world": W, "shape": list(shape), "geometry": world.geometry,
                  "flops_per_step": 2 * M * N * K * W, "clocks_before": clocks(gpus)}
        if composition == "rs":
            report.update(rs_route=world.rs_route, completion_geometry=world.completion_geometry)
        # Correctness at the timing shape (both images) before and after timing.
        for r in range(W):
            world.fill(r, 7)
        results = {}
        for image in ("linked", "one-tu"):
            for r in range(W):
                world.poison(r)
            for r in range(W):
                world.launch(r, image)
            world.synchronize()
            results[image] = world.outputs()
        report["oracle"] = compare_all(results["linked"], oracle.expected(7), "large-linked")
        report["linked_vs_one_tu_bitwise"] = bitwise(results["linked"], results["one-tu"])

        linked = AllRank(world.backends, lambda: [world.launch(r, "linked") for r in range(W)])
        control = AllRank(world.backends, lambda: [world.launch(r, "one-tu") for r in range(W)])
        report["o2"] = time_pair(linked, control, repetitions=20)
        scored_batches = 2 * len(report["o2"]["blocks"])
        report["o2"]["all_rank"] = {"linked": summarize(linked, scored_batches),
                                    "one_tu": summarize(control, scored_batches)}
        report["o2"]["linked_mean_step_ms"] = report["o2"]["all_rank"]["linked"]["mean_step_ms"]
        report["o2"]["control_mean_step_ms"] = report["o2"]["all_rank"]["one_tu"]["mean_step_ms"]
        push_passed = True
        if world.completion_backends:
            push_linked = PushPhase(world, "linked")
            stack.callback(push_linked.close)
            push_control = PushPhase(world, "one-tu")
            stack.callback(push_control.close)
            phase = report["o2_push"] = time_pair(push_linked, push_control, repetitions=20)
            scored_batches = 2 * len(phase["blocks"])
            phase["measurement"] = "max_rank(sum(push intervals)); resets/completion excluded"
            phase["all_rank"] = {"linked": summarize(push_linked, scored_batches),
                                 "one_tu": summarize(push_control, scored_batches)}
            phase["linked_mean_push_ms"] = phase["all_rank"]["linked"]["mean_step_ms"]
            phase["control_mean_push_ms"] = phase["all_rank"]["one_tu"]["mean_step_ms"]
            push_passed = phase["status"] == "PASS" and phase["ratio_linked_over_onetu"] <= 1.03
        world.synchronize()
        after = world.outputs()
        report["after_timing_bitwise"] = bitwise(after, results["linked"])

        # Sequential baselines: the unchanged hopper-gemm-split GEMM (base link) per rank.
        if composition == "ag":
            rows = M // W
            # Each rank's base A buffer holds its own shard; peers' shards are pulled per step.
            for r in range(W):
                base.drivers[r].fill(base.a[r] + r * rows * K * 2, rows, K, K, seed(7, r, 0), base.streams[r])
                base.drivers[r].fill(base.b[r], N, K, K, seed(7, r, 1), base.streams[r])

            def baseline():
                for r in range(W):
                    for p in range(W):
                        if p != r:
                            offset = p * rows * K * 2
                            base.drivers[r].copy(base.a[r] + offset, base.a[p] + offset, rows * K * 2,
                                                 base.streams[r])
                    base.launch(r)
            expected = oracle.expected(7)
        else:
            rows = M // W
            shards = [b.allocate(rows * N * 4, 256) for b in base.backends]
            stack.callback(lambda: [b.free(s) for b, s in zip(base.backends, shards)])
            for r in range(W):
                base.drivers[r].fill(base.a[r], M, K, K, seed(7, r, 0), base.streams[r])
                base.drivers[r].fill(base.b[r], N, K, K, seed(7, r, 1), base.streams[r])
            reduced = [b.event_create() for b in base.backends]
            stack.callback(lambda: [b.event_destroy(e) for b, e in zip(base.backends, reduced)])

            def baseline():
                for r in range(W):
                    base.launch(r)
                    base.backends[r].event_record(base.events[r], base.streams[r])
                for o in range(W):
                    for q in range(W):
                        base.drivers[o].wait(base.events[q], base.streams[o])
                    base.drivers[o].reduce([base.output[q] + o * rows * N * 4 for q in range(W)], shards[o],
                                           rows * N // 4, base.streams[o])
                    base.backends[o].event_record(reduced[o], base.streams[o])
                # WAR: a rank's next GEMM must not overwrite partials an owner is still reducing.
                for q in range(W):
                    for o in range(W):
                        base.drivers[q].wait(reduced[o], base.streams[q])
            expected = oracle.expected(7)
        base.synchronize()
        baseline()
        base.synchronize()
        if composition == "ag":
            actual = base.outputs()
        else:
            import numpy as np
            actual = [np.frombuffer(base.backends[r].readback(shards[r], rows * N * 4), dtype=np.float32)
                      .reshape(rows, N) for r in range(W)]
        report["baseline_oracle"] = compare_all(actual, expected, "baseline")
        sequential = AllRank(base.backends, baseline)
        linked_again = AllRank(world.backends, lambda: [world.launch(r, "linked") for r in range(W)])
        report["placement"] = time_pair(linked_again, sequential, repetitions=20)
        scored_batches = 2 * len(report["placement"]["blocks"])
        report["placement"]["all_rank"] = {"linked": summarize(linked_again, scored_batches),
                                           "sequential": summarize(sequential, scored_batches)}
        report["placement"]["linked_mean_step_ms"] = report["placement"]["all_rank"]["linked"]["mean_step_ms"]
        report["placement"]["baseline_mean_step_ms"] = report["placement"]["all_rank"]["sequential"]["mean_step_ms"]
        report["placement"]["descriptive_only"] = True
        report["placement"]["baseline"] = ("cuMemcpyDtoDAsync peer pulls of every remote shard, then the base GEMM"
                                           if composition == "ag" else
                                           "base GEMM to a full f32 partial per rank, cross-rank event join, "
                                           "then a separate rank-order reduce kernel per owner")
        report["clocks_after"] = clocks(gpus)
        report["linked_step_ms_estimate"] = report["o2"]["linked_mean_step_ms"]
        for adapter in (linked, control, sequential, linked_again):
            adapter.close()
        report["status"] = "PASS" if (report["oracle"]["status"] == "PASS" and report["linked_vs_one_tu_bitwise"]
                                      and report["after_timing_bitwise"] and report["o2"]["status"] == "PASS"
                                      and report["o2"]["ratio_linked_over_onetu"] <= 1.03
                                      and push_passed) else "FAIL"
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("check", "sanitize", "time", "sass", "single"))
    parser.add_argument("--profile", choices=("sm_90a",), required=True)
    parser.add_argument("--gpus", default=None, help="physical GPUs, comma-separated (default PIPELINK_GPUS)")
    parser.add_argument("--tp", type=int, nargs="+", default=None)
    parser.add_argument("--composition", choices=("ag", "rs"), nargs="+", default=["ag", "rs"])
    parser.add_argument("--image", choices=("linked", "one-tu"), default="linked", help="single mode only")
    parser.add_argument("--lock-mhz", type=int, default=None, help="time: hold a gpulockd lease at N MHz")
    parser.add_argument("--artifacts", type=Path, default=None,
                        help="artifact root containing index.json (default out/PROFILE)")
    args = parser.parse_args()
    from tools.pipelink_env import load

    config = load(REPO)
    gpus = [int(g) for g in (args.gpus or config["PIPELINK_GPUS"]).split(",")]
    os.environ.update(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    scratch = Path(config["PIPELINK_SCRATCH_ROOT"]) / "comm-overlap" / "run"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["CUDA_CACHE_PATH"] = str(scratch / "cuda-cache")
    out = args.artifacts.resolve() if args.artifacts is not None else HERE / "out" / args.profile
    index = json.loads((out / "index.json").read_text())
    cuda_home = config["PIPELINK_CUDA_HOME"]

    if args.mode == "sass":
        from evaluator.harness.parity.example import sass_diagnostic
        report = {}
        selected = [name for name in args.composition if name in index["compositions"]]
        if not selected:
            parser.error("no selected composition is present in this artifact index")
        for name in selected:
            entry = index["compositions"][name]
            result = sass_diagnostic(out / entry["linked"], out / entry["one_tu"], env=dict(os.environ),
                                     out_dir=out / "sass" / name, backend_matched=out / entry["one_tu_lto"])
            result["comparison"] = str(Path(result["comparison"]).relative_to(out))
            report[name] = result
        (out / "results-sass.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0

    if args.mode == "single":  # sanitizer subprocess: one composition, one image, small shape
        world = args.tp[0]
        report = check_world(out, index, args.composition[0], world, small(args.composition[0], world),
                             cuda_home, images=(args.image,), launches=3, replays=2, free=3)
        print(json.dumps({"status": report["status"]}))
        return 0 if report["status"] == "PASS" else 1

    if args.mode == "sanitize":
        from evaluator.harness.parity.example import sanitize
        world = (args.tp or [2])[0]
        report = {"world": world, "gpus": gpus[:world], "legs": []}
        for composition in args.composition:
            for image in ("linked", "one-tu"):
                records = sanitize([sys.executable, str(HERE / "run.py"), "single", "--profile", args.profile,
                                    "--artifacts", str(out), "--gpus", ",".join(map(str, gpus[:world])),
                                    "--tp", str(world),
                                    "--composition", composition, "--image", image],
                                   env=dict(os.environ), log_dir=out / "sanitize" / f"tp{world}" / composition / image)
                for record in records:
                    record["log"] = str(Path(record["log"]).relative_to(out))
                    report["legs"].append({"composition": composition, "image": image, **record})
        report["status"] = "PASS" if all(leg["status"] == "PASS" for leg in report["legs"]) else "FAIL"
        name = "results-sanitize.json" if world == 2 else f"results-sanitize-tp{world}.json"
        (out / name).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1

    if args.mode == "check":
        report = {"gpus": gpus, "rtol": RTOL, "atol": ATOL, "cells": []}
        for composition in args.composition:
            for world in args.tp or (2, 3, 4):
                if composition == "rs":
                    presets = (None, 2**32 - 3, 2 * (2**32 // (world - 1)) - 4)
                else:
                    presets = (None, *WRAP[composition]) if world == 3 else (None,)
                for preset in presets:
                    cell = check_world(out, index, composition, world, small(composition, world), cuda_home,
                                       preset=preset)
                    print(json.dumps({k: cell[k] for k in ("composition", "world", "initial_generation", "status")}),
                          flush=True)
                    report["cells"].append(cell)
            # The timing shapes: long mainloops let SM contention push ranks a generation apart,
            # which the small shapes cannot (the AG cross-launch release-wait negative control passes at small shapes).
            for world in [w for w in (args.tp or (2, 3, 4)) if w in (2, 4)]:
                cell = check_world(out, index, composition, world, large(composition, world), cuda_home,
                                   launches=2, replays=2)
                print(json.dumps({k: cell[k] for k in ("composition", "world", "shape", "status")}), flush=True)
                report["cells"].append(cell)
        report["status"] = "PASS" if all(c["status"] == "PASS" for c in report["cells"]) else "FAIL"
        (out / "results-check.json").write_text(json.dumps(report, indent=2) + "\n")
        return 0 if report["status"] == "PASS" else 1

    # time
    lock = []
    if args.lock_mhz:
        tool = [sys.executable, str(REPO / "tools/gpulockd.py")]
        started = subprocess.run([*tool, "start", "--gpus", ",".join(map(str, gpus)), "--sm-mhz",
                                  str(args.lock_mhz)], capture_output=True, text=True)
        lock.append({"start": started.returncode, "stdout": started.stdout, "stderr": started.stderr})
        if started.returncode:
            raise RuntimeError(f"gpulockd start failed: {started.stdout}{started.stderr}")
    try:
        report = {"gpus": gpus, "lock_mhz": args.lock_mhz, "clock_locked": False, "cells": []}
        for composition in args.composition:
            for world in args.tp or (2, 4):
                cell = time_world(out, index, composition, world, cuda_home, gpus[:world])
                print(json.dumps({"composition": composition, "world": world, "status": cell["status"],
                                  "ratio": cell["o2"]["ratio_linked_over_onetu"],
                                  "push_ratio": cell.get("o2_push", {}).get("ratio_linked_over_onetu"),
                                  "placement_linked_over_sequential": cell["placement"]["ratio_linked_over_onetu"]}),
                      flush=True)
                report["cells"].append(cell)
        if args.lock_mhz:
            status = subprocess.run([*tool, "status", "--gpus", ",".join(map(str, gpus))],
                                    capture_output=True, text=True)
            lock.append({"status": status.returncode, "stdout": status.stdout, "stderr": status.stderr})
            report["clock_locked"] = status.returncode == 0
    finally:
        if args.lock_mhz:
            stopped = subprocess.run([*tool, "stop", "--gpus", ",".join(map(str, gpus))],
                                     capture_output=True, text=True)
            lock.append({"stop": stopped.returncode, "stdout": stopped.stdout})
    report["lock"] = lock
    report["status"] = "PASS" if (all(c["status"] == "PASS" for c in report["cells"])
                                  and (not args.lock_mhz or report["clock_locked"])) else "FAIL"
    name = "results-time" + ("" if args.lock_mhz else "-unlocked") + ".json"
    (out / name).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
