#!/usr/bin/env python3
"""Four in-process ranks: bitwise checks, balanced timing, SASS and sanitizers."""
from __future__ import annotations

import argparse
import ctypes as C
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


class Driver:
    def __init__(self, backend, cubin):
        self.backend = backend
        self.module, self.function = C.c_void_p(), C.c_void_p()
        with backend.current():
            backend._call("cuModuleLoad", [C.POINTER(C.c_void_p), C.c_char_p],
                          C.byref(self.module), str(cubin).encode())
            backend._call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                          C.byref(self.function), self.module, b"witness_spin")

    def spin(self, nanoseconds, stream):
        duration = C.c_uint64(nanoseconds)
        args = (C.c_void_p * 1)(C.addressof(duration))
        with self.backend.current():
            self.backend._call("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7 +
                               [C.c_void_p, C.c_void_p, C.c_void_p],
                               self.function, 1, 1, 1, 32, 1, 1, 0, stream, args, None)

    def poison(self, address, words, stream):
        with self.backend.current():
            self.backend._call("cuMemsetD32Async", [C.c_uint64, C.c_uint, C.c_size_t, C.c_void_p],
                               address, 0x7FC00000, words, stream)

    def close(self):
        self.backend.synchronize()
        with self.backend.current():
            self.backend._call("cuModuleUnload", [C.c_void_p], self.module)


def payload(elements, generation, rank):
    import numpy as np
    # Exactly representable signed fractions, distinct for every generation/rank/element.
    x = np.arange(elements, dtype=np.int32)
    return (((x * (rank * 2 + 3) + generation * 97 + rank * 31) % 4093 - 2046)
            .astype(np.float32) * np.float32(0.125))


class World:
    def __init__(self, stack, out, index, *, slices, slice_elements, generations):
        from pipec2.host.broker import open_host
        from pipec2.host.cuda import CudaBackend, MulticastGroup
        from pipec2.host.fixture import Domain, Fixture, symmetric
        from pipec2.host.profile import DTYPES
        from pipec2.model import LinkResult

        self.elements = elements = slices * slice_elements
        self.generations = generations
        result = LinkResult.from_json_obj(json.loads((out / index["link"]).read_text()))
        geometry = dict(dynamic_smem_bytes=result.dyn_smem_bytes, cluster=tuple(result.cluster),
                        block=(result.threads, 1, 1))
        self.backends = [stack.enter_context(CudaBackend(out / index["linked"], result.kernel_symbol,
                                                        device=r, **geometry)) for r in range(4)]
        CudaBackend.enable_peer_access(self.backends)
        self.controls = [stack.enter_context(CudaBackend(out / index["one_tu"], result.kernel_symbol,
                                                        device=r, **geometry)) for r in range(4)]
        self.streams = [b.stream_token for b in self.backends]
        self.drivers = [Driver(b, out / index["helpers"]) for b in self.backends]
        for driver in self.drivers:
            stack.callback(driver.close)
        f32, u32 = (DTYPES[t]["id"] for t in ("f32", "u32"))
        # One bound backing per tp pair: distinct in[g] subviews plus shared red/counters.
        offsets, cursor = {}, 0
        for name, count in [*((f"in{g}", elements) for g in range(generations)),
                            ("red", elements), ("tp_bar", slices), ("tp_done", slices)]:
            offsets[name] = (cursor, count)
            cursor += (count * 4 + 255) // 256 * 256
        groups = [stack.enter_context(MulticastGroup(self.backends[start:start + 2], cursor))
                  for start in (0, 2)]
        rank_stacks = [stack.enter_context(ExitStack()) for _ in range(4)]
        fixtures = [rank_stacks[r].enter_context(Fixture(b)) for r, b in enumerate(self.backends)]
        resources = [{} for _ in range(4)]
        for pair, group in enumerate(groups):
            for name, (offset, count) in offsets.items():
                views = group.views(offset, count * 4, domain="tp", extents=(count,),
                                    bit_strides=(32,), dtype_id=u32 if name.startswith("tp_") else f32,
                                    alignment=256)
                for local, view in enumerate(views):
                    rank = pair * 2 + local
                    resources[rank][name] = view
                    if name.startswith("in"):
                        data = payload(elements, int(name[2:]), rank).tobytes()
                    else:
                        data = bytes(count * 4)
                    self.backends[rank].upload(view.address, data)
        for pair in ((0, 2), (1, 3)):
            for name, count, dtype in [("dp_bar", slices, u32), ("dp_done", slices, u32),
                                        *((f"out{g}", 2 * elements, f32) for g in range(generations))]:
                views = symmetric([fixtures[r] for r in pair], name, domain="dp",
                                  data=[bytes(count * 4)] * 2, allocation_bytes=count * 4,
                                  extents=(count,), bit_strides=(32,), dtype_id=dtype, alignment=256)
                for rank, view in zip(pair, views):
                    resources[rank][name] = view
        for r, fixture in enumerate(fixtures):
            for name in ("mc_epoch", "dp_epoch"):
                resources[r][name] = fixture.tensor(name, data=bytes(4 * slices), allocation_bytes=4 * slices,
                                                    extents=(slices,), bit_strides=(32,), dtype_id=u32,
                                                    alignment=256)
        providers = {name: (out / path).resolve() for name, path in index["providers"].items()}
        self.sessions, self.outputs = [], []
        for g in range(generations):
            sessions, outputs = [], []
            for r in range(4):
                loans = {name: value for name, value in resources[r].items()
                         if not name.startswith(("in", "out"))}
                loans.update({"in": resources[r][f"in{g}"], "out": resources[r][f"out{g}"]})
                session = rank_stacks[r].enter_context(open_host(
                    result, providers, {}, loans, {"elements": elements, "slice_elements": slice_elements},
                    self.backends[r], tuple(result.cluster), domains={"tp": Domain(r % 2, 2),
                                                                     "dp": Domain(r // 2, 2)}))
                sessions.append(session)
                outputs.append(loans["out"].address)
            self.sessions.append(sessions)
            self.outputs.append(outputs)
        self.geometry = self.sessions[0][0].geometry
        self.residency = []
        grids = []
        for r in range(4):
            grid = self.sessions[0][r]._geometry.grid.x
            grids.append(grid)
            for image, backend in (("linked", self.backends[r]), ("one-tu", self.controls[r])):
                observed = backend.observations
                assert grid == min(slices, observed["observed_sm_count"])
                assert observed["max_active_clusters"] >= grid, (r, image, grid, observed)
                self.residency.append({"rank": r, "image": image, "grid": grid,
                                       "sm_count": observed["observed_sm_count"],
                                       "max_active_clusters": observed["max_active_clusters"]})
        assert len(set(grids)) == 1, "all ranks must traverse the same persistent slice schedule"
        self.backing = [{"bytes": group.size, "granularity": group.granularity,
                         "logical_bytes": cursor} for group in groups]
        self.synchronize()

    def launch(self, rank, image, generation=0):
        self.sessions[generation][rank].launch(self.streams[rank],
                                               image=self.controls[rank] if image == "one-tu" else None)

    def poison(self, rank, generation):
        self.drivers[rank].poison(self.outputs[generation][rank], self.elements * 2, self.streams[rank])

    def sequence(self, rank, image, generations, *, skew=False):
        for g in generations:
            self.poison(rank, g)
            if skew and rank == g % 4:
                self.drivers[rank].spin(2_000_000, self.streams[rank])
            self.launch(rank, image, g)

    def synchronize(self):
        for backend in self.backends:
            backend.synchronize()

    def check(self, generation):
        import numpy as np
        expected = np.concatenate([payload(self.elements, generation, r) +
                                   payload(self.elements, generation, r + 1) for r in (0, 2)])
        ranks = []
        for rank, backend in enumerate(self.backends):
            actual = np.frombuffer(backend.readback(self.outputs[generation][rank], self.elements * 8),
                                   dtype=np.float32)
            mismatch = np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))
            ranks.append({"rank": rank, "bitwise": bool(mismatch == 0), "mismatches": int(mismatch)})
        return {"generation": generation, "ranks": ranks,
                "status": "PASS" if all(r["bitwise"] for r in ranks) else "FAIL"}


def check_world(out, index, *, slices, slice_elements, generations, images):
    with ExitStack() as stack:
        world = World(stack, out, index, slices=slices, slice_elements=slice_elements, generations=generations)
        report = {"geometry": world.geometry, "residency": world.residency, "backing": world.backing,
                  "elements": world.elements, "images": {}}
        for image in images:
            results = []
            for g in range(generations):
                for r in range(4):
                    world.sequence(r, image, [g])
                world.synchronize()
                results.append({"mode": "direct", **world.check(g)})
            graphs = [stack.enter_context(b.capture(world.streams[r],
                      lambda r=r: world.sequence(r, image, [0]))) for r, b in enumerate(world.backends)]
            for replay in range(3):
                for r, graph in enumerate(graphs):
                    graph.launch(world.streams[r])
                world.synchronize()
                results.append({"mode": "graph", "replay": replay, **world.check(0)})
            # Rank-major enqueue deliberately permits different ranks to be many host launches apart.
            for r in range(4):
                world.sequence(r, image, range(generations), skew=True)
            world.synchronize()
            results.extend({"mode": "pipelined-skew", **world.check(g)} for g in range(generations))
            pipelines = [stack.enter_context(b.capture(world.streams[r],
                         lambda r=r: world.sequence(r, image, range(generations), skew=True)))
                         for r, b in enumerate(world.backends)]
            for replay in range(3):
                for r, graph in enumerate(pipelines):
                    graph.launch(world.streams[r])
                world.synchronize()
                results.extend({"mode": "graph-pipelined-skew", "replay": replay, **world.check(g)}
                               for g in range(generations))
            report["images"][image] = {"status": "PASS" if all(r["status"] == "PASS" for r in results)
                                      else "FAIL", "checks": results}
        report["status"] = "PASS" if all(r["status"] == "PASS" for r in report["images"].values()) else "FAIL"
        return report


class AllRank:
    """All-rank event boundary: one batch time is max(rank elapsed), not a host join per step."""
    def __init__(self, world, image):
        self.world, self.image = world, image
        self.start = [b.event_create() for b in world.backends]
        self.end = [b.event_create() for b in world.backends]
        self.records = []

    def launch(self):
        for r in range(4):
            self.world.launch(r, self.image)

    def time(self, n):
        start = time.perf_counter()
        for b, e, stream in zip(self.world.backends, self.start, self.world.streams):
            b.event_record(e, stream)
        for _ in range(n):
            self.launch()
        for b, e, stream in zip(self.world.backends, self.end, self.world.streams):
            b.event_record(e, stream)
        self.world.synchronize()
        elapsed = [b.event_elapsed_time(s, e) for b, s, e in zip(self.world.backends, self.start, self.end)]
        self.records.append({"n": n, "per_rank_ms": elapsed, "wall_ms": (time.perf_counter() - start) * 1000})
        return max(elapsed)

    def close(self):
        for b, s, e in zip(self.world.backends, self.start, self.end):
            b.event_destroy(s)
            b.event_destroy(e)


def time_world(out, index, slices, slice_elements):
    from evaluator.harness.parity.example import time_pair
    with ExitStack() as stack:
        world = World(stack, out, index, slices=slices, slice_elements=slice_elements, generations=1)
        checks = []
        for image in ("linked", "one-tu"):
            for r in range(4):
                world.sequence(r, image, [0])
            world.synchronize()
            checks.append({"image": image, **world.check(0)})
        linked, control = AllRank(world, "linked"), AllRank(world, "one-tu")
        stack.callback(linked.close)
        stack.callback(control.close)
        timing = time_pair(linked, control)
        scored = 2 * len(timing["blocks"])
        timing["all_rank"] = {"linked": linked.records[-scored:], "one_tu": control.records[-scored:]}
        checks.append({"image": "after-timing", **world.check(0)})
        passed = (all(c["status"] == "PASS" for c in checks) and timing["status"] == "PASS"
                  and timing["ratio_linked_over_onetu"] <= 1.03)
        return {"status": "PASS" if passed else "FAIL", "geometry": world.geometry,
                "residency": world.residency, "elements": world.elements, "checks": checks,
                "timing": timing, "ratio_gate": 1.03, "clock_locked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "time", "sass", "sanitize", "single"))
    parser.add_argument("--profile", choices=("sm_103a",), default="sm_103a")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--gpus")
    parser.add_argument("--image", choices=("linked", "one-tu"), default="linked")
    parser.add_argument("--slices", type=int)
    parser.add_argument("--slice-elements", type=int, default=1024)
    parser.add_argument("--generations", type=int, default=6)
    args = parser.parse_args()
    from tools.pipelink_env import load
    config = load(REPO)
    os.environ.update(config)
    gpus = (args.gpus or config["PIPELINK_GPUS"]).split(",")
    if len(gpus) != 4:
        parser.error("exactly four caller-selected devices are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    scratch = Path(config["PIPELINK_SCRATCH_ROOT"]) / "multimem-domains" / "run"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["CUDA_CACHE_PATH"] = str(scratch / "cuda-cache")
    out = args.artifacts.resolve() if args.artifacts else HERE / "out" / args.profile
    index = json.loads((out / "index.json").read_text())
    if args.mode == "sass":
        from evaluator.harness.parity.example import sass_diagnostic
        report = sass_diagnostic(out / index["linked"], out / index["one_tu"], env=dict(os.environ),
                                 out_dir=out / "sass", backend_matched=out / index["one_tu_lto"])
        report["comparison"] = str(Path(report["comparison"]).relative_to(out))
    elif args.mode == "sanitize":
        sanitizer = Path(config["PIPELINK_CUDA_HOME"]) / "bin/compute-sanitizer"
        legs = []
        for image in ("linked", "one-tu"):
            for tool in ("racecheck", "synccheck"):
                log = out / "sanitize" / image / f"{tool}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = ["timeout", "180", str(sanitizer), "--tool", tool, "--error-exitcode", "1",
                           sys.executable, str(HERE / "run.py"), "single", "--artifacts", str(out),
                           "--gpus", ",".join(gpus), "--image", image, "--slices", "4", "--generations", "3"]
                completed = subprocess.run(command, capture_output=True, text=True)
                text = completed.stdout + completed.stderr
                log.write_text(text)
                unsupported = any(s in text.lower() for s in ("not supported", "unsupported"))
                status = "PASS" if completed.returncode == 0 else ("UNSUPPORTED" if unsupported else "FAIL")
                legs.append({"image": image, "tool": tool, "returncode": completed.returncode,
                             "status": status, "log": str(log.relative_to(out)), "command": command})
        report = {"legs": legs, "status": "PASS" if all(l["status"] == "PASS" for l in legs)
                  else "UNSUPPORTED" if all(l["status"] in ("PASS", "UNSUPPORTED") for l in legs) else "FAIL"}
    elif args.mode == "time":
        report = time_world(out, index, args.slices or 4096, args.slice_elements)
    else:
        report = check_world(out, index, slices=args.slices or 257, slice_elements=args.slice_elements,
                             generations=args.generations,
                             images=(args.image,) if args.mode == "single" else ("linked", "one-tu"))
    target = out / (f"results-single-{args.image}.json" if args.mode == "single" else f"results-{args.mode}.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
