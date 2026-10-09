#!/usr/bin/env python3
"""GPU-free checked-facts experiments; derived sources and images live in scratch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

PORT = Path(__file__).resolve().parents[1]
REPO = PORT.parents[2]
BASE = PORT / "build"
EVIDENCE = REPO / "campaigns/mok/evidence/p3"
OWNERS = ("framework", "comm", "gemm", "act")
PACKAGES = ("comm", "gemm", "act", "framework.fwd", "framework.bwd", "framework.rec")
DEFAULT_CELL = "mok-fwd-mxfp8-ep4-c0"
FORWARD = BASE / "links" / DEFAULT_CELL
SCRIPT = "campaigns/mok/port/p3/run_p3.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Experiment:
    def __init__(self, mode):
        from pipec2.host.abi import header_path
        if not os.environ.get("PIPELINK_CUDA_HOME") or not os.environ.get("PIPELINK_SCRATCH_ROOT"):
            raise RuntimeError("source tools/env.sh before running P3")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PIPEC_CUDA"] = os.environ["PIPELINK_CUDA_HOME"]
        self.mode = mode
        self.scratch = Path(os.environ["PIPELINK_SCRATCH_ROOT"])
        self.cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
        self.root = self.scratch / "mok/p3"
        self.out = self.root / mode
        self.out.mkdir(parents=True, exist_ok=True)
        self.nvcc = self.cuda / "bin/nvcc"
        spec = importlib.util.spec_from_file_location("mok_p2_build", PORT / "build.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        self.flags = build.FLAGS
        self.includes = [BASE / "csrc", BASE / "csrc/megakernel", PORT / "include",
                         *(PORT / owner for owner in OWNERS),
                         REPO / "third_party/mixture-of-kittens/third_party/ThunderKittens/include",
                         REPO / "pipec/src/pipec2/pack/include", header_path().parent]
        self.report = {"phase": mode, "status": "RUNNING", "gpu_execution": "NOT RUN",
                       "invocation": f"source tools/env.sh && CUDA_VISIBLE_DEVICES= uv run --project pipec --locked python {SCRIPT} {mode}",
                       "commands": [], "api_calls": []}

    def scrub(self, value):
        if isinstance(value, dict):
            return {k: self.scrub(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.scrub(v) for v in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            value = value.replace(str(self.scratch), "$PIPELINK_SCRATCH_ROOT")
            value = value.replace(str(self.cuda), "$PIPELINK_CUDA_HOME")
            value = value.replace(str(REPO) + "/", "")
            value = value.replace(sys.executable, "$PIPEC_PYTHON")
        return value

    def publish(self):
        save(EVIDENCE / f"{self.mode}.json", self.scrub(self.report))

    def command(self, argv, label):
        argv = list(map(str, argv))
        recorded = argv
        if argv[0] == sys.executable:
            recorded = ["uv", "run", "--project", "pipec", "--locked", "python", *argv[1:]]
        row = {"argv": self.scrub(recorded), "cwd": ".", "label": label}
        self.report["commands"].append(row)
        result = subprocess.run(argv, cwd=REPO, text=True, capture_output=True)
        row["returncode"] = result.returncode
        log = self.out / "logs" / f"{label}.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(result.stdout + result.stderr)
        row["log"] = self.scrub(log)
        if result.returncode:
            raise RuntimeError(f"[P3-BUILD] {self.scrub(log)}: exit {result.returncode}: {result.stderr[-6000:]}")
        return result

    def sources(self, owner, directory, old, new, extra=()):
        target = directory / "sources" / owner
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(PORT / owner, target)
        source = target / ("interp_fwd.cu" if owner == "framework" else f"{owner}.cu")
        text = source.read_text()
        for old, new in ((old, new), *extra):
            if text.count(old) != 1:
                raise RuntimeError(f"[P3-SOURCE] {source}: expected exactly one {old!r}")
            text = text.replace(old, new)
        source.write_text(text)
        return source

    def pack(self, owner, source, directory, provider=False):
        from pipec2.container import read
        from pipec2.host.abi import ABI_TYPES
        from pipec2.host.provider import build_provider
        directory.mkdir(parents=True, exist_ok=True)
        includes = [source.parent, *self.includes]
        stage_id = "mok.interp.fwd" if owner == "framework" else f"mok.{owner}"
        package_name = "framework.fwd" if owner == "framework" else owner
        package = directory / f"{package_name}.plink"
        self.command([sys.executable, "-m", "pipec2.cli", "pack", source,
                      "--profile", "sm_103a", "-o", package,
                      *[v for inc in includes for v in ("-I", inc)],
                      *[f"--nvcc-flag={f}" for f in [*self.flags, f"--frandom-seed={stage_id}"]]],
                     f"{directory.name}-{owner}-pack")
        manifest, _ = read(package)
        if provider:
            host = manifest["host"]
            inventory = host["provider"]["abi_types"]
            local = [row for row in inventory if row["name"] not in ABI_TYPES]
            kwargs = dict(owner=f"mok_{owner}",
                          owner_contract_digest=host["provider"]["owner_contract_digest"], profile="sm_103a",
                          dtypes_digest=host["target"]["dtypes"],
                          type_ids={r["name"]: r["id"] for r in local},
                          extra_types={r["name"]: r["fields"] for r in local},
                          extra_headers=[source.parent / ("framework_abi.h" if owner == "framework" else f"{owner}.cu")],
                          compiler=str(self.nvcc), compile_flags=["-x", "cu", "-O3", *self.flags,
                          *({"framework": ["-DMOK_FRAMEWORK_PASS=0"], "comm": ["-DMOK_COMM_PROVIDER"],
                             "act": ["-DMOK_ACT_PROVIDER"]}[owner]),
                          *[f"-I{inc}" for inc in includes], f"-L{self.cuda / 'lib64/stubs'}", "-lcuda"])
            output = directory / f"provider_{package_name}.so"
            sources = [source.parent / "provider.cu"]
            self.report["api_calls"].append(self.scrub({"api": "pipec2.host.provider.build_provider",
                                                        "sources": sources, "output": output, "kwargs": kwargs}))
            observed = build_provider(sources, output, **kwargs)
            if observed != inventory:
                raise RuntimeError(f"[P3-ABI] {output}: provider ABI inventory differs")
        return package

    def pipeline(self, directory, replacements=None, cell=DEFAULT_CELL):
        directory.mkdir(parents=True, exist_ok=True)
        baseline = BASE / "links" / cell
        pipeline = load(baseline / "pipeline.json")
        for stage in pipeline["pipeline"]["stages"]:
            original = (baseline / stage["artifact_path"]).resolve()
            artifact = (replacements or {}).get(original.stem, original)
            stage["artifact_path"] = os.path.relpath(artifact, directory)
        return pipeline

    def link(self, pipeline, directory):
        from pipec2.link import link
        path = directory / "pipeline.json"
        save(path, pipeline)
        self.report["api_calls"].append(self.scrub({"api": "pipec2.link.link", "pipeline_path": path,
                                                    "out_dir": directory}))
        return link(path, directory)

    def harness(self, directory, changed_provider=None, cell=DEFAULT_CELL):
        baseline = BASE / "links" / cell
        index = load(baseline / "index.json")
        # The harness opens its control backend even for --image linked.
        for key in ("one_tu", "one_tu_lto"):
            index[key] = os.path.relpath(baseline / index[key], directory)
        index.pop("diagnostics", None)
        link = load(directory / "link.json")
        for key in ("kernel_symbol", "threads", "dyn_smem_bytes", "regs_cap"):
            index[key] = link[key]
        for group, relative in index["providers"].items():
            original = (baseline / relative).resolve()
            source = directory / original.name
            if original.name != f"provider_{changed_provider}.so":
                shutil.copy2(original, source)
            index["providers"][group] = source.name
        save(directory / "cell-index.json", index)
        matrix = load(BASE / "matrix.json")
        for row in matrix["cells"]:
            row["index"] = ("cell-index.json" if row["id"] == cell else
                            os.path.relpath(BASE / row["index"], directory))
        save(directory / "matrix.json", matrix)
        save(directory / "index.json", {"matrix": "matrix.json"})

    def gpu_command(self, directory, tag, shape="sanity"):
        # Run from the repository root of the GPU machine.
        return ("source tools/env.sh && tools/gpu_window.sh uv run --project campaigns/mok/host --locked python "
                f"campaigns/mok/host/run.py check --build-dir \"{self.scrub(directory)}\" "
                f"--cell {DEFAULT_CELL} --world 4 --image linked --shape {shape} --steps 3 --timeout 300 --tag {tag} "
                "--source \"$PIPELINK_SCRATCH_ROOT/mok/harness/upstream-src\"")


def inventory(directory, replacements=None):
    from pipec2.container import read
    rows = {}
    for owner in PACKAGES:
        path = (replacements or {}).get(owner, directory / f"{owner}.plink")
        manifest, payload = read(path)
        rows[owner] = {"artifact_sha256": digest(path),
                       "payload_sha256": hashlib.sha256(payload).hexdigest(),
                       "declared_payload_sha256": manifest["payload"]["sha256"]}
    return rows


def facts_change(exp):
    before = inventory(BASE)
    providers_before = {owner: digest(BASE / f"provider_{owner}.so") for owner in PACKAGES}
    source = exp.sources("comm", exp.out, "DISPATCH_Nb = 512", "DISPATCH_Nb = 256")
    package = exp.pack("comm", source, exp.out, provider=True)
    cells = []
    for baseline in sorted((BASE / "links").iterdir()):
        if not (baseline / "pipeline.json").exists():
            continue
        directory = exp.out / "links" / baseline.name
        exp.link(exp.pipeline(directory, {"comm": package}, baseline.name), directory)
        shutil.copy2(exp.out / "provider_comm.so", directory / "provider_comm.so")
        exp.harness(directory, "comm", baseline.name)
        host = load(directory / "link.json")["host"]
        fact = host["fact_inventory"]["mok.comm.DISPATCH_Nb"]
        if fact["value"] != 256 or set(fact["consumers"]) != {"mok.framework", "mok.gemm"}:
            raise RuntimeError(f"[P3-FACT] {baseline.name}: incorrect DISPATCH_Nb propagation")
        cells.append({"cell": baseline.name, "facts": host["facts"], "dispatch_fact": fact,
                      "image": directory / "pipeline.cubin",
                      "providers": {p: digest(directory / f"provider_{p}.so")
                                    for p in (f"framework.{baseline.name.split('-')[1]}", "comm", "gemm", "act")}})
    if len(cells) != 24:
        raise RuntimeError(f"[P3-MATRIX] expected 24 cells, got {len(cells)}")
    after = inventory(BASE, {"comm": package})
    providers_after = {p: digest((exp.out if p == "comm" else BASE) / f"provider_{p}.so") for p in PACKAGES}
    if inventory(BASE) != before or any(digest(BASE / f"provider_{p}.so") != providers_before[p] for p in PACKAGES):
        raise RuntimeError("[P3-IDENTITY] baseline packages/providers changed")
    for owner in PACKAGES:
        if owner != "comm" and (before[owner] != after[owner] or providers_before[owner] != providers_after[owner]):
            raise RuntimeError(f"[P3-IDENTITY] {owner}: non-comm artifact changed")
    exp.report.update(status="PASS", before=before, after=after, cells=cells,
                      unchanged_owners=[p for p in PACKAGES if p != "comm"],
                      providers_before=providers_before, providers_after=providers_after,
                      source_change={"path": "campaigns/mok/port/comm/comm.cu",
                                     "old": "DISPATCH_Nb = 512", "new": "DISPATCH_Nb = 256"},
                      gpu_command=exp.gpu_command(exp.out / "links" / DEFAULT_CELL, "p3-facts-change"))


def mutate_package(owner, path, mutation):
    from pipec2.container import read, pack
    manifest, payload = read(BASE / f"{'framework.fwd' if owner == 'framework' else owner}.plink")
    mutation(manifest)
    del manifest["payload"]["sha256"]
    pack(manifest, payload, path)


def negative(exp):
    from pipec2.model import PipeLinkError
    rows = []
    cases = [
        ("mixed-precision", "comm", {"L-HANDLE"}, "Restrict comm to BF16 while the P2 forward cell forces MXFP8"),
        ("topology", "act", {"L-TOPO"}, "Repack act with 1-CTA group and (1,1,1) compiled/launch cluster"),
        ("tmem-capacity", "gemm", {"L-OVERLAP"}, "Increase GEMM TMEM extent from 512 to 544 columns"),
        ("counter-capacity", None, {"L-OVERLAP"}, "Reduce forward pipeline gmem_counter capacity by one slot"),
        ("counter-scope", "comm", {"L-PROTO", "L-PAIR"}, "Change x_routed_ready producer to system scope and sys release/acquire"),
        ("register-pool", "framework", {"L-REGS", "L-HANDLE"}, "Add a 256-register all-worker group against a 224-register launch pool"),
        ("counter-no-producer", "comm", {"L-PAIR"}, "Prune x_routed_ready producer to backward-only exports, retaining its forward consumer"),
    ]
    for name, owner, clauses, change in cases:
        directory = exp.out / name
        directory.mkdir(exist_ok=True)
        replacements = {}
        if owner:
            package = directory / f"{owner}.plink"
            if name == "topology":
                source = exp.sources(owner, directory,
                    'pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}',
                    'pl::topology{1, pl::cluster{{1,1,1}, pl::launch{1,{1,1,1}}}')
                exp.pack(owner, source, directory)
            else:
                def mutation(manifest):
                    if name == "mixed-precision":
                        manifest["stage"]["variants"] = [v for v in manifest["stage"]["variants"]
                                                          if v["key"]["pipelink_mok_precision"] == 0]
                        manifest["host"]["variants"]["mok.comm"] = [
                            v for v in manifest["host"]["variants"]["mok.comm"]
                            if v["key"]["pipelink_mok_precision"] == 0]
                    elif name == "tmem-capacity":
                        next(c for c in manifest["claims"] if c["domain"] == "tmem")["extent"]["const"] = 544
                    elif name == "counter-scope":
                        endpoint = next(e for e in manifest["endpoints"] if e["name"] == "x_routed_ready")
                        endpoint["scope"] = "system"
                        endpoint["protocol"]["release"] = "red.release.sys"
                        endpoint["protocol"]["acquire"] = "ld.acquire.sys"
                    elif name == "counter-no-producer":
                        next(e for e in manifest["endpoints"] if e["name"] == "x_routed_ready")["exports"] = ["mok_comm_bwd"]
                    elif name == "register-pool":
                        manifest["claims"].append({"domain": "regs", "handle": "pipelink_mok_worker_regs",
                                                  "reg_group": "worker", "roles": ["worker"], "values": [256]})
                mutate_package(owner, package, mutation)
            replacements["framework.fwd" if owner == "framework" else owner] = package
        pipeline = exp.pipeline(directory, replacements)
        if name == "counter-capacity":
            capacity = pipeline["pipeline"]["domains"]["gmem_counter"]["capacity"]
            pipeline["pipeline"]["domains"]["gmem_counter"]["capacity"] = capacity - 1
            change = f"Reduce pipeline gmem_counter capacity from {capacity} to {capacity - 1}"
        if name == "register-pool":
            for handle, group, value in (("pipelink_mok_worker_regs", "worker", 256),
                                         ("pipelink_launch_regs", "launch", 224)):
                pipeline["pipeline"]["handles"][handle] = {
                    "class": "regs", "binding": "bind", "domain": "regs", "values": [value],
                    "owner": f"reg_group:{group}"}
        row = {"case": name, "change": change, "expected_clauses": sorted(clauses),
               "pipeline": directory / "pipeline.json", "mutation_kind": "source-repack" if name == "topology" else "declaration"}
        try:
            exp.link(pipeline, directory)
        except PipeLinkError as error:
            row["failures"] = [failure.to_json_obj() for failure in error.failures]
            row["status"] = "REJECTED" if clauses & {f.clause_id for f in error.failures} else "WRONG CLAUSE"
        else:
            row.update(status="UNSAFE ACCEPT", failures=[])
        rows.append(row)
        exp.report["cases"] = rows
        exp.publish()
    directory = exp.out / "provider-padding"
    source = exp.sources("framework", directory,
                         'fact("mok.framework.EXPERT_PADDING").i32(256)',
                         'fact("mok.framework.EXPERT_PADDING").i32(384)')
    package = exp.pack("framework", source, directory, provider=True)
    exp.link(exp.pipeline(directory, {"framework.fwd": package}), directory)
    exp.harness(directory, "framework.fwd")
    exp.report.update(status="PASS" if all(row["status"] == "REJECTED" for row in rows) else "FAIL",
                      declaration_trust="Manifest-only mutations retain original stage.o and source/pack provenance; provenance no longer attests the mutated declarations. PipeLink trusts declarations (L3). Payload checksums are recomputed and valid. These are rejection fixtures, never launchable packages.",
                      provider_case={"status": "PREPARED; NOT RUN", "EXPERT_PADDING": 384, "MLP_Mb": 256,
                                     "expected": "GEMM provider query rejects EXPERT_PADDING % MLP_Mb != 0",
                                     "source": "campaigns/mok/port/gemm/provider.cu:46",
                                     "gpu_command": exp.gpu_command(directory, "p3-provider-padding", "sanity")})


def rdc_contrast(exp):
    from pipec2.container import read
    changed = exp.root / "facts-change/comm.plink"
    if not changed.exists():
        source = exp.sources("comm", exp.out, "DISPATCH_Nb = 512", "DISPATCH_Nb = 256")
        changed = exp.pack("comm", source, exp.out)
    manifest, _ = read(changed)
    if next(f for f in manifest["host"]["published_facts"] if f["name"] == "mok.comm.DISPATCH_Nb")["value"] != 256:
        raise RuntimeError(f"[P3-FACT] {changed}: changed comm package does not publish 256")
    objects = []
    for owner in ("framework.fwd", "comm", "gemm", "act"):
        _, payload = read(BASE / f"{owner}.plink")
        path = exp.out / f"{owner}.o"
        path.write_bytes(payload)
        objects.append(path)
    _, payload = read(changed)
    changed_object = exp.out / "comm-256.o"
    changed_object.write_bytes(payload)
    generated = []
    for stem in ("binding_stub", "dispatch"):
        source = exp.out / f"{stem}.cu"
        shutil.copy2(FORWARD / "generated" / source.name, source)
        obj = exp.out / f"{stem}.o"
        exp.command([exp.nvcc, "-std=c++20", "-O3", "--use_fast_math", "-gencode=arch=compute_103a,code=lto_103a",
                     "-rdc=true", "-dc", f"-I{REPO / 'pipec/src/pipec2/pack/include'}", source, "-o", obj],
                    f"compile-{stem}")
        generated.append(obj)
    images = []
    for name, comm in (("baseline", objects[1]), ("stale-comm-256", changed_object)):
        directory = exp.out / name
        directory.mkdir(exist_ok=True)
        inputs = [objects[0], comm, *objects[2:], *generated]
        exp.command([exp.nvcc, "-arch=sm_103a", "--use_fast_math", "-rdc=true", "-dlink", "-dlto", "-cubin",
                     *inputs, "-o", directory / "pipeline.cubin"], f"device-link-{name}")
        shutil.copy2(FORWARD / "link.json", directory / "link.json")
        exp.harness(directory)
        images.append({"case": name, "status": "DEVICE LINK ACCEPTED", "image": directory / "pipeline.cubin",
                       "cubin_sha256": digest(directory / "pipeline.cubin"),
                       "objects": {p.name: digest(p) for p in inputs},
                       "gpu_command": exp.gpu_command(directory, f"p3-rdc-{name}", "families")})
    exp.report.update(status="PASS", images=images,
                      generated_source_origin="Unchanged copies of P2 forward cell generated/binding_stub.cu and dispatch.cu; no hand-written equivalent. These are required glue for the opaque stage call ABI and host residue offsets.",
                      checked="nvcc -rdc=true -dlink -dlto accepts symbol resolution, target code generation and ordinary device-link constraints for the same package LTO stage.o payloads.",
                      not_checked="Plain device linking does not consume manifests: no fact propagation, variant intersection, resource-claim solver, counter protocol, or provider-contract validation.",
                      stale_contract="Both images retain original host link metadata and all original providers (DISPATCH_Nb=512). Only stale image comm stage.o uses 256; GEMM, framework, act and generated glue are byte-identical. Old GEMM code is fact-driven, so old GEMM plus REFRESHED facts would be valid P3a, not a stale counterexample. The stale host facts and comm descriptors are intentionally preserved here.",
                      watchdog="Host launcher --timeout 300 terminates the whole torchrun process group, then kills after 10 seconds. Runtime correctness, failure or hang is not inferred from successful device linking.")


def skew_cell(pass_name):
    return f"mok-{pass_name}-mxfp8-ep1-c0"


# One owner upgrades a geometry constant another owner schedules against. Depth edits
# also update the owner's own semaphore size and claim, so each package is internally
# consistent. Neither owner's provider reads these constants (comm/provider.cu,
# act/provider.cu), so stale providers isolate the cross-owner fact.
SKEW = (
    ("combine-nb", "comm", "fwd", "mok.comm.COMBINE_Nb", 512,
     (("COMBINE_Nb = 1024", "COMBINE_Nb = 512"),)),
    ("combine-depth", "comm", "fwd", "mok.comm.COMBINE_PIPE_DEPTH", 6,
     (("COMBINE_PIPE_DEPTH = 7", "COMBINE_PIPE_DEPTH = 6"),
      ("sizeof(Semaphores) == 64", "sizeof(Semaphores) == 56"),
      ('"smem", 64, 8, "kernel"', '"smem", 56, 8, "kernel"'))),
    ("dispatch-nb", "comm", "fwd", "mok.comm.DISPATCH_Nb", 256,
     (("DISPATCH_Nb = 512", "DISPATCH_Nb = 256"),)),
    ("swiglu-fwd-depth", "act", "fwd", "mok.act.SWIGLU_FWD_PIPE_DEPTH", 2,
     (("SWIGLU_FWD_PIPE_DEPTH = 3", "SWIGLU_FWD_PIPE_DEPTH = 2"),
      ("sizeof(Semaphores) == 40", "sizeof(Semaphores) == 32"),
      ('"smem", 40, 8, "kernel"', '"smem", 32, 8, "kernel"'))),
    ("swiglu-bwd-depth", "act", "bwd", "mok.act.SWIGLU_BWD_PIPE_DEPTH", 1,
     (("SWIGLU_BWD_PIPE_DEPTH = 2", "SWIGLU_BWD_PIPE_DEPTH = 1"),
      ("sizeof(Semaphores) == 40", "sizeof(Semaphores) == 32"),
      ('"smem", 40, 8, "kernel"', '"smem", 32, 8, "kernel"'))),
)
# The sanity shape cannot distinguish these task counts; this upstream family can.
SKEW_SHAPE = ("Minimum local experts", "Default")


def skew(exp):
    from pipec2.container import read
    objects = {}
    for owner in ("framework.fwd", "framework.bwd", "comm", "gemm", "act"):
        _, payload = read(BASE / f"{owner}.plink")
        objects[owner] = exp.out / f"{owner}.o"
        objects[owner].write_bytes(payload)
    glue = {}
    for pass_name in ("fwd", "bwd"):
        glue[pass_name] = []
        for stem in ("binding_stub", "dispatch"):
            source = exp.out / f"glue-{pass_name}" / f"{stem}.cu"
            source.parent.mkdir(exist_ok=True)
            shutil.copy2(BASE / "links" / skew_cell(pass_name) / "generated" / source.name, source)
            obj = source.with_suffix(".o")
            exp.command([exp.nvcc, "-std=c++20", "-O3", "--use_fast_math", "-gencode=arch=compute_103a,code=lto_103a",
                         "-rdc=true", "-dc", f"-I{REPO / 'pipec/src/pipec2/pack/include'}", source, "-o", obj],
                        f"compile-{pass_name}-{stem}")
            glue[pass_name].append(obj)
    rows = []
    controls = [(f"baseline-{p}", None, p, None, None, ()) for p in ("fwd", "bwd")]
    for name, owner, pass_name, fact, value, edits in (*controls, *SKEW):
        directory = exp.out / name
        directory.mkdir(exist_ok=True)
        cell = skew_cell(pass_name)
        baseline = BASE / "links" / cell
        changed = None
        row = {"case": name, "owner": owner, "cell": cell}
        if owner:
            source = exp.sources(owner, directory, *edits[0], extra=edits[1:])
            package = exp.pack(owner, source, directory, provider=True)
            manifest, payload = read(package)
            published = next(f["value"] for f in manifest["host"]["published_facts"] if f["name"] == fact)
            if published != value:
                raise RuntimeError(f"[P3-FACT] {package}: publishes {fact}={published}, expected {value}")
            changed = directory / f"{owner}.o"
            changed.write_bytes(payload)
            row.update(fact=fact, source_change=[{"old": o, "new": n} for o, n in edits],
                       package_sha256=digest(package))
        inputs = [changed if o == owner else objects[o]
                  for o in (f"framework.{pass_name}", "comm", "gemm", "act")] + glue[pass_name]
        rdc = directory / "rdc"
        rdc.mkdir(exist_ok=True)
        exp.command([exp.nvcc, "-arch=sm_103a", "--use_fast_math", "-rdc=true", "-dlink", "-dlto", "-cubin",
                     *inputs, "-o", rdc / "pipeline.cubin"], f"{name}-rdc")
        shutil.copy2(baseline / "link.json", rdc / "link.json")
        exp.harness(rdc, cell=cell)
        arms = {"rdc": {"build_dir": rdc, "image_sha256": digest(rdc / "pipeline.cubin"),
                        "link": "nvcc -rdc -dlink -dlto accepted",
                        "objects": {p.name: digest(p) for p in inputs}}}
        if owner:
            linked = directory / "pipelink"
            exp.link(exp.pipeline(linked, {owner: package}, cell), linked)
            shutil.copy2(directory / f"provider_{owner}.so", linked / f"provider_{owner}.so")
            exp.harness(linked, owner, cell)
            facts = {key: load(path / "link.json")["host"]["fact_inventory"][fact]
                     for key, path in (("rdc", rdc), ("pipelink", linked))}
            if facts["pipelink"]["value"] != value or facts["rdc"]["value"] == value:
                raise RuntimeError(f"[P3-FACT] {name}: expected stale RDC and propagated PipeLink {fact}")
            arms["rdc"]["fact"] = facts["rdc"]
            arms["pipelink"] = {"build_dir": linked, "image_sha256": digest(linked / "pipeline.cubin"),
                                "link": "pipec2 link accepted", "fact": facts["pipelink"]}
        else:
            arms["pipelink"] = {"build_dir": BASE, "image_sha256": digest(baseline / "pipeline.cubin"),
                                "link": "P2 build"}
        row["arms"] = arms
        rows.append(row)
        exp.report["cases"] = rows
        exp.publish()
    exp.report.update(status="PASS", cases=rows, shape={"name": SKEW_SHAPE[0], "parameter_family": SKEW_SHAPE[1]},
                      definition="One owner repacked with a changed geometry constant; all other owners' stage.o reused. "
                                 "RDC: plain nvcc device link of the same payloads with the cell's unchanged generated glue, "
                                 "link.json host metadata/facts and providers. PipeLink: pipec2 link with the repacked package "
                                 "and its rebuilt provider.",
                      gpu_execution="NOT RUN; run_p3.py skew --gpu")


def skew_errors(cell):
    """Maximum absolute error per compared tensor over direct steps and graph replays."""
    worst = {}
    reports = [r for step in cell["steps"] for r in step["images"].get("linked", {}).items()]
    reports += [r for graph in cell["graphs"] for r in graph["reference"].items()]
    for key, report in reports:
        worst.setdefault(key, []).append(float(report["abs_max"]))
    # NaN dominates: an output that is NaN anywhere is reported as NaN.
    return {key: max(values, key=lambda v: (math.isnan(v), v)) for key, values in worst.items()}


def skew_gpu(exp):
    exp.report = load(EVIDENCE / "skew.json")
    if exp.report["status"] != "PASS":
        raise RuntimeError("[P3-SKEW] build record is not PASS")
    env = dict(os.environ, MOK_EVIDENCE_DIR=str(EVIDENCE))
    env.pop("CUDA_VISIBLE_DEVICES")
    for row in exp.report["cases"]:
        for arm, record in row["arms"].items():
            tag = f"p3-skew-{row['case']}-{arm}"
            argv = ["uv", "run", "--project", "campaigns/mok/host", "--locked", "python", "campaigns/mok/host/run.py",
                    "check", "--build-dir", record["build_dir"].replace("$PIPELINK_SCRATCH_ROOT", str(exp.scratch)),
                    "--cell", row["cell"], "--world", "1", "--image", "linked", "--shape", "families",
                    "--shape-name", SKEW_SHAPE[0], "--parameter-family", SKEW_SHAPE[1],
                    "--steps", "3", "--timeout", "300", "--tag", tag]
            result = subprocess.run(argv, cwd=REPO, env=env, text=True, capture_output=True)
            summary = load(EVIDENCE / f"{tag}-check-summary.json")
            window = load(EVIDENCE / f"{tag}-check-window.json")
            gpu = {"command": exp.scrub(argv), "returncode": result.returncode,
                   "gpu_window": [l for l in window["stderr"].splitlines() if l.startswith("gpu_window:")],
                   "evidence": f"campaigns/mok/evidence/p3/{tag}-check-linked-rank0.json"}
            run_code = summary["runs"][0]["returncode"]
            cells = load(EVIDENCE / f"{tag}-check-linked-rank0.json")["cells"]
            if run_code == 124:
                gpu["outcome"] = "HANG: 300-second watchdog, no configuration completed"
            elif cells:
                gpu["outcome"] = cells[0]["status"]
                gpu["max_abs_error"] = {k: v if math.isfinite(v) else str(v)
                                        for k, v in skew_errors(cells[0]).items()}
            else:
                gpu["outcome"] = f"ERROR: torchrun exit {run_code}"
            record["gpu"] = gpu
            exp.publish()
            print(json.dumps({"case": row["case"], "arm": arm, "outcome": gpu["outcome"]}), flush=True)
    baseline = [r["arms"][a]["gpu"]["outcome"] for r in exp.report["cases"] if not r["owner"] for a in ("rdc", "pipelink")]
    exp.report["gpu_execution"] = "RUN"
    exp.report["status"] = "PASS" if all(o == "PASS" for o in baseline) and all(
        r["arms"]["pipelink"]["gpu"]["outcome"] == "PASS" for r in exp.report["cases"]) else "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("facts-change", "negative", "rdc-contrast", "skew"))
    parser.add_argument("--gpu", action="store_true", help="skew only: run the built images on one GPU")
    args = parser.parse_args()
    if args.gpu and args.mode != "skew":
        parser.error("--gpu applies only to skew")
    exp = Experiment(args.mode)
    try:
        {"facts-change": facts_change, "negative": negative, "rdc-contrast": rdc_contrast,
         "skew": skew_gpu if args.gpu else skew}[args.mode](exp)
    except Exception as error:
        exp.report.update(status="FAIL", error={"type": type(error).__name__, "message": str(error)})
        if hasattr(error, "failures"):
            exp.report["error"]["failures"] = [f.to_json_obj() for f in error.failures]
        exp.publish()
        raise
    exp.publish()
    print(json.dumps(exp.scrub(exp.report), indent=2))
    return 0 if exp.report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
