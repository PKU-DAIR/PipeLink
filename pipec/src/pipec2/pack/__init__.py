"""Owner-side stage packaging: annotated source in, attested source-free artifact out."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


def pack_cu(
    source: Path, out: Path, *, includes: Sequence[Path] = (), defines: Sequence[str] = (),
    nvcc_flags: Sequence[str] = (), profile: str = "sm_103a", toolkit: str = "cuda-13.3.0",
) -> dict[str, Any]:
    """Compile one object, fit, cross-check and attest its declared variants without a GPU."""
    from .. import container, symbols
    from ..model import PackError
    from ..profile import get_profile
    from ..schema import load_stage_manifest
    from .assemble import assemble, check_sources
    from .emit import compile_registry
    from .objectfacts import observe

    selected = get_profile(profile)
    env = os.environ.copy()
    cuda = symbols.cuda_home(env)
    env["PATH"] = str(cuda / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    root = env.get("PIPELINK_SCRATCH_ROOT")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pipec2-pack-", dir=root) as directory:
        scratch = Path(directory)
        compilation = compile_registry(source, scratch, includes=includes, defines=defines,
                                       nvcc_flags=nvcc_flags, profile=profile, toolkit=toolkit, env=env)
        manifest_obj = assemble(compilation.registry, profile=profile, toolkit=toolkit,
                                provenance_inputs=compilation.provenance_inputs, math=compilation.math)
        seed = symbols.import_seed(load_stage_manifest(manifest_obj))
        facts = observe(compilation.object_path, scratch, seed, env, selected)
        check_sources(compilation.registry, manifest_obj, facts)
        manifest = load_stage_manifest(manifest_obj)
        failures = [*symbols.check_pack(facts.symbols, manifest),
                    *symbols.check_imports(facts.symbols, (*symbols.TOOLCHAIN_ALLOWLIST,
                      *symbols.exact_import_patterns(symbols.declared_imports(manifest))))]
        if failures:
            raise PackError(failures)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = container.pack(manifest_obj, compilation.object_path.read_bytes(), out)
        params = next(record for record in compilation.registry["records"] if record["kind"] == "params")
        report["params"] = {key: params[key] for key in ("type", "bytes", "align", "members")}
        report["compile"] = compilation.compile
    report.update({"stage_id": manifest.id, "allowlist_version": symbols.ALLOWLIST_VERSION,
                   "symbol_mechanism": symbols.MECHANISM, "symbols": facts.symbols.to_json_obj(),
                   "comdat_advisory": symbols.advisory_report(facts.symbols)})
    return report


__all__ = ["pack_cu"]
