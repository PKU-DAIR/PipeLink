"""Turn the finite annotation registry into the linker's tensor-v2 carrier."""
from __future__ import annotations

from collections import defaultdict
from itertools import product
import re
from typing import Any, Mapping

from ..canonjson import canonical_bytes, sha256_hex
from ..decls import declaration_digest, normalize_declaration
from ..model import ClauseFailure, MathMode, PackError, register_clause
from ..profile import get_profile
from ..schema import _normalized_prefix
from ..tensor import expr
from ..tensor.fit import fit_named, fit_named_or_raise
from .objectfacts import ObjectFacts
from .host import RECORD_KINDS as HOST_RECORD_KINDS, assemble_host


for _clause, _description in {
    "P-SYM": "Declared callable symbols agree with the compiled object.",
    "P-LAUNCH": "Declared role threads respect compiled launch bounds.",
    "P-CLAIM": "Type-derived shared storage has consistent extent and alignment.",
    "P-TOPO": "Declared and compiled cluster dimensions agree.",
    "P-MAP": "Registry maps are total and demand participants belong to their sites.",
    "P-INIT": "A declared initializing warp belongs to its init owner's absolute warp claim.",
    "P-TENSOR-SOURCE": "Tensor source parameters use the opaque tensor source ABI.",
    "P-TENSOR-CALLS": "Tensor exports name their declared source and release call endpoints.",
}.items():
    register_clause(_clause, phase="pack", description=_description)


def _reject(clause: str, path: str, message: str, **detail: Any) -> None:
    raise PackError([ClauseFailure(clause, path, message, detail)])


def _records(registry: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    kinds = {"stage", "params", "type", "role", "export", "call", "endpoint", "claim", "claim_smem", "claim_regs",
             "variant", "descriptor", "topology", "scheduler_strategy", "tensor_export", "tensor_import", "site", "binding", "map", "hint"}
    kinds |= HOST_RECORD_KINDS
    if set(registry) != {"records"} or not isinstance(registry["records"], list):
        _reject("P-MAP", "$.records", "expected a registry records array")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str | int]] = set()
    for index, record in enumerate(registry["records"]):
        if not isinstance(record, dict) or record.get("kind") not in kinds:
            _reject("P-MAP", f"$.records[{index}]", "unknown registry record kind")
        kind = record["kind"]
        name = record.get("name", record.get("handle", record.get("key", "")))
        if kind == "variant":
            name = record.get("index")
            if type(name) is not int or name < 0:
                _reject("P-MAP", f"$.records[{index}].index", "variant index must be a nonnegative integer")
        namespace = "claim" if kind in {"claim_smem", "claim_regs"} else "endpoint" if kind == "call" else kind
        # One channel name may carry a produce and a consume endpoint in the same stage.
        key = (namespace, name, record.get("direction")) if namespace == "endpoint" else (namespace, name)
        if key in seen:
            _reject("P-MAP", f"$.records[{index}]", "duplicate registry record", kind=kind, name=name)
        seen.add(key)
        grouped[kind].append(record)
    for kind in ("stage", "params"):
        if len(grouped[kind]) != 1:
            _reject("P-MAP", "$.records", f"expected exactly one {kind} record")
    return grouped


class _Maps:
    def __init__(self, records: list[dict[str, Any]], bindings: set[str]) -> None:
        self.records = {record["name"]: record for record in records}
        self.fitted: dict[str, dict[str, list[Any]]] = {}
        self.domains: dict[str, list[dict[str, Any]]] = {}
        for name, record in self.records.items():
            path = f"$.maps.{name}"
            inputs = []
            for raw in record["inputs"]:
                role = raw.get("role", "bind" if raw["name"] in bindings else "coord")
                if role not in {"coord", "bind"}:
                    _reject("P-MAP", path, "map input has an unknown role", input=raw)
                if role == "bind" and raw["name"] not in bindings:
                    _reject("P-MAP", path, "map binding must be declared by PL_BINDING in the same TU", input=raw["name"])
                domain_json = {key: value for key, value in raw.items() if key != "role"}
                try:
                    domain = expr.Domain.from_json_obj(domain_json)
                except expr.ExprError as error:
                    _reject("P-MAP", path, "map input requires a finite domain", input=raw, error=str(error))
                inputs.append((domain, role))
            names = [domain.name for domain, _ in inputs]
            outputs = record["outputs"]
            if len(set(names)) != len(names) or not outputs or len(set(outputs)) != len(outputs):
                _reject("P-MAP", path, "map input and output names must each be unique")
            domains = [domain.values if domain.values is not None else range(domain.lo, domain.hi)
                       for domain, _ in inputs]
            count = 1
            for domain in domains:
                count *= len(domain)
            rows = record["rows"]
            if not count or len(rows) != count:
                _reject("P-MAP", path, "map rows do not cover the full input product", expected=count, actual=len(rows))
            width = len(inputs) + len(outputs)
            for index, (point, row) in enumerate(zip(product(*domains), rows)):
                if (len(row) != width or any(type(value) is not int for value in row)
                        or tuple(row[:len(inputs)]) != point):
                    _reject("P-MAP", path, "map row is not the next complete input/output row",
                            row_index=index, row=row, expected_inputs=list(point))
            env = {domain.name: domain for domain, _ in inputs}
            fitted = {}
            for index, output in enumerate(outputs):
                fitted_expr = fit_named(rows, inputs, index, sort_env=env)
                if fitted_expr is None:
                    # Only the failure path repeats fitting to obtain its exact first-row witness.
                    fitted_expr = fit_named_or_raise(rows, inputs, index, sort_env=env,
                                                    output_name=f"{name}.{output}")
                fitted[output] = fitted_expr.to_json_obj()
            self.fitted[name] = fitted
            self.domains[name] = [domain.to_json_obj() for domain, role in inputs if role == "coord"]

    def get(self, name: str, outputs: list[str]) -> dict[str, list[Any]]:
        if name not in self.fitted:
            _reject("P-MAP", f"$.maps.{name}", "annotation references an absent map")
        missing = set(outputs) - self.fitted[name].keys()
        if missing:
            _reject("P-MAP", f"$.maps.{name}", "map lacks required outputs", missing=sorted(missing))
        return {key: self.fitted[name][key] for key in outputs}

    def values(self, name: str, output: str) -> set[int]:
        self.get(name, [output])
        record = self.records[name]
        column = len(record["inputs"]) + record["outputs"].index(output)
        return {row[column] for row in record["rows"]}


def _declaration(record: Mapping[str, Any]) -> dict[str, str]:
    declaration = normalize_declaration(record["declaration"])
    return {"decl": declaration, "decl_sha256": declaration_digest(declaration)}


def _signature(record: Mapping[str, Any]) -> str:
    declaration = normalize_declaration(record["declaration"])
    symbol = record["entry"] if record["kind"] == "variant" else record["symbol"]
    match = re.search(r'extern "C" __device__ (.+?)\s+' + re.escape(symbol) + r'\((.*)\);?\Z', declaration)
    if match is None:
        _reject("P-SYM", f"$.records.{record.get('name', record.get('index'))}", "unsupported emitted callable declaration")
    return f"{match[1]}({match[2]})"


def _call(record: dict[str, Any]) -> dict[str, Any]:
    declaration = _declaration(record)
    export = record.get("export", record["name"])
    return {
        "name": record["name"], "direction": record["direction"], "owner_role": record["role"],
        "space": "reg", "scope": "cta", "proxy": "generic", "tile": {"dtype": "u64", "shape": [1]},
        "layout": {"encoding": "pl-layout-v2", "shape": [[1]], "stride": [[1]], "elem_bits": 64},
        "protocol": {"kind": "call", "export": export, "signature": _signature(record), **declaration},
        "binds": {},
    }


def _claim_roles(claim: dict[str, Any], roles: dict[str, Any]) -> list[str]:
    # Like the solver, a single-role stage may leave interval claim roles implicit.
    return claim.get("roles") or (list(roles) if len(roles) == 1 else [])


def _check_init_warps(manifest: dict[str, Any]) -> None:
    roles = manifest["stage"]["roles"]
    for index, endpoint in enumerate(manifest["endpoints"]):
        protocol = endpoint["protocol"]
        if protocol["kind"] != "mbarrier" or "warp" not in protocol["init"]:
            continue
        init = protocol["init"]
        if "owner" not in init:
            continue  # The linker derives the owner from the solved warp intervals.
        for claim in manifest["claims"]:
            if (claim["domain"] != "warp" or "absolute_base" not in claim
                    or init["owner"] not in _claim_roles(claim, roles)):
                continue
            extent = claim["extent"]
            for variant in manifest["stage"]["variants"]:
                values = variant["key"]
                if "table" in extent:
                    sizes = [size for key, size in extent["table"].items()
                             if all(values.get(name) == int(value)
                                    for name, value in (part.split("=") for part in key.split(",")))]
                    if not sizes:
                        continue  # Partner-owned finite handles are resolved at link.
                    size = sizes[0]
                else:
                    if not extent["per"].keys() <= values.keys():
                        continue
                    size = extent["const"] + sum(values[name] * coefficient
                                                 for name, coefficient in extent["per"].items())
                base = claim["absolute_base"]
                if not base <= init["warp"] < base + size:
                    _reject("P-INIT", f"$.endpoints[{index}].protocol.init.warp",
                            "initializing warp is outside the init owner's absolute warp claim",
                            owner=init["owner"], warp=init["warp"], handle=claim["handle"],
                            base=base, extent=size, variant=variant["index"])


def assemble(
    registry: Mapping[str, Any], *, profile: str = "sm_103a", toolkit: str = "cuda-13.3.0",
    provenance_inputs: Mapping[str, str] | None = None,
    facts: ObjectFacts | None = None,
    math: MathMode | None = None,
) -> dict[str, Any]:
    """Assemble and fit one registry; optional object facts enforce cross-source checks."""
    records = _records(registry)
    stage = records["stage"][0]
    params = records["params"][0]
    prefix = _normalized_prefix(stage["id"])
    maps = _Maps(records["map"], {record["name"] for record in records["binding"]})
    roles = {record["name"]: {"entry": record["symbol"], "warps": record["warps"]} for record in records["role"]}
    for record in records["role"]:
        signature = _signature(record)
        if re.sub(r"\s+", "", signature) not in {
            "void(constvoid*,uint32_t)", "void(constvoid*,unsigned)", "void(constvoid*,unsignedint)",
        }:
            _reject("P-SYM", f"$.stage.roles.{record['name']}", "role declaration does not implement the fixed v2 entry ABI",
                    signature=signature)
    exports = {record["name"]: {"symbol": record["symbol"], **_declaration(record)} for record in records["export"]}
    claims = []
    for record in [*records["claim"], *records["claim_smem"]]:
        typed = record["kind"] == "claim_smem"
        if typed and (type(record["align"]) is not int or record["align"] <= 0 or record["bytes"] % record["align"]):
            _reject("P-CLAIM", f"$.claims.{record['handle']}", "sizeof shared storage must be divisible by its alignment",
                    bytes=record["bytes"], align=record["align"])
        extent = record["bytes"] if typed else record["extent"]
        claim = {"handle": record["handle"], "domain": "smem" if typed else record["domain"],
                 "extent": {"const": extent, "per": {}} if type(extent) is int else extent,
                 "align": record["align"], "lifetime": record["lifetime"]}
        if record.get("roles"):
            claim["roles"] = record["roles"]
        claim.update({key: record[key] for key in ("absolute_base", "alias", "offset") if key in record})
        claims.append(claim)
    claims.extend({"domain": "regs", **{key: record[key] for key in ("handle", "reg_group", "values", "roles")}}
                  for record in records["claim_regs"])
    covered = {role for claim in claims if claim["domain"] == "warp" for role in _claim_roles(claim, roles)}
    for name, role in roles.items():
        if name not in covered:
            claims.append({"handle": f"pipelink_{name}_warps", "domain": "warp",
                           "extent": {"const": role["warps"], "per": {}}, "align": 1,
                           "lifetime": "kernel", "roles": [name]})
    variants = [{key: record[key] for key in ("index", "key", "entry") if key in record}
                for record in sorted(records["variant"], key=lambda item: item["index"])]
    if not variants:
        variants = [{"key": {}, "index": 0}]
    for record in records["variant"]:
        if "entry" in record and "declaration" in record:
            signature = _signature(record)
            if re.sub(r"\s+", "", signature) not in {
                "void(constvoid*,uint32_t)", "void(constvoid*,unsigned)", "void(constvoid*,unsignedint)",
            }:
                _reject("P-SYM", f"$.stage.variants[{record['index']}].entry",
                        "variant declaration does not implement the fixed v2 entry ABI", signature=signature)
    topology = records["topology"][0] if records["topology"] else {
        "cta_group": 1, "cluster": {"compiled": [1, 1, 1], "launch": {"x_multiple_of": 1, "max": [1, 1, 1]}},
        "persistent": False, "scheduler": "static", "multicast": [],
    }
    tensor: dict[str, Any] = {"version": "tensor-v2", "exports": [], "imports": [], "sites": [],
                              "bindings": [{key: value for key, value in record.items() if key != "kind"}
                                           for record in records["binding"]]}
    for record in records["tensor_export"]:
        placement = record["placement"]
        guard = None
        if "guard" in placement:
            guard_outputs = maps.records.get(placement["guard"], {}).get("outputs", [])
            if len(guard_outputs) != 1:
                _reject("P-MAP", f"$.maps.{placement['guard']}", "guard map requires exactly one output")
            guard = ["eq", maps.get(placement["guard"], guard_outputs)[guard_outputs[0]], ["const", 1]]
        calls = record["calls"]
        endpoints = {item["name"]: item for item in records["call"]}
        for operation, endpoint in calls.items():
            if endpoint is None and operation == "release":
                continue
            if endpoint not in endpoints or endpoints[endpoint]["direction"] != "consume":
                _reject("P-TENSOR-CALLS", f"$.tensor.exports.{record['name']}.calls.{operation}",
                        "export call must name a declared provided call endpoint", endpoint=endpoint)
        tensor["exports"].append({
            **{key: record[key] for key in ("name", "identity", "read_capabilities", "readiness", "fixed_bindings", "backing", "calls")},
            "placements": [{"space": placement["space"], "units": placement["units"], "guard": guard,
                            "components": maps.get(placement["map"], placement["components"])}],
        })
    site_values = {}
    for record in records["site"]:
        participant = record["participants"]
        if participant.get("all_threads") is True:
            if record["role"] not in roles:
                _reject("P-MAP", f"$.sites.{record['name']}", "all_threads requires a declared role")
            values = set(range(roles[record["role"]]["warps"] * 32))
            participant = {"set": sorted(values)}
        else:
            name = participant["map"]
            values = maps.values(name, "participant")
            participant = {"expr": maps.get(name, ["participant"])["participant"], "domain": maps.domains[name]}
            if any(value < 0 for value in values):
                _reject("P-MAP", f"$.sites.{record['name']}", "site participant must be nonnegative")
        site_values[record["name"]] = values
        tensor["sites"].append({**{key: record[key] for key in ("name", "role", "invocation", "guarantees", "observes")},
                                "participants": participant})
    for record in records["tensor_import"]:
        for index, argument in enumerate(record["args"]):
            if argument["meaning"] == "source_base" and argument["ctype"] != "pipelink_tensor_source":
                _reject("P-TENSOR-SOURCE", f"$.tensor.imports.{record['name']}.callable.params[{index}].ctype",
                        "source_base must use pipelink_tensor_source", actual=argument["ctype"])
        rendezvous = []
        for event in record["rendezvous"]:
            cohort = event["cohort"]
            if cohort.get("all_threads") is True:
                site_roles = {site["role"] for site in records["site"] if site["name"] in record["sites"]}
                cohort = {"set": sorted({thread for role in site_roles for thread in range(roles[role]["warps"] * 32)})}
            rendezvous.append({**event, "cohort": cohort})
        demand = record["demand"]
        coords = [domain["name"] for domain in record["identity"]["coords"]]
        mapped = maps.get(demand["map"], [*coords, *( ["participant", "position"] if demand["kind"] == "positions" else ["address"])])
        view = {"kind": demand["kind"], "domain": demand["domain"], "guard": None,
                "identity_coords": {name: mapped[name] for name in coords}}
        if demand["kind"] == "positions":
            view.update({key: mapped[key] for key in ("participant", "position")})
            wanted = maps.values(demand["map"], "participant")
            for site in record["sites"]:
                if site not in site_values or not wanted <= site_values[site]:
                    _reject("P-MAP", f"$.tensor.imports.{record['name']}", "demand participants are outside the import site",
                            site=site, participants=sorted(wanted - site_values.get(site, set())))
        else:
            view.update({"space": demand["space"], "units": demand["units"], "components": {"address": mapped["address"]}})
        tensor["imports"].append({**{key: record[key] for key in ("name", "identity", "sites", "completion", "lends")},
                                  "demands": [view], "callable": {"symbol": record["symbol"],
                                  "declaration": normalize_declaration(record["declaration"]), "params": record["args"],
                                  "events": {"ready": record["symbol"] + "_ready", "done": record["symbol"] + "_done"}},
                                  "rendezvous": rendezvous})
    sections = {"$.stage": "declared", "$.stage.roles": "compiled", "$.stage.exports": "compiled",
                "$.params": "compiled", "$.claims": "declared", "$.topology": "declared", "$.endpoints": "declared",
                "$.tensor": "declared", "$.symbols": "observed", "$.payload": "compiled"}
    for name in roles:
        sections[f"$.stage.roles.{name}.warps"] = "declared"
    for index in range(len(records["claim"]), len(records["claim"]) + len(records["claim_smem"])):
        sections[f"$.claims[{index}].extent"] = "compiled"
        sections[f"$.claims[{index}].align"] = "compiled"
    for index, record in enumerate(records["tensor_export"]):
        sections[f"$.tensor.exports[{index}].placements"] = "fitted"
    for index, record in enumerate(records["tensor_import"]):
        sections[f"$.tensor.imports[{index}].demands"] = "fitted"
        sections[f"$.tensor.imports[{index}].callable"] = "compiled"
    for index, record in enumerate(records["site"]):
        sections[f"$.tensor.sites[{index}].participants"] = "fitted" if "map" in record["participants"] else "compiled"
    inputs = dict(provenance_inputs or {})
    inputs["registry"] = sha256_hex(canonical_bytes(registry))
    inputs.update({f"map:{name}": sha256_hex(canonical_bytes(record)) for name, record in maps.records.items()})
    manifest = {
        "pipelink_abi": {"schema": "3.0", "target": {"profile": profile, "toolkit": toolkit,
                                                      "lto_arch": get_profile(profile).arch.lto,
                                                      **({"math": math.to_json_obj()} if math else {})}},
        "stage": {"id": stage["id"], "version": stage["version"], "symbol_prefix": prefix,
                  "entry_abi": "v2", "entry_signature": "void(const void* params, unsigned variant)",
                  "roles": roles, "exports": exports, "variants": variants},
        "endpoints": [*[_call(record) for record in records["call"]],
                      *[{"owner_role": record["role"],
                         **{key: value for key, value in record.items() if key not in {"kind", "role"}}}
                        for record in records["endpoint"]]], "claims": claims,
        "topology": {**{key: topology[key] for key in ("cluster", "persistent", "scheduler", "multicast")},
                     "cta_group": {"fixed": topology["cta_group"]}},
        "params": {**{key: params[key] for key in ("type", "bytes", "align")},
                   "sha256_of_layout_probe": sha256_hex(canonical_bytes(params))},
        "payload": {"file": "stage.o"}, "tensor": tensor,
        "provenance": {"mechanism": "pack-cu-v1", "inputs": inputs, "sections": sections},
    }
    if records["scheduler_strategy"]:
        manifest["topology"]["scheduler_strategy"] = {
            key: records["scheduler_strategy"][0][key] for key in ("id", "inputs")
        }
    if records["descriptor"]:
        manifest["stage"]["descriptor"] = {key: records["descriptor"][0][key] for key in ("bytes", "align")}
        sections["$.stage.descriptor"] = "compiled"
    _check_init_warps(manifest)
    if records["hint"]:
        manifest["hints"] = {record["key"]: record["value"] for record in records["hint"]}
    assemble_host(records, manifest)
    if facts is not None:
        check_sources(registry, manifest, facts)
    return manifest


def check_sources(registry: Mapping[str, Any], manifest: dict[str, Any], facts: ObjectFacts) -> None:
    """Check independent compiled evidence, without revalidating fitted registry data."""
    strong = set(facts.symbols.defined_strong)
    undefined = set(facts.symbols.undefined)
    failures = []
    for record in registry["records"]:
        kind = record["kind"]
        symbol = record.get("entry") if kind == "variant" else record.get("symbol")
        defined = (kind in {"role", "export"} or kind == "call" and record["direction"] == "consume"
                   or kind == "variant" and "entry" in record)
        imported = kind == "tensor_import" or kind == "call" and record["direction"] == "produce"
        if (defined and symbol not in strong) or (imported and symbol not in undefined):
            failures.append(ClauseFailure("P-SYM", f"$.records.{record.get('name', record.get('index'))}",
                            f"symbol {symbol!r} must be {'strongly defined' if defined else 'undefined'} in the object",
                            {"symbol": symbol}))
    threads = sum(role["warps"] for role in manifest["stage"]["roles"].values()) * 32
    if facts.max_threads is not None and threads > facts.max_threads:
        failures.append(ClauseFailure("P-LAUNCH", "$.stage.roles", "role threads exceed compiled maximum",
                                      {"threads": threads, "max_threads": facts.max_threads}))
    declared = manifest["topology"]["cluster"]["compiled"]
    if facts.cluster_dim is not None and declared != "dynamic" and tuple(declared) != facts.cluster_dim:
        failures.append(ClauseFailure("P-TOPO", "$.topology.cluster.compiled", "declared and compiled cluster dimensions differ",
                                      {"declared": declared, "compiled": list(facts.cluster_dim)}))
    if failures:
        raise PackError(failures)
    manifest["symbols"] = facts.symbols.to_manifest()
