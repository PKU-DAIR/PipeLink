"""Checked Access IR to borrowed TileLang device adapters.

TileLang is isolated in the configured interpreter. Unsupported recipes, effects,
control flow and tooling are deployment outcomes: this module never changes the
semantic status of its input plan. The restricted backend does not own storage.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from itertools import product
import os
import re
import subprocess
from types import MappingProxyType
from typing import Any, Mapping

from pipec2.canonjson import canonical_bytes, sha256_hex
from pipec2.decls import declaration_digest, normalize_declaration
from pipec2.model import Solution
from . import expr, ir, machine, tables
from .contracts import TensorImport
from .plan import TensorPlan


class LoweringUnsupported(RuntimeError):
    """A backend rejection, not a semantic rejection of a MODEL_SAT plan."""

    status = "LOWERING_UNSUPPORTED"

    def __init__(self, step: str, reason: str, detail: str, **details: Any):
        self.step, self.reason, self.detail, self.details = step, reason, detail, details
        super().__init__(f"{self.status}: {step}: {reason}: {detail}")

    def to_json_obj(self) -> dict[str, Any]:
        return {"status": self.status, "step": self.step, "reason": self.reason,
                "detail": self.detail, "details": self.details}


class LoweringToolingUnavailable(LoweringUnsupported):
    """The configured backend cannot run at all, so no other candidate route can lower either."""


@dataclass(frozen=True)
class AdapterSource:
    name: str
    source: str
    defined_symbols: tuple[str, ...]
    trusted_premises: tuple[dict, ...]
    resources: dict


@dataclass(frozen=True)
class Recipe:
    name: str
    families: tuple[str, ...]
    participants: str
    charges: tuple[tuple[str, int], ...] = ()
    storage: tuple[str, ...] = ()
    synchronization: str = "none"
    extra_storage: int = 0


# IDs belong to the target catalogue; recipe names identify the evidenced route.
# There is no generic "copy anything" fallback. Scalar shared operations remain
# separate steps: this recipe cannot invent a publication/reuse rendezvous.
RECIPES = MappingProxyType({
    "tilelang.tcgen05.packet": Recipe("tcgen05_natural_receipt", ("tmem_packet_receipt",), "warp",
                                     (("handles", 1), ("outstanding", 1)), ("reg",), "adjacent_completion_wait"),
    "tilelang.private.copy": Recipe("private_permute", ("reg_copy",), "thread", storage=("reg",)),
    "tilelang.private.copy.round": Recipe("private_permute", ("reg_copy_round",), "thread", storage=("reg",)),
    "tilelang.warp.shuffle.xor": Recipe("warp_shfl_xor", ("shfl_xor",), "warp", storage=("reg",)),
    "tilelang.shared.store": Recipe("staged_shared_relay", ("smem_store_scalar",), "thread", storage=("reg", "smem")),
    "tilelang.shared.load": Recipe("staged_shared_relay", ("smem_load_scalar",), "thread", storage=("reg", "smem")),
    "tilelang.shared.store.round": Recipe("staged_shared_relay", ("smem_store_round",), "thread", storage=("reg", "smem")),
    "tilelang.shared.load.round": Recipe("staged_shared_relay", ("smem_load_round",), "thread", storage=("reg", "smem")),
    "tilelang.cta.rendezvous": Recipe("cta_rendezvous", ("cta_rendezvous",), "cta",
                                     (("barriers", 1),), ("barrier",), "fenced_rendezvous"),
    "tilelang.fenced.readiness.observe": Recipe("published_readiness", ("observe_ready",), "thread"),
})

# Per-target entries of the same evidenced routes. The fence pair brackets every CTA
# rendezvous on a target whose adapters share the body with tcgen05 traffic; a target
# without a native receipt has no async proxy to fence, so its rendezvous is the bare
# barrier. `force_include` is the header the fences need when no receipt pulled it in.
_TCGEN05_TARGET = MappingProxyType({
    "rendezvous_fences": ("tl::tcgen05_before_thread_sync", "tl::tcgen05_after_thread_sync"),
    "receipt": "tilelang.tcgen05.packet", "force_include": "tl_templates/cuda/copy_sm100.h"})
_WARP_TARGET = MappingProxyType({"rendezvous_fences": (), "receipt": None, "force_include": None})
_TARGETS = MappingProxyType({"sm_103a": _TCGEN05_TARGET, "sm_100a": _TCGEN05_TARGET,
                             "sm_90a": _WARP_TARGET, "sm_120a": _WARP_TARGET, "sm_121a": _WARP_TARGET})

_CACHE: dict[tuple[str, str], dict] = {}
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")

# Sixteen-bit floating payloads are transported as bits, never converted.
_PAYLOAD_DTYPES = MappingProxyType({
    name: (dtype, elem_bytes, ctype)
    for names, dtype, elem_bytes, ctype in (
        (("f32", "fp32", "float32"), "float32", 4, "float"),
        (("bf16", "f16", "fp16"), "uint16", 2, "uint16_t"),
        (("i32", "int32"), "int32", 4, "int32_t"),
        (("u32", "uint32"), "uint32", 4, "uint32_t"),
    )
    for name in names
})


def _reject(step: str, reason: str, detail: str, **details: Any):
    raise LoweringUnsupported(step, reason, detail, **details)


def _declaration(imp: TensorImport) -> str:
    callable = imp.callable
    if not _IDENTIFIER.fullmatch(callable.symbol) or any(not _IDENTIFIER.fullmatch(p.name) for p in callable.params):
        _reject("callable", "declaration_mismatch", "callable identifiers are not C identifiers")
    # The packed declaration is printed from the callable's type: parameter names are not part of it.
    # The generated definition names its parameters from the contract's params.
    typed = f'extern "C" __device__ void {callable.symbol}(' + ", ".join(
        param.ctype for param in callable.params) + ");"
    if normalize_declaration(typed) != normalize_declaration(callable.declaration):
        _reject("callable", "declaration_mismatch", "frozen declaration differs from its typed callable parameters",
                expected=typed, actual=callable.declaration)
    expected = f'extern "C" __device__ void {callable.symbol}(' + ", ".join(
        f"{param.ctype} {param.name}" for param in callable.params) + ");"
    meanings = [param.meaning for param in callable.params if param.meaning != "binding"]
    if len(set(meanings)) != len(meanings):
        _reject("callable", "unsupported_ir", "multiple pointer views for one callable meaning are not in the recipe scope")
    element_ctype = _PAYLOAD_DTYPES[imp.identity.dtype][2]
    pointer = f"{element_ctype}*"
    supported = {"source_base": {"pipelink_tensor_source"},
                 "dest_fragment": {pointer}, "dest_view": {"const void**"},
                 "scratch": {"void*", pointer}, "workspace": {"void*", pointer},
                 "binding": {"uint32_t", "int", "int32_t"}}
    for param in callable.params:
        if param.ctype not in supported[param.meaning]:
            _reject("callable", "unsupported_ir", f"unsupported {param.meaning} type {param.ctype!r}")
    return normalize_declaration(expected)


def _resource_bindings(plan, imp, solved, used):
    resources = {resource.name: resource for resource in plan.program.resources}
    context = plan.provenance["lowering_context"]
    native = solved.native_bases
    params = {param.meaning: param.name for param in imp.callable.params}
    destination = "dest_view" if plan.program.view_binding is not None else "dest_fragment"
    if len([p for p in imp.callable.params if p.meaning == destination]) != 1:
        _reject("callable", "unsupported_ir", f"exactly one {destination} parameter is required")
    if destination == "dest_view" and "dest_fragment" in params:
        _reject("callable", "unsupported_ir", "a memory view cannot also deliver a destination fragment")
    handles = {handle.name: handle for handle in plan.handles}
    claims = {claim.handle: claim for claim in plan.claims}
    bound = {}
    shared = [r for r in resources.values() if r.name in used and r.domain == "smem"]
    if len(shared) > 1:
        _reject("resources", "binding", "multiple scratch reservations require distinct published pointer views")
    for name in sorted(used):
        if name.startswith("@export:"):
            if "source_base" not in params:
                _reject(name, "binding", "exported storage has no source-base callable parameter")
            source = next(p for p in imp.callable.params if p.meaning == "source_base")
            descriptor = plan.provenance["lowering_context"].get("exports", {}).get(name)
            if (source.ctype != "pipelink_tensor_source"
                    or descriptor is None or descriptor["domain"] not in {"reg", "smem"}
                    or descriptor["base_convention"] != "relative_to_source_base"):
                _reject(name, "binding", "exported storage requires its declared relative scalar pointer view")
            bound[name] = {**descriptor, "param": source.name, "base": None}
            continue
        if name not in resources or name not in context["storage"]:
            _reject(name, "effect_mismatch", "recipe accesses undeclared or unpermitted storage")
        resource = resources[name]
        handle = resource.location.handle
        base = resource.location.base
        required = plan.provenance["resource_demands"][name]
        allocated = resource.extent
        if handle in handles:
            claim = claims.get(handle)
            domain = "named_barrier" if resource.domain == "barrier" else resource.domain
            if (claim is None or claim.domain != domain or claim.extent is None
                    or not required <= claim.extent.const <= resource.extent
                    or claim.align != resource.align):
                _reject(name, "effect_mismatch", "generated claim does not cover demand within declared capacity")
            if handle not in native:
                _reject(name, "binding", "generated resource has no solved native base", handle=handle)
            base = native[handle]
        elif handle in native:
            base = native[handle]
        if resource.domain in {"smem", "barrier"}:
            domain = "named_barrier" if resource.domain == "barrier" else resource.domain
            allocated = max((interval.hi - interval.lo for interval in solved.intervals.get(domain, ())
                             if interval.handle == handle), default=0)
        if base is not None and (type(base) is not int or base < 0 or base % resource.align):
            _reject(name, "binding", "solved base violates the resource alignment", base=base)
        param = None
        if resource.domain == "smem":
            param = params.get("scratch") or params.get("workspace")
            if param is None:
                _reject(name, "binding", "shared recipe must borrow caller scratch")
            if resource.binding == "relocatable" and base is None:
                _reject(name, "binding", "relocatable caller scratch has no solved base", handle=handle)
        elif resource.domain == "reg":
            if name == plan.provenance["lowering_context"]["result_resource"]:
                param = params["dest_fragment"]
        elif resource.domain == "barrier" and base is None:
            _reject(name, "binding", "rendezvous ID has no solved or fixed binding")
        bound[name] = {"domain": resource.domain, "param": param, "extent": resource.extent,
                       "declared_extent": resource.extent, "required_extent": required,
                       "allocated_extent": allocated,
                       "base": base, "handle": handle, "scope": resource.scope, "align": resource.align}
    return bound


def _receipt_address(packet, metadata, context, members, source_param):
    """Decode at a declared base, then relocate in columns, never packet indices."""
    domains = {name: expr.Domain.from_json_obj(value)
               for name, value in context["binding_domains"].items()}
    domains["participant"] = expr.Domain("participant", "int", values=tuple(members))
    base = context.get("source_base_binding")
    reference = context.get("source_base_reference", 0)
    if base in domains:
        values = domains[base].values
        reference = values[0] if values is not None else domains[base].lo

    def expand(value, substitutions):
        if isinstance(value, int):
            return ["const", value]
        if value[0] in {"bind", "coord"} and value[1] in substitutions:
            return substitutions[value[1]]
        return [value[0], *(expand(arg, substitutions) if isinstance(arg, list) else arg
                            for arg in value[1:])]

    substitutions = {}
    for name, value in metadata.items():
        substitutions[name] = expand(value, substitutions)
    selector = expr.parse(expand(packet, substitutions), domains)
    points = tables.coordinate_points(domains)
    names = sorted(domains)
    selected, checked = set(), 0
    packets = context["packets"]
    for point in points:
        environment = dict(zip(names, point))
        index = expr.evaluate(selector, environment)
        anchored = {**environment, base: reference} if base in domains else environment
        anchor = expr.evaluate(selector, anchored)
        absolute, relative = packets[index], packets[anchor]
        delta = environment.get(base, reference) - reference
        if absolute["dp"] != relative["dp"] or absolute["column"] != relative["column"] + delta:
            _reject("tmem_packet_receipt", "refinement", "packet selection is not a source-base relocation",
                    bindings=environment)
        selected.add(anchor)
        checked += 1
    constant = (["cbits", int(domains[base].sort[5:-1]), reference]
                if base in domains and domains[base].sort.startswith("bits[") else ["const", reference])
    anchored_selector = expand(selector.to_json_obj(), {base: constant} if base else {})
    indices = sorted(selected)
    def offset(index):
        entry = packets[index]
        return ["const", (entry["dp"] << 16) + entry["column"] - reference]
    address = offset(indices[-1])
    for index in reversed(indices[:-1]):
        address = ["select", ["eq", anchored_selector, ["const", index]], offset(index), address]
    address = ["add", ["toint", ["bind", source_param]], address]
    return address, {
        "source_base_binding": base, "source_param": source_param, "reference": reference,
        "source_base_values": list(domains[base].values if domains[base].values is not None
                                  else range(domains[base].lo, domains[base].hi)) if base in domains else [reference],
        "binding_domains": context["binding_domains"], "participants": members,
        "equality_rows": checked,
        "outside_declared_worlds": "full column-base relocation; not model-certified",
    }


def _prepare(plan: TensorPlan, imp: TensorImport, solved: Solution, profile) -> dict:
    if plan.status != "MODEL_SAT" or plan.program is None or plan.ir_digest != ir.digest(plan.program):
        _reject("plan", "unsupported_ir", "lowering requires an unchanged MODEL_SAT checked program")
    target = _TARGETS.get(getattr(profile, "name", None))
    if target is None:
        _reject("profile", "unsupported_ir", f"no evidenced TileLang recipes for target {getattr(profile, 'name', None)!r}")
    payload = _PAYLOAD_DTYPES.get(imp.identity.dtype)
    if payload is None:
        _reject("plan", "unsupported_ir", f"unsupported payload dtype {imp.identity.dtype!r}")
    dtype, elem_bytes, element_ctype = payload
    catalogue = machine.CATALOGUE(profile)
    used: set[str] = set()
    recipes = {}
    context = plan.provenance.get("lowering_context")
    if not isinstance(context, dict):
        _reject("plan", "unsupported_ir", "checked packet/site lowering context is absent")
    view_binding = plan.program.view_binding
    view_demands = [demand for demand in imp.demands if demand.kind == "memory_view"]
    if view_demands:
        if len(view_demands) != len(imp.demands) or any(
                demand.space != "smem" or demand.units != "byte" for demand in view_demands):
            _reject("plan", "unsupported_ir", "view lowering requires only shared-memory byte-address demands")
        if (not isinstance(view_binding, str) or not view_binding
                or context.get("view_binding") != view_binding
                or plan.provenance.get("view_binding") != view_binding):
            _reject("plan", "binding", "checked view binding differs from its lowering context")
        if view_binding != "@export:smem" and view_binding not in context["storage"]:
            _reject("plan", "binding", "view must bind exported shared memory or declared import storage")
    elif view_binding is not None:
        _reject("plan", "binding", "positions lowering cannot bind a delivered memory view")
    resources = {resource.name: {"domain": resource.domain, "extent": resource.extent}
                 for resource in plan.program.resources}
    resources.update(context.get("exports", {}))
    readonly = frozenset(context["readonly_storage"])
    for site, graph in sorted(plan.program.graphs.items()):
        for index, node in enumerate(graph.nodes):
            if not isinstance(node, ir.Issue):
                continue
            step = f"{site}:{index}"
            if node.primitive not in context["permitted"].get(site, ()):
                _reject(step, "effect_mismatch", "primitive is not permitted by the admitted lends and guarantees")
            if node.primitive != "observe_ready":
                destination = node.operands.get("dst", node.operands.get("dest"))
                if isinstance(destination, Mapping):
                    destination = destination.get("resource")
                if isinstance(destination, str) and (destination in readonly or destination.startswith("@export:")):
                    _reject(step, "effect_mismatch", "recipe writes read-only exported storage")
            primitive = catalogue.get(node.primitive)
            recipe = RECIPES.get(primitive.lowering_recipe) if primitive else None
            if recipe is None:
                _reject(step, "no_recipe", "planned primitive has no installed TileLang recipe", primitive=node.primitive)
            if (node.primitive not in recipe.families or primitive.participants != recipe.participants
                    or dict(primitive.charges) != dict(recipe.charges) or recipe.extra_storage != 0):
                _reject(step, "effect_mismatch", "catalogue effects differ from the pinned lowering recipe")
            recipes[step] = recipe.name
            for operand in node.operands.values():
                if isinstance(operand, Mapping) and "resource" in operand:
                    used.add(operand["resource"])
            if node.primitive in {"reg_copy_round", "smem_store_round", "smem_load_round"}:
                used.update((node.operands["src"], node.operands["dst"]))
                source, destination = resources.get(node.operands["src"]), resources.get(node.operands["dst"])
                domains = {"reg_copy_round": ("reg", "reg"), "smem_store_round": ("reg", "smem"),
                           "smem_load_round": ("smem", "reg")}[node.primitive]
                if source is None or destination is None or (source["domain"], destination["domain"]) != domains:
                    _reject(step, "effect_mismatch", "round operands do not match the declared scalar recipe domains")
                width, offset = node.operands.get("width"), node.operands.get("offset", 0)
                private = source if node.primitive == "smem_store_round" else destination
                if (type(width) is not int or type(offset) is not int or width <= 0 or offset < 0
                        or offset + width > private["extent"]):
                    _reject(step, "effect_mismatch", "round exceeds its declared private view")
                if node.primitive == "reg_copy_round":
                    src_offset = node.operands.get("src_offset", 0)
                    if type(src_offset) is not int or src_offset < 0 or src_offset + width > source["extent"]:
                        _reject(step, "effect_mismatch", "private round exceeds its declared source view")
                elif "address" not in node.operands:
                    _reject(step, "effect_mismatch", "shared round omits its shared address")
            if node.primitive == "tmem_packet_receipt" and node.operands.get("phase", "issue") == "issue":
                used.add(node.operands["dest"])
                wait = graph.nodes[node.next]
                if (not isinstance(wait, ir.Issue) or wait.primitive != node.primitive
                        or wait.operands.get("phase") != "wait"
                        or wait.operands.get("handle", "receipt") != node.operands.get("handle", "receipt")):
                    _reject(step, "refinement", "synchronous TileLang receipt requires the immediately following matching completion wait")
            if node.primitive == "cta_rendezvous":
                used.add(node.operands["resource"])
    used.add(view_binding if view_binding is not None else context["result_resource"])
    bound = _resource_bindings(plan, imp, solved, used)
    if view_binding is not None and (bound[view_binding]["domain"] != "smem" or bound[view_binding]["param"] is None):
        _reject(view_binding, "binding", "delivered view must borrow a shared-memory pointer")
    sites = context["participants"]
    if set(sites) != set(plan.program.graphs) or any(site not in imp.sites for site in sites):
        _reject("sites", "effect_mismatch", "controller participants differ from the admitted import sites")
    all_members = [member for members in sites.values() for member in members]
    if not all_members or len(set(all_members)) != len(all_members):
        _reject("sites", "unsupported_ir", "overlapping/empty physical invocation sites need a site selector")
    if min(all_members) < 0 or max(all_members) >= solved.launch.threads:
        _reject("sites", "effect_mismatch", "import participants lie outside the solved launch")
    for site, graph in plan.program.graphs.items():
        members = sites[site]
        for index, node in enumerate(graph.nodes):
            if isinstance(node, ir.Issue) and catalogue[node.primitive].participants == "warp":
                if context["warp_size"] != 32 or any(set(range((m // 32) * 32, (m // 32 + 1) * 32)) - set(members) for m in members):
                    _reject(f"{site}:{index}", "effect_mismatch", "native warp recipe requires the exact full physical warps")
            if isinstance(node, ir.Issue) and node.primitive == "tmem_packet_receipt":
                if context["packet_columns"] != 16 or context["scalar_bytes"] != 4:
                    _reject(f"{site}:{index}", "refinement", "native receipt must be exactly 32dp32b16x")
                source = [param for param in imp.callable.params if param.meaning == "source_base"]
                if len(source) != 1 or source[0].ctype != "pipelink_tensor_source":
                    _reject(f"{site}:{index}", "binding", "TMEM receipt requires the opaque source handle")
            if isinstance(node, ir.Issue) and node.primitive == "cta_rendezvous":
                if len(members) % 32:
                    _reject(f"{site}:{index}", "effect_mismatch", "native rendezvous requires full physical warps")
                if bound[node.operands["resource"]]["base"] == 0 and set(members) != set(range(solved.launch.threads)):
                    _reject(f"{site}:{index}", "effect_mismatch", "reserved barrier zero requires every solved launch participant")
    # A readiness event is not a callable runtime flag. Only an observation
    # guaranteed before entry permits folding it in the frozen call.
    ready = context["ready_at_entry"]
    graphs = {}
    for site, graph in sorted(plan.program.graphs.items()):
        active = set()
        def walk(index, metadata):
            if index in active:
                _reject(f"{site}:{index}", "unsupported_ir", "cyclic control requires a published runtime observation recipe")
            node = graph.nodes[index]
            active.add(index)
            try:
                if isinstance(node, ir.Return):
                    return []
                if isinstance(node, ir.Branch):
                    if node.observation in {"ready", context.get("readiness")} and ready:
                        return walk(node.next_true, metadata)
                    predicate = metadata.get(node.observation)
                    if predicate is not None and machine.is_static_expression(predicate):
                        parsed = expr.parse(predicate, {"participant": expr.Domain("participant", "int", values=tuple(sites[site]))})
                        if parsed.sort != "pred":
                            integer = ["toint", predicate] if parsed.sort.startswith("bits[") else predicate
                            predicate = ["not", ["eq", integer, ["const", 0]]]
                        taken, skipped = walk(node.next_true, metadata), walk(node.next_false, metadata)
                        shared = 0
                        while shared < min(len(taken), len(skipped)) and taken[-1 - shared] == skipped[-1 - shared]:
                            shared += 1
                        common = taken[len(taken) - shared:] if shared else []

                        def guarded(steps, condition):
                            if any(step["primitive"] == "cta_rendezvous" for step in steps):
                                _reject(f"{site}:{index}", "effect_mismatch", "static guards cannot enclose the rendezvous")
                            return [{**step, "guard": ["pand", condition, step["guard"]] if "guard" in step else condition}
                                    for step in steps]

                        return (guarded(taken[:len(taken) - shared], predicate)
                                + guarded(skipped[:len(skipped) - shared], ["not", predicate]) + common)
                    _reject(f"{site}:{index}", "unsupported_ir", "runtime branch observation has no borrowed binding")
                if isinstance(node, ir.Meta):
                    return walk(node.next, {**metadata, node.dest: node.expr})
                if node.primitive == "observe_ready":
                    if not ready:
                        _reject(f"{site}:{index}", "unsupported_ir", "readiness is not established before the frozen call")
                    return walk(node.next, {**metadata, node.operands["dest"]: 1})
                following = node.next
                if node.primitive == "tmem_packet_receipt":
                    if node.operands.get("phase", "issue") == "wait":
                        _reject(f"{site}:{index}", "refinement", "completion wait reached without its adjacent issue")
                    following = graph.nodes[node.next].next
                return [{"step": f"{site}:{index}", "primitive": node.primitive,
                         "operands": dict(node.operands), "metadata": metadata}] + walk(following, metadata)
            finally:
                active.remove(index)
        graphs[site] = walk(graph.entry, {})
    receipt_relocations = {}
    for site, steps in graphs.items():
        for step in steps:
            if step["primitive"] == "tmem_packet_receipt":
                source = next(param.name for param in imp.callable.params if param.meaning == "source_base")
                address, proof = _receipt_address(step["operands"]["packet"], step["metadata"], context, sites[site], source)
                step["operands"]["address"] = address
                receipt_relocations[step["step"]] = proof
    declaration = _declaration(imp)
    request = {"plan_digest": plan.ir_digest, "contract_digest": sha256_hex(canonical_bytes(imp.to_json_obj())),
            "declaration": declaration, "params": [p.to_json_obj() for p in imp.callable.params],
            "dtype": dtype, "elem_bytes": elem_bytes, "element_ctype": element_ctype,
            # The pinned receipt template is a 32-bit bit-transport projection.
            "receipt_dtype": _PAYLOAD_DTYPES["f32"][0],
            "context": context, "borrows": dict(plan.program.borrows), "resources": bound,
            "receipt_relocations": receipt_relocations,
            "graphs": graphs, "recipes": recipes, "threads": max(all_members) + 1,
            "symbol": imp.callable.symbol, "target": profile.name, "launch_threads": solved.launch.threads,
            "target_entry": {"rendezvous_fences": list(target["rendezvous_fences"]),
                             "receipt": target["receipt"], "force_include": target["force_include"]}}
    if view_binding is not None:
        request["view_shell"] = {
            "symbol": imp.callable.symbol, "declaration": declaration,
            "view_binding": view_binding, "binding": bound[view_binding],
            "dest_param": next(p.name for p in imp.callable.params if p.meaning == "dest_view"),
            "bound_param": bound[view_binding]["param"],
        }
        if view_binding == "@export:smem":
            if any(graphs.values()):
                _reject("plan", "effect_mismatch", "an alias shell requires a zero-movement checked program")
        else:
            request["symbol"] = imp.callable.symbol + "__fill"
            request["params"] = [p for p in request["params"] if p["meaning"] != "dest_view"]
            request["declaration"] = normalize_declaration(
                f'static __device__ void {request["symbol"]}('
                + ", ".join(f'{p["ctype"]} {p["name"]}' for p in request["params"]) + ");")
    return request


def _backend(request: dict) -> dict:
    interpreter = os.environ.get("PIPELINK_TILELANG_PYTHON")
    if not interpreter:
        raise LoweringToolingUnavailable("tilelang", "tooling", "PIPELINK_TILELANG_PYTHON is not configured",
                                         kind="missing_interpreter")
    key = (interpreter, sha256_hex(canonical_bytes(request)))
    if key in _CACHE:
        return _CACHE[key]
    try:
        result = subprocess.run([interpreter, "-m", "pipelink_tilelang.driver"], input=canonical_bytes(request).decode(),
                                capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                                     "CUDA_VISIBLE_DEVICES": ""})
    except OSError as exc:
        raise LoweringToolingUnavailable("tilelang", "tooling", str(exc), kind=type(exc).__name__,
                                         interpreter=interpreter) from exc
    if result.returncode:
        _reject("tilelang", "tooling", "configured TileLang backend failed", returncode=result.returncode,
                stderr=result.stderr, stdout=result.stdout, interpreter=interpreter)
    try:
        output = json.loads(result.stdout)
        valid = (output["tilelang_version"] == "0.1.14" and output["projection"]["inverse_structurally_equal"] is True
                 and isinstance(output["source"], str))
    except (ValueError, KeyError, TypeError) as exc:
        _reject("tilelang", "tooling", "malformed TileLang backend output", error=str(exc), stdout=result.stdout)
    if not valid:
        _reject("tilelang", "refinement", "pinned TileLang or inverse storage-view check did not pass")
    _CACHE[key] = output
    return output


_HEADER = re.compile(r'^extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+)?([A-Za-z_]\w*)\s*\(([^()]*)\)\s*([;{])', re.MULTILINE)


def _legalize(source: str, request: dict) -> tuple[str, str]:
    matches = list(_HEADER.finditer(source))
    if (len(matches) != 2 or [m[5] for m in matches] != [";", "{"]
            or matches[0].groups()[2:4] != matches[1].groups()[2:4]):
        _reject("entry", "declaration_mismatch", "backend must emit exactly one matching kernel prototype and definition")
    definition = matches[1]
    if definition[1] != str(request["threads"]) or definition[2] != "1":
        _reject("entry", "effect_mismatch", "backend launch participants differ from the checked site")
    generated = [re.sub(r"\s*__restrict__\s*", " ", p).strip() for p in definition[4].split(",")]
    expected = request["params"]
    if len(generated) != len(expected):
        _reject("entry", "declaration_mismatch", "backend parameter count differs from frozen callable")
    for actual, param in zip(generated, expected):
        ctype = param["ctype"]
        if param["meaning"] == "source_base":
            ctype = "uint32_t" if request["context"]["source_space"] == "tmem" else f'const {request["element_ctype"]}*'
        wanted = ctype + " " + param["name"]
        def compact(text):
            text = re.sub(r"\b(?:unsigned int|uint)\b", "uint32_t", text)
            text = re.sub(r"\b(?:unsigned short|ushort)\b", "uint16_t", text)
            text = re.sub(r"\bint32_t\b", "int", text)
            return re.sub(r"\s+", "", text)
        if compact(actual) != compact(wanted):
            _reject("entry", "declaration_mismatch", "backend parameter view differs from frozen callable", expected=wanted, actual=actual)
    start = definition.end() - 1
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', lambda m: " " * len(m.group()), source, flags=re.DOTALL)
    depth, end = 0, None
    for offset in range(start, len(masked)):
        depth += (masked[offset] == "{") - (masked[offset] == "}")
        if depth == 0:
            end = offset + 1
            break
    if end is None or masked[end:].strip():
        _reject("entry", "refinement", "unexpected extra source after the backend entry")
    body = source[start:end]
    forbidden = r"\b(?:tmem_allocate|tmem_deallocate|malloc|free|__shared__|blockIdx|blockDim|gridDim)\b|\b(?:ld|st)\.global|tcgen05\.(?:alloc|dealloc|st)"
    if re.search(forbidden, body):
        _reject("entry", "effect_mismatch", "backend introduced owning storage or global/launch effects")
    if (len(re.findall(r"\b__global__\b", masked)) != 2 or re.search(r"\b__device__\b", masked)
            or re.search(r"[{}]", masked[:definition.start()])):
        _reject("entry", "declaration_mismatch", "backend introduced an additional callable definition")
    permitted_arrays = {f"pl_private_{index}": resource["extent"]
                        for index, (_, resource) in enumerate(sorted(request["resources"].items()))
                        if resource["domain"] == "reg" and resource["param"] is None}
    for name, extent in re.findall(r"\b(?:float|double|ushort|unsigned\s+short|u?int(?:16_t|32_t|64_t)?)\s+(\w+)\s*\[\s*(\d+)\s*\]", body):
        if permitted_arrays.get(name) != int(extent):
            _reject("entry", "effect_mismatch", "backend introduced undeclared private payload storage", name=name, extent=int(extent))
    steps = [step for graph in request["graphs"].values() for step in graph]
    target = request["target_entry"]
    receipts = sum(step["primitive"] == "tmem_packet_receipt" for step in steps)
    if receipts and target["receipt"] is None:
        _reject(next(step["step"] for step in steps if step["primitive"] == "tmem_packet_receipt"), "no_recipe",
                "target has no native receipt recipe", primitive="tmem_packet_receipt")
    receipt_pattern = r"\btl::tcgen05_ld_32dp32bNx\s*<\s*16\s*,\s*false\s*>\s*\("
    if len(re.findall(receipt_pattern, body)) != receipts or len(re.findall(r"\btcgen05_ld_", body)) != receipts:
        _reject("entry", "refinement", "backend receipt family/width/count differs from planned 32dp32b16x receipts")
    shuffle_pattern = r"\b__shfl_xor_sync\s*\("
    if len(re.findall(shuffle_pattern, body)) != sum(step["primitive"] == "shfl_xor" for step in steps):
        _reject("entry", "refinement", "backend shuffle realization differs from the planned exchange count")
    barriers = sum(step["primitive"] == "cta_rendezvous" for step in steps)
    sync_pattern = r"\b(?:__syncthreads|tl::syncthreads)\s*\(|bar\.sync"
    sync = list(re.finditer(sync_pattern, body))
    fences = target["rendezvous_fences"]
    if fences:
        before_pattern = rf"\b{re.escape(fences[0])}\s*\(\)"
        before = list(re.finditer(before_pattern, body))
        after = list(re.finditer(rf"\b{re.escape(fences[1])}\s*\(\)", body))
        if len(before) != barriers or len(after) != barriers or len(sync) != barriers or any(
                not a.start() < b.start() < c.start() for a, b, c in zip(before, sync, after)):
            _reject("entry", "effect_mismatch", "backend synchronization does not match the explicit fenced rendezvous steps")
        rendezvous_pattern = before_pattern
    else:
        if len(sync) != barriers or re.search(r"tcgen05_", body):
            _reject("entry", "effect_mismatch", "backend synchronization does not match the bare rendezvous steps of an unfenced target")
        rendezvous_pattern = sync_pattern
    patterns = {"tmem_packet_receipt": receipt_pattern, "shfl_xor": shuffle_pattern, "cta_rendezvous": rendezvous_pattern}
    effects = sorted((match.start(), name) for name, pattern in patterns.items() for match in re.finditer(pattern, body))
    if [name for _, name in effects] != [step["primitive"] for step in steps if step["primitive"] in patterns]:
        _reject("entry", "refinement", "backend changed the checked native receipt/shuffle/rendezvous order")
    opaque = next((param for param in request["params"] if param["meaning"] == "source_base"), None)
    declaration = request["declaration"]
    prologue = ""
    if opaque is not None:
        name = opaque["name"]
        handle = name + "_pl_source"
        declaration = declaration.replace(f"pipelink_tensor_source {name}", f"pipelink_tensor_source {handle}")
        if request["context"]["source_space"] == "tmem":
            prologue = f"  uint32_t {name} = pl::source_tmem({handle});\n"
        else:
            ctype = request["element_ctype"]
            prologue = f"  const {ctype}* {name} = pl::source_pointer<const {ctype}>({handle});\n"
    for match in reversed(matches):
        replacement = declaration[:-1] + (";" if match[5] == ";" else " {\n" + prologue)
        source = source[:match.start()] + replacement + source[match.end() - (match[5] == "{"):]
    if not source.rstrip().endswith(body):
        _reject("entry", "refinement", "entry-linkage rewrite changed backend body bytes")
    # The original TIR body is a nested scope, byte-identical after the handle prologue.
    source = '#include <pipelink/tensor_source.h>\n' + source.rstrip() + "\n}\n"
    # TileLang emits this header unconditionally, but its debug helpers include
    # strong device definitions that cannot be shared by independent adapter TUs.
    if not re.search(r"\b(?:debug_print_\w*|device_assert\w*)\b", masked[start:end]):
        body_offset = source.rfind(body)
        prefix = re.sub(r'^[ \t]*#[ \t]*include[ \t]*[<"]tl_templates/cuda/debug\.h[>"][ \t]*\r?\n',
                        "", source[:body_offset], flags=re.MULTILINE)
        source = prefix + source[body_offset:]
    return source, sha256_hex(body.encode())


def _event_sources(plan: TensorPlan, imp: TensorImport, solved: Solution, source_space: str,
                   *, witness_events: frozenset[str] = frozenset()) -> str:
    lends = {lend.name: lend for lend in imp.lends}
    events = {event.event: event for event in imp.rendezvous}
    context = plan.provenance["lowering_context"]
    receipt = any(isinstance(node, ir.Issue) and node.primitive == "tmem_packet_receipt"
                  for graph in plan.program.graphs.values() for node in graph.nodes)
    source = "\n"
    for kind, event_name in (("ready", context["readiness"]), ("done", imp.completion)):
        if event_name not in events:
            _reject(kind, "binding", "outer event has no declared rendezvous", event=event_name)
        event = events[event_name]
        lend = lends[event.barrier]
        cohort = event.cohort
        if cohort.set is None:
            domains = {domain.name: domain for domain in cohort.domain}
            members = {expr.evaluate(cohort.expr, dict(zip(domains, values)))
                       for values in product(*(domain.values if domain.values is not None else range(domain.lo, domain.hi)
                                               for domain in domains.values()))}
        else:
            members = set(cohort.set)
        if not members or not members <= set(range(solved.launch.threads)):
            _reject(kind, "effect_mismatch", "outer rendezvous cohort lies outside the launch")
        if lend.location.base == 0:
            if members != set(range(solved.launch.threads)):
                _reject(kind, "effect_mismatch", "barrier zero requires the entire launch cohort")
            barrier = "  __syncthreads();\n"
        else:
            handle = lend.location.handle
            if lend.binding == "relocatable":
                if not handle or handle not in solved.native_bases:
                    _reject(kind, "binding", "outer barrier handle has no solved binding")
                source += f'extern "C" __device__ int {handle}();\n'
                operand = f"{handle}()"
            else:
                base = lend.location.base if lend.location.base is not None else solved.native_bases.get(handle)
                if base is None:
                    _reject(kind, "binding", "outer barrier has no solved binding")
                operand = str(base)
            # bar.sync is warp-aligned: every lane of a participating warp must arrive.
            if any(not set(range(warp * 32, warp * 32 + 32)) <= members for warp in {t // 32 for t in members}):
                _reject(kind, "effect_mismatch", "named barrier cohort must be a union of complete warps")
            barrier = f'  asm volatile("bar.sync %0, %1;" :: "r"({operand}), "r"({len(members)}) : "memory");\n'
        source += f'extern "C" __device__ void {imp.callable.events[kind]}() {{\n'
        if kind == "done" and receipt:
            source += '  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");\n'
        if source_space == "tmem":
            source += '  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");\n'
        if event_name in witness_events:
            source += '  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n'
        source += barrier
        if source_space == "tmem":
            source += '  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");\n'
        source += "}\n"
    return source


def emit_adapter(plan: TensorPlan, imp: TensorImport, solved: Solution, profile,
                 *, witness_events: frozenset[str] = frozenset()) -> AdapterSource:
    """Emit one exact frozen callable using solved bindings, or raise LoweringUnsupported.

    resources is JSON-able deployment evidence, not a native register estimate.
    Trusted premises explicitly retain unperformed identity/native checks.
    """
    request = _prepare(plan, imp, solved, profile)
    view_shell = request.get("view_shell")
    alias = view_shell is not None and view_shell["view_binding"] == "@export:smem"
    bindings_digest = sha256_hex(canonical_bytes(request["resources"]))
    provenance = (f"// PipeLink tensor adapter: plan_sha256={plan.ir_digest}\n"
                  f"// contract_sha256={request['contract_digest']} bindings_sha256={bindings_digest}\n")
    source, body_digest = "#include <pipelink/tensor_source.h>\n" if alias else "", None
    premises = []
    if not alias:
        output = _backend(request)
        source, body_digest = _legalize(output["source"], request)
        provenance += f"// tilelang=0.1.14 body_sha256={body_digest}\n"
        checks = ["structural_inverse_equality", "byte_identical_lowered_body", "identity_import_differential"]
        premises.extend((
            {"name": "entry_linkage_rewrite", "scope": request["symbol"],
             "justification": "pipec README: tensor imports, TileLang backend; device entry linkage preserves the body",
             "deployment_checks": checks, "body_byte_identical": True, "body_digest": body_digest},
            {"name": "borrowed_base_binding", "scope": {"symbol": request["symbol"], "resources": request["resources"]},
             "justification": "pipec README: tensor imports, TileLang backend; borrowed resources bind the adapter's private row",
             "deployment_checks": checks, "projection": output["projection"],
             "not_established": ["identity_import_differential", "native_resources", "native_instruction_retention"]},
        ))
    public_declaration = request["declaration"]
    if view_shell is not None:
        public_declaration = view_shell["declaration"]
        fill = None
        shell = public_declaration[:-1] + " {\n"
        checks = ["frozen_declaration_match", "plan_bound_view_assignment", "identity_import_differential"]
        if not alias:
            arguments = [param["name"] for param in request["params"]]
            shell += f'  {request["symbol"]}({", ".join(arguments)});\n'
            fill = {"symbol": request["symbol"], "declaration": request["declaration"],
                    "arguments": arguments, "body_digest": body_digest}
            checks.append("fill_before_view_assignment")
        bound_param = view_shell["bound_param"]
        if alias:
            bound_param = f"pl::source_pointer<const {request['element_ctype']}>({bound_param})"
        shell += f'  *{view_shell["dest_param"]} = {bound_param};\n' + "}\n"
        shell_digest = sha256_hex(shell.encode())
        premises.append({
            "name": "view_shell_emission", "scope": imp.callable.symbol,
            "justification": "pipec README: tensor imports; emit a view shell for the declared callable",
            "deployment_checks": checks, **view_shell,
            "declaration_digest": imp.callable.declaration_digest,
            "shell_digest": shell_digest, "fill": fill,
            "not_established": ["identity_import_differential", "native_resources", "native_instruction_retention"],
        })
        provenance += f"// view_shell_sha256={shell_digest}\n"
        source += ("\n" if source else "") + public_declaration + "\n" + shell
    source += _event_sources(plan, imp, solved, request["context"]["source_space"],
                             witness_events=witness_events)
    return AdapterSource(f"tensor_{plan.ir_digest[:16]}", provenance + source,
                         (imp.callable.symbol, *imp.callable.events.values()), tuple(premises),
                         {"bindings": request["resources"], "bindings_digest": bindings_digest,
                          "recipes": request["recipes"], "declaration_digest": imp.callable.declaration_digest,
                          "receipt_relocations": request["receipt_relocations"],
                          "extra_storage": 0, "native_resources_checked": False})
