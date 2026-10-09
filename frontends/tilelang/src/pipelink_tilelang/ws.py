"""WS stage annotations and structural role splitting from captured TIR."""
import json
from math import prod
from pathlib import Path

from tilelang import tvm

from .annotate import AnnotationError, FactsHeader, Ledger, fact, symbol_prefix
from .externalize import split_roles
from .layout import from_captured


def _require(condition, field, detail):
    if not condition:
        raise AnnotationError(field, detail)


def _allocation(facts, window, name):
    path = f"/captures/{window}/allocations"
    matches = [(index, item) for index, item in enumerate(fact(facts, path)) if item["name"] == name]
    _require(len(matches) == 1, path, f"expected one allocation {name}")
    index, value = matches[0]
    return value, f"{path}/{index}"


def geometry(facts, plane_names):
    allocations = {name: _allocation(facts, "post_layout", name)[0] for name in plane_names}
    depth = next(iter(allocations.values()))["shape"][0]
    planes = {}
    for name, allocation in allocations.items():
        _require(allocation["shape"][0] == depth, "allocations", "physical plane ring depths differ")
        _require(allocation["dtype"] == "bfloat16", "allocations", f"unsupported physical plane type for {name}")
        planes[name] = {"bytes": allocation["bytes"], "stride": allocation["bytes"] // depth,
                        "elements": prod(allocation["shape"][1:])}
    strides = [plane["stride"] for plane in planes.values()]
    branches = fact(facts, "/branches")
    _require(len(branches) == 2, "branches", "expected two WS branches")
    init = fact(facts, "/captures/pre_codegen/barrier_init")
    _require(len(init) == 2 * depth, "barrier_init", "barrier slots differ from ring depth")
    schedule = fact(facts, "/barrier_schedule")
    by_role = {role: [item for item in schedule if item["role"] == role] for role in ("producer", "consumer")}
    for role, base in (("producer", 0), ("consumer", depth)):
        rows = by_role[role]
        _require(bool(rows), "barrier_schedule", f"missing {role} uses")
        for item in rows:
            _require(item["barrier"] == base + item["iteration"] % depth,
                     "barrier_schedule", "unexpected ring-slot progression")
            _require(init[item["barrier"]] == [item["barrier"], item["arrivals"]],
                     "barrier_schedule", "arrival sites disagree with initialization")
            _require(item["transaction_bytes"] == (sum(strides) if role == "producer" else 0),
                     "barrier_schedule", "transaction total disagrees with physical planes")
    tx = [call["args"][-1] for call in branches[0]["calls"]
          if call["op"] in ("tl.mbarrier_expect_tx", "tirx.ptx_arrive_barrier_expect_tx")]
    _require(tx == strides, "branches/0/calls", "expect-tx sites must match mapped plane strides in TIR order")
    return {"depth": depth, "planes": planes,
            "threads": fact(facts, "/captures/pre_codegen/attrs/thread_extent/threadIdx.x"),
            "producer_threads": branches[0]["thread_count"],
            "offsets": fact(facts, "/captures/pre_codegen/offsets"),
            "registers": {branch: branches[index]["set_max_nreg"] for index, branch in enumerate(("then", "else"))},
            "barrier_init": init, "transaction_sites": tx}


def emit_annotations(stage, facts, ledger, *, planes, smem_handles, full_handle, empty_handle):
    prefix = symbol_prefix(stage)
    namespace = prefix + "facts"
    _require(len(stage.roles) == 1, "roles", "WS stages contain exactly one branch")
    role, declaration = next(iter(stage.roles.items()))
    branch = {"then": 0, "else": 1}[declaration.branch]
    depth = max(stage.variants.values)
    root = f"/variants/{depth}"
    captured = fact(facts, root)
    first_plane = planes[stage.endpoints[0].name]
    _, allocation_path = _allocation(captured, "post_layout", first_plane)
    geom = geometry(captured, planes.values())

    def ref(field, path):
        return ledger.reference(field, facts, root + path, namespace)

    def expr(field, expression, *, kind="reflected"):
        ledger.record(field, expression, kind=kind)
        return ledger.require(field)

    def convention(field, name):
        return expr(field, prefix + "conventions::" + name, kind="convention")

    warps = ref(f"roles.{role}.warps", f"/branches/{branch}/warps")
    base = ref("claims.warp.absolute_base", f"/branches/{branch}/warp_base")
    budget = ref("claims.regs.values", f"/branches/{branch}/set_max_nreg/0")
    ring_depth = ref("ring.depth", allocation_path + "/shape/0")
    zero = convention("claims.extent.const", "RingFixedExtent")
    unit = convention("claims.mbarrier.extent.per", "BarrierSlotUnit")
    empty_index = next(index for index, item in enumerate(captured["barrier_schedule"]) if item["role"] == "consumer")
    empty_tx = ref("protocol.empty.tx_bytes", f"/barrier_schedule/{empty_index}/transaction_bytes")
    full_count = ref("protocol.full.arrive.count", "/captures/pre_codegen/barrier_init/0/1")
    empty_count = ref("protocol.empty.arrive.count", f"/captures/pre_codegen/barrier_init/{geom['depth']}/1")
    warp_alignment = expr("claims.warp.align", prefix + "conventions::WarpAllocationAlignment", kind="declared")
    for name in ("bytes", "align"):
        expr("params." + name, f"{'sizeof' if name == 'bytes' else 'alignof'}({stage.params.name})")
    for field in stage.params.fields:
        expr(f"params.fields.{field.name}.offset", f"offsetof({stage.params.name}, {field.name})")
        expr(f"params.fields.{field.name}.bytes", f"sizeof((({stage.params.name}*)nullptr)->{field.name})")
        expr(f"params.fields.{field.name}.align", f"__alignof__((({stage.params.name}*)nullptr)->{field.name})")
    lines = [f'''// Adapter allocation/coordinate conventions, not TIR measurements.
namespace {prefix}conventions {{
// Ring extents have no depth-independent allocation; handles count logical slots.
inline constexpr int RingFixedExtent = 0;
inline constexpr int BarrierSlotUnit = 1;
// Authored allocation choice: role warp intervals start on a warpgroup boundary
// (the granularity setmaxnreg register reallocation operates on).
inline constexpr int WarpAllocationAlignment = 4;
}}''',
             f'PL_STAGE("{stage.stage_id}", "{stage.version}");',
             f'PL_PARAMS({stage.params.name}, {", ".join(field.name for field in stage.params.fields)});',
             f'PL_ROLE({role}, {prefix}role_{role}, {warps});',
             f'PL_CLAIM_REGS("pipelink_tilelang_regs_{role}", "{role}", {{{budget}}}, "{role}");']
    strides = {}
    for endpoint in stage.endpoints:
        name, handle = planes[endpoint.name], smem_handles[endpoint.name]
        _, plane_path = _allocation(captured, "post_layout", name)
        ring_bytes = ref(f"claims.{handle}.smem.ring_bytes", plane_path + "/bytes")
        stride = expr(f"ring.{handle}.stage_stride_bytes", f"({ring_bytes} / {ring_depth})", kind="captured")
        strides[handle] = stride
        alignment = ref(f"claims.{handle}.align", f"/captures/pre_codegen/attrs/tl.smem_alignment_map/{name}")
        ledger.alias(f"claims.{handle}.extent.const", "claims.extent.const")
        ledger.alias(f"claims.{handle}.extent.per", f"ring.{handle}.stage_stride_bytes")
        lines.append(f'PL_CLAIM_EX(pl::claim("{handle}", "smem").extent({zero}, {{{{"{stage.variants.depth_handle}", {stride}}}}}).align({alignment}));')
    for handle in (full_handle, empty_handle):
        ledger.alias(f"claims.{handle}.extent.const", "claims.extent.const")
        ledger.alias(f"claims.{handle}.extent.per", "claims.mbarrier.extent.per")
        ledger.alias(f"claims.{handle}.align", "claims.mbarrier.extent.per")
        lines.append(f'PL_CLAIM_EX(pl::claim("{handle}", "mbarrier").extent({zero}, {{{{"{stage.variants.depth_handle}", {unit}}}}}).align({unit}));')
    ledger.alias("claims.warp.extent", f"roles.{role}.warps")
    lines.append(f'PL_CLAIM_EX(pl::claim("pipelink_tilelang_warp_{role}", "warp").extent({warps}).align({warp_alignment}).at({base}).roles({{"{role}"}}));')
    for index, value in enumerate(stage.variants.values):
        other, other_path = _allocation(fact(facts, f"/variants/{value}"), "post_layout", first_plane)
        _require(other["shape"][0] == value, "variants", "declared depth was not lowered")
        value_ref = ledger.reference(f"variants.{index}.depth", facts, f"/variants/{value}" + other_path + "/shape/0", namespace)
        lines.append(f'PL_VARIANT({index}, {{{{"{stage.variants.depth_handle}", {value_ref}}}}});')
    for endpoint in stage.endpoints:
        handle = smem_handles[endpoint.name]
        stride = strides[handle]
        name = planes[endpoint.name]
        layout_path = f"/layouts/{name}/declaration"
        layout = fact(captured, layout_path)

        def tree(value):
            if isinstance(value, list):
                return "{" + ", ".join(tree(item) for item in value) + "}"
            return str(value)

        for field in ("shape", "stride", "elem_bits", "swizzle"):
            if field in layout:
                ledger.record(f"endpoints.{endpoint.name}.layout.{field}",
                              "facts.json#" + root + layout_path + "/" + field, kind="captured")
        ledger.record(f"endpoints.{endpoint.name}.tile.shape",
                      "facts.json#" + root + f"/layouts/{name}/logical_shape", kind="captured")
        logical_shape = fact(captured, f"/layouts/{name}/logical_shape")[1:]
        affine = f'pl::affine{{{tree(layout["shape"])}, {tree(layout["stride"])}, {layout["elem_bits"]}}}'
        if "swizzle" in layout:
            swizzle = layout["swizzle"]
            affine += f'.swizzle({swizzle["B"]}, {swizzle["M"]}, {swizzle["S"]})'
        sharing, init = endpoint.sharing, endpoint.init
        _require(sharing is not None and init is not None and init.materializer == "dispatch"
                 and init.owner is not None, "protocol", "WS initialization needs a dispatch owner promise")
        _require(endpoint.full_arrive_by is not None and endpoint.empty_arrive_by is not None,
                 "protocol.arrive.by", "both arrival role promises are required")
        for field, source in (("ring.stage_stride_bytes", f"ring.{handle}.stage_stride_bytes"),
                              ("protocol.empty.arrive.count", "protocol.empty.arrive.count"),
                              ("protocol.empty.tx_bytes", "protocol.empty.tx_bytes")):
            ledger.alias(f"endpoints.{endpoint.name}.{field}", source)
        if sharing.member_of:
            full = f'pl::member{{"{sharing.member_of}", pl::tile}}'
        else:
            _require(isinstance(sharing.tx_bytes, tuple), "protocol.full.tx_bytes", "WS full completion must sum physical planes")
            ledger.alias(f"endpoints.{endpoint.name}.protocol.full.arrive.count", "protocol.full.arrive.count")
            names = ", ".join(f'"{name}"' for name in sharing.tx_bytes)
            full = f'pl::side{{"{full_handle}", pl::arrive("{endpoint.full_arrive_by}", {full_count}), pl::sum{{{{{names}}}}}}}'
        lines.append(f'''PL_ENDPOINT({endpoint.name}, pl::endpoint("{endpoint.name}", pl::{endpoint.direction}, "{endpoint.owner_role}")
    .space("smem").scope("cta").proxy("async").tile("bf16", {tree(logical_shape)})
    .layout({affine})
    .protocol(pl::mbarrier{{{full},
        pl::side{{"{empty_handle}", pl::arrive("{endpoint.empty_arrive_by}", {empty_count}), {empty_tx}}},
        pl::init{{"{init.owner}", {init.phase}, "{init.materializer}"}}}})
    .ring(pl::handle("{stage.variants.depth_handle}"), {stride})
    .bind("base", "{handle}").bind("full", "{full_handle}").bind("empty", "{empty_handle}"));''')
    return "\n".join(lines) + "\n"


def emit(out, module, generated, *, target, planes):
    module.PIPELINK.validate()
    for generation in generated.values():
        buffers = {}

        def visit(node):
            if isinstance(node, tvm.tirx.SBlock):
                buffers.update((buffer.name, buffer) for buffer in node.annotations.get("layout_map", {}))

        for func in generation["modules"]["post_layout"].functions.values():
            if isinstance(func, tvm.tirx.PrimFunc):
                tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
        for name in planes.values():
            buffer = buffers[name]
            layout = from_captured(generation["layout_objects"][name],
                                   elem_bits=tvm.DataType(buffer.dtype).bits, buffer=buffer)
            allocation, _ = _allocation(generation["facts"], "post_layout", name)
            _require(layout.get("ring_stride_elements") * layout["elem_bits"] // 8
                     == allocation["bytes"] // allocation["shape"][0],
                     f"layouts.{name}", "captured ring stride differs from allocation")
            generation["facts"]["layouts"][name]["declaration"] = layout
    records = {}
    for stage in module.PIPELINK.stages:
        _require(set(generated) == set(stage.variants.values), "variants", "every declared depth must be captured")
        role, declaration = next(iter(stage.roles.items()))
        prefix = symbol_prefix(stage)
        _require(set(planes) == {endpoint.name for endpoint in stage.endpoints},
                 "planes", "each declared endpoint must map to a physical plane")
        # Both roles of a pipeline name the same physical channel resources.
        pipeline_id = stage.stage_id.rsplit(".", 1)[0].removeprefix("org.pipelink.")
        handle_prefix = "pipelink_" + pipeline_id.replace(".", "_").replace("-", "_").lower() + "_"
        smem_handles = {endpoint.name: handle_prefix + "smem_" + endpoint.name for endpoint in stage.endpoints}
        full_handle, empty_handle = handle_prefix + "full", handle_prefix + "empty"
        namespace = prefix + "facts"
        facts = {"variants": {str(depth): generation["facts"] for depth, generation in generated.items()}}
        ledger = Ledger()
        for field in ("stage_id", "version", "roles", "endpoints", "launch_register_cap", "params", "variants"):
            ledger.record("declarations." + field, "Declarations." + field, kind="declared")
        annotations = emit_annotations(stage, facts, ledger, planes=planes, smem_handles=smem_handles,
                                       full_handle=full_handle, empty_handle=empty_handle)
        split = split_roles({depth: generation["modules"]["pre_codegen"] for depth, generation in generated.items()},
                            branch=declaration.branch, prefix=prefix,
                            smem_handles={planes[name]: handle for name, handle in smem_handles.items()},
                            full=full_handle, empty=empty_handle, declared_params=tuple(field.name for field in stage.params.fields),
                            target=target)
        params = f'struct alignas({stage.params.alignas}) {stage.params.name} {{\n' + "".join(f"  {field.cpp_type} {field.name};\n" for field in stage.params.fields) + "};\n"
        symbol = prefix + "role_" + role
        functions = {function.depth: function for function in split.functions}
        fields = {field.name: field for field in stage.params.fields}
        entry = f'extern "C" __device__ void {symbol}(const void* raw_params, unsigned variant) {{\n  auto const& params = *static_cast<const {stage.params.name}*>(raw_params);\n  switch (variant) {{\n'
        for index, depth in enumerate(stage.variants.values):
            function = functions[depth]
            arguments = []
            for name in function.params:
                field = fields[name]
                if field.cpp_type == "CUtensorMap":
                    arguments.append("&params." + name)
                elif field.cpp_type.endswith("*"):
                    arguments.append("params." + name)
                else:
                    raise AnnotationError("params", f"unsupported role argument type for {name}: {field.cpp_type}")
            entry += f"    case {index}: {function.name}({', '.join(arguments)}); return;\n"
        entry += '    default: asm volatile("trap;");\n  }\n}\n'
        path = Path(out) / "generation" / stage.stage_id.rsplit(".", 1)[-1]
        path.mkdir(parents=True, exist_ok=True)
        header = FactsHeader.render(facts, namespace)
        # Retain complete captured maps alongside the recognized endpoint layouts.
        for depth, generation in generated.items():
            header += FactsHeader.render({"layouts": generation["facts"]["layouts"]}, namespace + f"_depth_{depth}_maps")
        source = split.preamble + '#include <climits>\n#include <pipelink/stage.hpp>\n#include "facts.cuh"\n#include "params.h"\n'
        source += f'inline constexpr int {prefix}LaunchRegisterCap = {stage.launch_register_cap};\n'
        source += f'extern "C" __device__ void {symbol}(const void*, unsigned);\n' + annotations + split.body + entry
        projection = {"stage_id": stage.stage_id, "launch_register_cap": stage.launch_register_cap,
                      "ledger": ledger.sources, "split": split.record,
                      "source": str(path / "roles.cu"), "symbol_prefix": prefix,
                      "params_type": stage.params.name, "params_header": str(path / "params.h"),
                      "physical_layout": "captured logical planes and structurally recognized TileLang bank swizzles"}
        (path / "roles.cu").write_text(source)
        (path / "facts.cuh").write_text(header)
        (path / "params.h").write_text('#pragma once\n#include <cuda.h>\n' + params)
        (path / "facts.json").write_text(json.dumps({**facts, "passes": {str(depth): generation["passes"] for depth, generation in generated.items()}}, indent=2) + "\n")
        (path / "projection.json").write_text(json.dumps(projection, indent=2) + "\n")
        for depth, generation in generated.items():
            for name, snapshot in generation["snapshots"].items():
                (path / f"depth{depth}.{name}.tir").write_text(snapshot)
        records[stage.stage_id] = projection
    return records
