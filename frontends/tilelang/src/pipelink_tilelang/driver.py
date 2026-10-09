"""Isolated pinned TileLang recipe codegen; stdin request, stdout JSON result.

Not imported by pipec2. No CUDA compilation or execution takes place here.
T.copy selects the natural packet intrinsic in an isolated compiler envelope;
only that exact primitive is projected, never an owning kernel's source body.
Scalar T.copy steps are already expression-addressed and cannot select a relay.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _bind_coordinates(value, coordinates):
    """Bind round-local coordinates without capturing published binding names."""
    if not isinstance(value, list):
        return value
    if value[0] == "coord" and value[1] in coordinates:
        return ["const", coordinates[value[1]]]
    return [value[0], *(_bind_coordinates(arg, coordinates) for arg in value[1:])]


def _require_native_receipt(args):
    # tl.tcgen05_ld(dps, repetitions, pack16, source, column_offset,
    #               pattern_bits, destination): N is NOT the pattern bitwidth.
    geometry = [int(args[index]) for index in (0, 1, 2, 5)] if len(args) == 7 else None
    if geometry != [32, 16, 0, 32]:
        raise RuntimeError(f"natural T.copy changed native receipt geometry: {geometry}")


def emit(request):
    if any(param["meaning"] == "dest_view" for param in request["params"]):
        raise RuntimeError("the view output belongs to the public shell, not the borrowed TileLang fill")
    import tilelang
    import tilelang.language as T
    from tilelang import tvm
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import device_codegen_without_compile

    if tilelang.__version__ != "0.1.14":
        raise RuntimeError(f"TileLang 0.1.14 required, found {tilelang.__version__}")
    tir = tvm.tirx
    target = tvm.target.Target({"kind": "cuda", "arch": request["target"]})
    target_entry = request["target_entry"]
    config = {"tl.disable_warp_specialized": True, "tirx.disable_vectorize": True}
    projection_checks = []
    dtype, elem_bytes = request["dtype"], request["elem_bytes"]
    element_ctype = request["element_ctype"]

    def nodes(body, cls):
        found = []
        tir.stmt_functor.post_order_visit(body, lambda node: found.append(node) if isinstance(node, cls) else None)
        return found

    def seq(statements):
        if len(statements) == 1:
            return statements[0]
        return tir.SeqStmt(statements) if statements else tir.Evaluate(0)

    def native_receipt_template():
        physical = T.Layout((128, 16), lambda m, n: (m, n))
        natural = T.Fragment((128, 16), forward_thread_fn=lambda m, n: m, forward_index_fn=lambda m, n: n)
        # This isolated template always projects the admitted 32-bit receipt.
        receipt_dtype = request["receipt_dtype"]

        @T.prim_func
        def receipt_recipe(Output: T.Tensor((128, 16), receipt_dtype)):
            with T.Kernel(1, threads=128):
                Source = T.alloc_tmem((128, 16), receipt_dtype)
                Natural = T.alloc_fragment((128, 16), receipt_dtype)
                T.annotate_layout({Source: physical, Natural: natural})
                T.copy(Source, Natural)
                T.copy(Natural, Output)

        artifact = tilelang.lower(receipt_recipe, target=target, enable_host_codegen=False, enable_device_compile=False)
        calls = [call for func in artifact.device_mod.functions.values()
                 for call in nodes(func.body, tir.Call)
                 if getattr(call.op, "name", "").endswith(".tcgen05_ld")]
        if len(calls) != 1:
            raise RuntimeError("natural T.copy did not select one native receipt")
        call = calls[0]
        _require_native_receipt(call.args)
        # This wrapper has exactly one x16 native instruction and an internal
        # wait::ld. The parent only supplies adjacent IR issue/wait pairs.
        header = (Path(tilelang.__file__).parent / "src/tl_templates/cuda/copy_sm100.h").read_text()
        start = header.index("tcgen05_ld_32dp32bNx(")
        wrapper = header[start:header.index("\n}", start)]
        if "fence_view_async_tmem_load();" not in wrapper or "tcgen05_ld_core<tl::tmem_ld_32dp32bNx<pack16>, 7, N>" not in wrapper:
            raise RuntimeError("pinned wrapper no longer has the evidenced load/completion realization")
        return call

    params, variables = [], {}
    pointer_domains = {resource["param"]: resource["domain"] for resource in request["resources"].values()
                       if resource["param"] is not None}
    for param in request["params"]:
        ctype = param["ctype"]
        if param["meaning"] == "source_base":
            ctype = "uint32_t" if request["context"]["source_space"] == "tmem" else f"const {request['element_ctype']}*"
        if ctype in {"uint32_t", "int32_t", "int"}:
            var = tir.Var(param["name"], "uint32" if ctype == "uint32_t" else "int32")
        else:
            domain = pointer_domains.get(param["name"])
            scope = "shared" if domain == "smem" or domain is None and param["meaning"] in {"scratch", "workspace"} else "local"
            var = tir.Var(param["name"], tvm.ir.PointerType(tvm.ir.PrimType("void" if ctype == "void*" else dtype), scope))
        params.append(var)
        variables[param["name"]] = var
    thread = tir.Var("tx", "int32")
    bindings = {param["name"]: variables[param["name"]] for param in request["params"]
                if param["meaning"] in {"binding", "source_base"}}
    for name, binding in request["borrows"].items():
        if isinstance(binding, int):
            bindings[name] = tir.IntImm("int64", binding)
        elif binding in variables:
            bindings[name] = variables[binding]
        elif name == request["context"].get("source_base_binding"):
            source = next(p for p in request["params"] if p["meaning"] == "source_base")
            bindings[name] = variables[source["name"]]
        else:
            raise RuntimeError(f"published binding {name!r} has no callable parameter")

    original_buffers, buffers, declarations = {}, {}, []
    for index, (name, resource) in enumerate(sorted(request["resources"].items())):
        if resource["domain"] not in {"reg", "smem"}:
            continue
        param = resource["param"]
        extent = resource["extent"]
        if extent is None:
            extent = max((domain.get("hi", 1) for domain in request["context"].get("binding_domains", {}).values()), default=1)
        if resource["domain"] == "smem":
            extent //= elem_bytes
        scope = "shared" if resource["domain"] == "smem" else "local"
        if param is None:
            buffer = tir.decl_buffer((extent,), dtype, name=f"pl_private_{index}", scope=scope, data_alignment=elem_bytes)
            declarations.append(tir.AllocBuffer(buffer))
        else:
            param_info = next(p for p in request["params"] if p["name"] == param)
            if param_info["ctype"] == "void*":
                pointer = tir.Var(f"pl_scratch_{index}", tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
                declarations.append(tir.Bind(pointer, tir.call_extern("handle", f"static_cast<{element_ctype}*>", variables[param])))
            else:
                pointer = variables[param]
            buffer = tir.decl_buffer((extent,), dtype, name=f"pl_view_{index}", data=pointer, scope=scope, data_alignment=elem_bytes)
            declarations.append(tir.DeclBuffer(buffer))
        buffers[name] = buffer
        original_buffers[name] = tir.decl_buffer(buffer.shape, dtype, name=f"ordinary_{index}", scope="global", data_alignment=elem_bytes)

    def native_width(width):
        for native in (8, 16, 32, 64):
            if width <= native:
                return native
        raise RuntimeError("fixed-width expression exceeds native integer width")

    def bit_width(value, metadata):
        op, *args = value
        if op in {"cbits", "tobits"}:
            return args[0]
        if op in {"band", "bor", "bxor", "shlc", "shrc"}:
            return bit_width(args[0], metadata)
        if op == "select":
            return bit_width(args[1], metadata)
        if op in {"coord", "bind"} and args[0] in metadata:
            return bit_width(metadata[args[0]], metadata)
        if op == "bind":
            return int(request["context"]["binding_domains"][args[0]]["sort"][5:-1])
        raise RuntimeError(f"expression {op!r} has no fixed bit width")

    def expression(value, metadata):
        if isinstance(value, int):
            return tir.IntImm("int64", value)
        op, *args = value
        if op == "const":
            return expression(args[0], metadata)
        if op == "cbits":
            return tir.IntImm(f"uint{native_width(args[0])}", args[1])
        if op == "bind" and args[0] in bindings:
            return bindings[args[0]]
        if op == "coord" and args[0] == "participant":
            return thread
        if op in {"coord", "bind"}:
            if args[0] in metadata:
                return expression(metadata[args[0]], {k: v for k, v in metadata.items() if k != args[0]})
            raise RuntimeError(f"unbound {op} reference {args[0]!r}")
        if op == "mulc":
            return args[0] * expression(args[1], metadata)
        if op in {"divc", "modc"}:
            a, b = expression(args[0], metadata), args[1]
            return tir.floordiv(a, b) if op == "divc" else tir.floormod(a, b)
        if op in {"shlc", "shrc"}:
            width = bit_width(args[0], metadata)
            dtype = f"uint{native_width(width)}"
            if args[1] >= width:
                return tir.IntImm(dtype, 0)
            source = expression(args[0], metadata)
            if op == "shrc":
                return source >> args[1]
            return (source << args[1]) & tir.IntImm(dtype, (1 << width) - 1)
        if op in {"tobits", "toint"}:
            width, arg = (args[0], args[1]) if op == "tobits" else (64, args[0])
            return tir.Cast(f"uint{native_width(width)}" if op == "tobits" else "int64", expression(arg, metadata))
        if op == "not":
            return tir.Not(expression(args[0], metadata))
        a, b = expression(args[0], metadata), expression(args[1], metadata)
        if op == "select":
            return tir.Select(a, b, expression(args[2], metadata))
        return {"add": lambda: a + b, "sub": lambda: a - b, "band": lambda: a & b,
                "bor": lambda: a | b, "bxor": lambda: a ^ b, "eq": lambda: a == b,
                "lt": lambda: a < b, "le": lambda: a <= b, "pand": lambda: tir.And(a, b),
                "por": lambda: tir.Or(a, b)}[op]()

    def view(operand, metadata, original=False):
        name = operand["resource"]
        index = expression(operand["index"], metadata)
        if request["resources"][name]["domain"] == "smem":
            index = tir.floordiv(index, elem_bytes)
        return tir.BufferLoad((original_buffers if original else buffers)[name], [index])

    def project_copy(step):
        operands, metadata = step["operands"], step["metadata"]
        original = T.copy(view(operands["src"], metadata, True), view(operands["dst"], metadata, True))
        forward = {original_buffers[name]: buffer for name, buffer in buffers.items()}
        backward = {buffer: original_buffers[name] for name, buffer in buffers.items()}
        def rewrite(body, mapping):
            def change(node):
                if isinstance(node, tir.BufferLoad):
                    return tir.BufferLoad(mapping[node.buffer], node.indices)
                if isinstance(node, tir.BufferStore):
                    return tir.BufferStore(mapping[node.buffer], node.value, node.indices)
                return None
            return tir.stmt_functor.ir_transform(body, None, change)
        projected = rewrite(original, forward)
        recovered = rewrite(projected, backward)
        if not tvm.ir.structural_equal(original, recovered):
            raise RuntimeError("scalar T.copy storage-view inverse differs")
        projection_checks.append(hashlib.sha256(tvm.ir.save_json(original).encode()).hexdigest())
        return projected

    with target, tvm.transform.PassContext(opt_level=3, config=config):
        template = native_receipt_template() if target_entry["receipt"] is not None and any(
            step["primitive"] == "tmem_packet_receipt" for graph in request["graphs"].values() for step in graph) else None
        site_bodies = []
        for site, steps in sorted(request["graphs"].items()):
            body = []
            members = request["context"]["participants"][site]
            for step in steps:
                primitive, operands, metadata = step["primitive"], step["operands"], step["metadata"]
                start = len(body)
                if primitive in {"reg_copy", "smem_store_scalar", "smem_load_scalar"}:
                    body.append(project_copy(step))
                elif primitive == "reg_copy_round":
                    for slot in range(operands["width"]):
                        scalar = {"src": {"resource": operands["src"], "index": operands.get("src_offset", 0) + slot},
                                  "dst": {"resource": operands["dst"], "index": operands.get("offset", 0) + slot}}
                        body.append(project_copy({"operands": scalar, "metadata": metadata}))
                elif primitive in {"smem_store_round", "smem_load_round"}:
                    # The model expands each round into these independently
                    # interleavable scalar slots; no atomic batch is invented.
                    for slot in range(operands["width"]):
                        private = {"resource": operands["src"] if primitive == "smem_store_round" else operands["dst"],
                                   "index": operands.get("offset", 0) + slot}
                        shared = {"resource": operands["dst"] if primitive == "smem_store_round" else operands["src"],
                                  "index": _bind_coordinates(operands["address"], {"slot": slot})}
                        scalar = {"src": private, "dst": shared} if primitive == "smem_store_round" else {"src": shared, "dst": private}
                        body.append(project_copy({"operands": scalar, "metadata": metadata}))
                elif primitive == "shfl_xor":
                    src, dst = view(operands["src"], metadata), view(operands["dst"], metadata)
                    body.append(tir.BufferStore(dst.buffer, T.shfl_xor(src, expression(operands["mask"], metadata), 32, 0xFFFFFFFF), dst.indices))
                elif primitive == "cta_rendezvous":
                    barrier = request["resources"][operands["resource"]]["base"]
                    sync = tir.Evaluate(T.sync_threads() if barrier == 0 else T.sync_threads(barrier, len(members)))
                    fences = target_entry["rendezvous_fences"]
                    if fences:
                        before, after = fences
                        body.extend([tir.Evaluate(tir.call_extern("void", before)), sync,
                                     tir.Evaluate(tir.call_extern("void", after))])
                    else:
                        body.append(sync)
                elif primitive == "tmem_packet_receipt":
                    # pipec preserves relocation in column units and checks this
                    # address against the certified finite packet worlds.
                    address = tir.Cast("uint32", expression(operands["address"], metadata))
                    dest = buffers[operands["dest"]]
                    offset = expression(operands.get("offset", 0), metadata)
                    args = list(template.args)
                    args[3], args[4], args[6] = address, tir.IntImm("int32", 0), dest.access_ptr("w", offset=offset, extent=16)
                    projected = tir.Call(template.dtype, template.op, args)
                    restored = list(projected.args)
                    for index in (3, 4, 6):
                        restored[index] = template.args[index]
                    if not tvm.ir.structural_equal(template, tir.Call(template.dtype, template.op, restored)):
                        raise RuntimeError("natural receipt inverse storage-view projection differs")
                    projection_checks.append(hashlib.sha256(tvm.ir.save_json(template).encode()).hexdigest())
                    body.append(tir.Evaluate(projected))
                else:
                    raise RuntimeError(f"unregistered recipe primitive {primitive!r}")
                if "guard" in step:
                    guarded = seq(body[start:])
                    del body[start:]
                    body.append(tir.IfThenElse(expression(step["guard"], metadata), guarded, None))
            predicate = None
            for member in members:
                clause = thread == member
                predicate = clause if predicate is None else tir.Or(predicate, clause)
            site_bodies.append(seq(body) if members == list(range(request["threads"])) else tir.IfThenElse(predicate, seq(body), None))
        thread_axis = tir.IterVar(tvm.ir.Range(0, request["threads"]), thread, tir.IterVar.ThreadIndex, "threadIdx.x")
        body = tir.AttrStmt(thread_axis, "thread_extent", request["threads"], seq(declarations + site_bodies))
        function = tir.PrimFunc(params, body).with_attr("global_symbol", "pipelink_adapter_kernel")
        function = function.with_attr("calling_conv", 2).with_attr("target", target)
        function = function.with_attr("tirx.is_global_func", True).with_attr("tirx.noalias", False)
        function = function.with_attr("thread_extent", {"threadIdx.x": request["threads"]})
        function = function.with_attr("tl.readonly_param_indices", [
            index for index, param in enumerate(request["params"])
            if param["ctype"].startswith("const ") or
            param["meaning"] == "source_base" and request["context"]["source_space"] != "tmem"])
        module = tvm.IRModule({"pipelink_adapter_kernel": function})
        backend = create_backend_context(target, None, "auto")
        source = device_codegen_without_compile(module, backend).inspect_source()
        force_include = target_entry["force_include"]
        if force_include and any(step["primitive"] == "cta_rendezvous" for graph in request["graphs"].values() for step in graph):
            header = f"#include <{force_include}>"
            if header not in source:
                # Keep the dependency order emitted by the natural receipt recipe.
                headers = "#include <tl_templates/cuda/copy_sm90.h>\n" + header + "\n"
                offset = source.index("#include <tl_templates/cuda/")
                source = source[:offset] + headers + source[offset:]
    return {"tilelang_version": tilelang.__version__, "source": source,
            "projection": {"inverse_structurally_equal": True, "scope": "scalar storage views and exact native receipt operands only",
                           "checked_original_tir_digests": projection_checks,
                           "device_tir_digest": hashlib.sha256(tvm.ir.save_json(module).encode()).hexdigest()}}


if __name__ == "__main__":
    # Compiler diagnostics stay separate from the machine-readable response.
    from contextlib import redirect_stdout
    request = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        result = emit(request)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
