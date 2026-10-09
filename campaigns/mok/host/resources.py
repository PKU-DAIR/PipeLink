"""CUDA-free P2 caller inventory, shared by dry-run and real loans."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple
    peer: bool = False
    active: bool = True

    @property
    def extents(self):
        return self.shape if self.active else (0,) * len(self.shape)

    @property
    def strides(self):
        stride = {"bf16": 16, "u8": 8, "i32": 32, "f32": 32}[self.dtype]
        result = []
        for extent in reversed(self.extents):
            result.append(stride if self.active else 0)
            stride *= extent
        return tuple(reversed(result))

    @property
    def alignment(self):
        return 4 if self.dtype == "i32" else 16


def inventory(s, pass_name, precision, act_width):
    L, H, I, E, P, A, k = (s[n] for n in ("num_local_tokens", "hidden", "intermediate", "num_local_experts", "schedule_capacity", "macrobatch_size", "topk"))
    result = {}
    def add(names, shape, passes="fwd bwd rec", dtype="bf16", peer=False, representation=None):
        for name in names.split():
            result[name] = TensorSpec(dtype, shape, peer, pass_name in passes.split() and representation in (None, precision))
    add("x_buffer", (L,H), peer=True)
    add("combine_buffer", (L*k,H), "fwd", peer=True)
    add("w_shared_gate w_shared_up", (I,H))
    add("w_shared_down", (H,I), "fwd bwd")
    add("gate_shared up_shared hidden_shared", (L,I))
    add("gate_routed up_routed", (A,I))
    add("y_shared", (L,H), "fwd")
    add("y_routed", (A,H), "fwd")
    add("schedule_peer_rank schedule_peer_token_idx", (P,), dtype="i32")
    add("num_tokens", (1,), dtype="i32")
    add("tokens_per_expert", (E,), dtype="i32")
    add("d_y_buffer", (L,H), "bwd", peer=True)
    add("d_x_routed_buffer", (L*k,H), "bwd", peer=True)
    add("router_weight_buffer d_router_weight_buffer", (L,k), "bwd", "f32", True)
    add("router_weights", (A,), "bwd", "f32")
    add("d_router_weight_partials", (A,I//act_width), "bwd", "f32")
    add("d_hidden_shared d_gate_shared d_up_shared", (L,I), "bwd")
    add("d_hidden_routed", (A,I), "bwd")
    add("d_x_shared", (L,H), "bwd")
    add("d_x_routed", (A,H), "bwd")
    add("d_w_shared_gate d_w_shared_up", (I,H), "bwd")
    add("d_w_shared_down", (H,I), "bwd")
    add("d_w_routed_gate d_w_routed_up", (E,I,H), "bwd")
    add("d_w_routed_down", (E,H,I), "bwd")
    def fp8(names, shape, passes="fwd bwd rec"):
        rows, cols = shape[-2:]
        experts = shape[0] if len(shape) == 3 else 1
        for name in names.split():
            add(name+"_fp8", shape, passes, "u8", representation="mxfp8")
            add(name+"_sc", (experts*rows//128,cols//128,32,16), passes, "u8", representation="mxfp8")
    fp8("x_routed", (A,H)); fp8("x_routed_t", (H,A))
    fp8("w_routed_gate w_routed_up", (E,I,H))
    fp8("w_routed_down", (E,H,I), "fwd")
    fp8("gate_routed up_routed hidden_routed", (A,I))
    fp8("hidden_routed_t", (I,A))
    fp8("w_routed_gate_t w_routed_up_t", (E,H,I), "bwd")
    fp8("w_routed_down_t", (E,I,H), "bwd")
    fp8("d_y_routed", (A,H), "bwd"); fp8("d_y_routed_t", (H,A), "bwd")
    fp8("d_gate_routed d_up_routed", (A,I), "bwd")
    fp8("d_gate_routed_t d_up_routed_t", (I,A), "bwd")
    for names, shape, passes in (
        ("x_routed", (A,H), "fwd bwd rec"), ("hidden_routed", (A,I), "fwd bwd rec"),
        ("w_routed_gate w_routed_up", (E,I,H), "fwd bwd rec"),
        ("w_routed_down", (E,H,I), "fwd bwd"), ("d_y_routed", (A,H), "bwd"),
        ("d_gate_routed d_up_routed", (A,I), "bwd")):
        add(" ".join(n+"_bf16" for n in names.split()), shape, passes, representation="bf16")
    return result


def load_cells(build_dir):
    root = Path(build_dir)
    index = json.loads((root/"index.json").read_text())
    matrix = json.loads((root/index["matrix"]).read_text())
    cells = []
    for row in matrix["cells"]:
        path = root/row["index"]
        cell = json.loads(path.read_text())
        link = json.loads((path.parent/cell["link"]).read_text())
        axes = {"pass": ("fwd","bwd","rec")[link["host"]["bindings"]["pipelink_mok_pass"]],
                "precision": ("bf16","mxfp8")[link["host"]["bindings"]["pipelink_mok_precision"]],
                "world": link["host"]["bindings"]["pipelink_mok_world"],
                "clamp": link["host"]["bindings"]["pipelink_mok_clamp"]}
        if any(cell[k] != v or row[k] != v for k,v in axes.items()):
            raise ValueError(f"index/binding mismatch: {row['id']}")
        cells.append({**cell, "directory": path.parent, "manifest": link})
    expected = {(p,q,w,c) for p in ("fwd","bwd","rec") for q in ("bf16","mxfp8") for w in (1,4) for c in (0,1)}
    actual = {(c["pass"],c["precision"],c["world"],c["clamp"]) for c in cells}
    if len(cells)!=24 or actual!=expected:
        raise ValueError("build matrix must contain exactly the 24 contract cells")
    return cells


def validate_inventory(cell, scalars):
    specs = inventory(scalars, cell["pass"], cell["precision"], cell["manifest"]["host"]["facts"]["mok.act.SWIGLU_Nb"])
    keys = set()
    for group in cell["manifest"]["host"]["groups"]:
        for channel in group["channels"]:
            if channel["source"]["kind"] != "caller":
                continue
            key = channel["source"]["name"]
            spec = specs[key]
            if (channel["storage_dtype"], channel["rank"], channel["domain"], channel["views"]) != (spec.dtype, len(spec.shape), "ep" if spec.peer else None, ["peer"] if spec.peer else []):
                raise ValueError(f"resource metadata mismatch: {key}")
            keys.add(key)
    if keys != specs.keys():
        raise ValueError(f"resource inventory differs: {sorted(keys ^ specs.keys())}")
    return specs
