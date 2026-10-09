"""Host declarations, cross-owner agreement and selected parameter routing."""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
import json
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

from ..canonjson import canonical_digest
from ..model import ClauseFailure, LinkError, ParamsLayout, SchemaError, StageManifest

_NAME = re.compile(r"[A-Za-z0-9_-]+(?:[.][A-Za-z0-9_-]+)+\Z")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCALARS = {"u64", "i64", "f64", "bool"}




def _fail(path: str, message: str, subreason: str = "inventory", **detail: Any) -> None:
    raise SchemaError([ClauseFailure("L-HOSTCHAN", path, message, {"subreason": subreason, **detail})])


def _obj(value: Any, path: str, required: str, optional: str = "") -> dict:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        _fail(path, "expected an object")
    missing = set(required.split()) - value.keys()
    extra = value.keys() - set(required.split()) - set(optional.split())
    if missing or extra:
        _fail(path, "object inventory mismatch", missing=sorted(missing), extra=sorted(extra))
    return value


def _list(value: Any, path: str) -> list:
    if not isinstance(value, list):
        _fail(path, "expected an explicit array")
    if len(value) >= 2**32:
        _fail(path, "inventory count exceeds C protocol uint32 capacity")
    return value


def _str(value: Any, path: str, *, qualified: bool = False) -> str:
    if type(value) is not str or not value or (qualified and not _NAME.fullmatch(value)):
        _fail(path, "expected a pipeline-qualified name" if qualified else "expected a nonempty string")
    return value


def _int(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**64:
        _fail(path, "integer outside declared uint64 range", minimum=minimum, maximum=2**64 - 1, actual=value)
    return value


def _wire(value: Any, path: str) -> int:
    _int(value, path, 1)
    if value >= 2**32:
        _fail(path, "wire ID or rank exceeds C protocol uint32 range", actual=value)
    return value


def _align(value: Any, path: str) -> int:
    _int(value, path, 1)
    if value & (value - 1):
        _fail(path, "alignment must be a power of two")
    return value


def _digest(value: Any, path: str) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        _fail(path, "expected a lowercase SHA-256 digest")
    return value


def _enum(value: Any, choices: Any, path: str) -> Any:
    if type(value) is not str or value not in choices:
        _fail(path, "unregistered contract value", expected=sorted(choices), actual=value)
    return value


def _unique(items: list, path: str, keys: tuple = ("id", "name")) -> None:
    for key in keys:
        seen = set()
        for item in items:
            value = item[key] if key is not None else item
            if value in seen:
                _fail(path, "duplicate local inventory entry", field=key, actual=value)
            seen.add(value)


def _names(value: Any, path: str, qualified: bool = False) -> list:
    items = _list(value, path)
    for item in items:
        _str(item, path, qualified=qualified)
    _unique(items, path, (None,))
    return items


def _role(value: Any, path: str) -> str:
    _str(value, path)
    parts = value.split(":")
    if len(parts) != 2 or not _NAME.fullmatch(parts[0]) or not _IDENT.fullmatch(parts[1]):
        _fail(path, "role reference must be stage.id:role")
    return value


def _reads(value: Any, path: str) -> None:
    for item in _names(value, path):
        _role(item, path)


def _expr(value: Any, path: str) -> None:
    if type(value) is int:
        return
    if not isinstance(value, dict):
        _fail(path, "invalid alias offset expression")
    if set(value) in ({"scalar"}, {"binding"}, {"fact"}):
        _str(next(iter(value.values())), path)
    elif set(value) == {"op", "args"}:
        _enum(value["op"], {"add", "mul"}, path)
        args = _list(value["args"], path)
        if len(args) != 2:
            _fail(path, "alias offset requires exactly two operands")
        for arg in args:
            _expr(arg, path)
    else:
        _fail(path, "unknown alias offset expression")


def _type(value: Any, path: str) -> None:
    if type(value) is str:
        _enum(value, _SCALARS | {"i32", "u32", "dtype"}, path)
        return
    obj = _obj(value, path, "kind", "type value items name values")
    kind = _enum(obj["kind"], {"static", "runtime", "tuple", "enum"}, path)
    if kind in {"static", "runtime"}:
        _obj(obj, path, "kind type" + (" value" if kind == "static" else ""))
        _enum(obj["type"], {"i32", "i64"}, path)
        if kind == "static":
            _value(obj["value"], obj["type"], path)
    elif kind == "tuple":
        _obj(obj, path, "kind items")
        for child in _list(obj["items"], path):
            _type(child, path)
    else:
        _obj(obj, path, "kind name values")
        _str(obj["name"], path, qualified=True)
        if not _names(obj["values"], path):
            _fail(path, "semantic enum must enumerate its values", "schema")


def _value(value: Any, typ: Any, path: str) -> None:
    if isinstance(typ, dict):
        kind = typ["kind"]
        if kind == "tuple":
            if not isinstance(value, list) or len(value) != len(typ["items"]):
                _fail(path, "typed tuple value has the wrong arity", "schema")
            for item, item_type in zip(value, typ["items"]):
                _value(item, item_type, path)
        elif kind == "enum":
            _enum(value, typ["values"], path)
        else:
            _value(value, typ["type"], path)
            if kind == "static" and value != typ["value"]:
                _fail(path, "value disagrees with static type leaf", "schema")
        return
    if typ == "dtype":
        from .profile import DTYPES
        _enum(value, DTYPES, path)
    elif typ == "bool":
        if type(value) is not bool:
            _fail(path, "boolean fact requires JSON boolean", "schema")
    elif typ == "f64":
        _obj(value, path, "bits")
        if _int(value["bits"], path) >= 2**64:
            _fail(path, "binary64 bits exceed 64 bits", "schema")
    else:
        signed = typ.startswith("i")
        bits = int(typ[1:])
        if type(value) is not int or not -(2 ** (bits - 1) if signed else 0) <= value < 2 ** (bits - int(signed)):
            _fail(path, "typed integer value is out of range", "schema", type=typ, actual=value)


def validate_typed_value(value: Any, typ: Any, path: str) -> None:
    try:
        _type(typ, path)
        _value(value, typ, path)
    except SchemaError as exc:
        raise LinkError(exc.failures) from exc


def _source(value: Any, path: str, typ: Any = None, *, profile: str) -> None:
    obj = _obj(value, path, "kind", "name value items")
    kind = _enum(obj["kind"], {"caller", "binding", "fact", "hardware", "constant", "tuple"}, path)
    if kind == "tuple":
        _obj(obj, path, "kind items")
        items = _list(obj["items"], path)
        if not isinstance(typ, dict) or typ.get("kind") != "tuple" or len(items) != len(typ["items"]):
            _fail(path, "tuple source requires matching ordered tuple leaves")
        for source, child in zip(items, typ["items"]):
            _source(source, path, child, profile=profile)
        return
    _obj(obj, path, "kind " + ("value" if kind == "constant" else "name"))
    if kind == "constant":
        if typ is not None:
            if isinstance(typ, dict) and "kind" in typ:
                _leaf_value(obj["value"], typ, path)
            else:
                _value(obj["value"], typ, path)
    else:
        _str(obj["name"], path, qualified=kind == "fact")
    if kind == "hardware":
        from .profile import HARDWARE_SCALARS
        _enum(obj["name"], HARDWARE_SCALARS, path)


def _selection(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        _fail(path, "selection must map binding names to finite integer domains", "schema")
    for name, values in value.items():
        _str(name, path)
        if not _list(values, path):
            _fail(path, "selection domain may not be empty", "schema")
        for item in values:
            _int(item, path)
        _unique(values, path, (None,))


def _initializer(value: Any, path: str) -> None:
    obj = _obj(value, path, "kind", "owner")
    kind = _enum(obj["kind"], {"caller", "owner", "none", "zero"}, path)
    if kind == "owner":
        _obj(obj, path, "kind owner")
        _str(obj["owner"], path, qualified=True)
    elif "owner" in obj:
        _fail(path, "only owner initialization may name an owner", "initializer")


def _validate_host_target(target: Any, host: dict, path: str) -> None:
    from .profile import HOST_ABI, dtypes_digest
    if target.host_abi != HOST_ABI:
        _fail("$.target.host_abi", "unsupported public host ABI", expected=HOST_ABI, actual=target.host_abi)
    expected = {"dtypes": dtypes_digest(target.profile)}
    if host["target"] != expected:
        _fail(path, "host target must bind the public dtype inventory", expected=expected, actual=host["target"])


def _leaf(value: Any, path: str) -> None:
    obj = _obj(value, path, "kind", "type values items")
    kind = _enum(obj["kind"], {"static", "runtime", "enum", "tuple", "dtype"}, path)
    if kind in {"static", "runtime"}:
        _obj(obj, path, "kind type")
        _enum(obj["type"], {"i32", "i64", "u64", "bool"} if kind == "static" else {"i32", "i64", "f64"}, path)
    elif kind == "enum":
        _obj(obj, path, "kind values")
        if not _names(obj["values"], path):
            _fail(path, "enum leaf requires a nonempty ordered value inventory")
    elif kind == "tuple":
        _obj(obj, path, "kind items")
        for item in _list(obj["items"], path):
            _leaf(item, path)
    else:
        _obj(obj, path, "kind")


def _leaf_value(value: Any, leaf: dict, path: str) -> None:
    kind = leaf["kind"]
    if kind == "tuple":
        if not isinstance(value, list) or len(value) != len(leaf["items"]):
            _fail(path, "scheduler tuple has the wrong arity")
        for item, child in zip(value, leaf["items"]):
            _leaf_value(item, child, path)
    elif kind == "enum":
        _enum(value, leaf["values"], path)
    else:
        _value(value, "dtype" if kind == "dtype" else leaf["type"], path)


def _leaf_source(source: dict, leaf: dict, path: str) -> None:
    kind = source["kind"]
    if kind == "tuple":
        if leaf["kind"] != "tuple" or len(source["items"]) != len(leaf["items"]):
            _fail(path, "scheduler tuple source has the wrong arity")
        for child, item in zip(source["items"], leaf["items"]):
            _leaf_source(child, item, path)
        return
    runtime = any(node.get("kind") == "runtime" for node in _walk(leaf))
    if runtime and kind not in {"caller", "hardware"}:
        _fail(path, "runtime scheduler leaves require caller or hardware sources")
    if kind in {"caller", "hardware"} and not runtime and leaf["kind"] != "enum":
        _fail(path, "static scheduler leaves cannot bind runtime sources")


def _scheduler(value: Any, path: str, profile: str) -> None:
    if value == {}:
        return
    if isinstance(value, dict) and set(value) == {"owner"}:
        _str(value["owner"], path, qualified=True)
        return
    obj = _obj(value, path, "owner strategy schema input_schema_digest inputs runtime_inputs workspace_ids")
    _str(obj["owner"], path, qualified=True)
    _str(obj["strategy"], path, qualified=True)
    schema = _obj(obj["schema"], path + ".schema", "required optional pinned runtime_i32 hardware_rules allow_no_launch")
    for key in ("required", "optional", "pinned", "hardware_rules"):
        if not isinstance(schema[key], dict):
            _fail(path, "scheduler schema inventories must be objects", field=key)
        for name in schema[key]:
            _str(name, path)
    if set(schema["required"]) & set(schema["optional"]):
        _fail(path, "required and optional inputs overlap")
    leaves = {**schema["required"], **schema["optional"]}
    for name, leaf in leaves.items():
        _leaf(leaf, path + ".schema." + name)
    if type(schema["allow_no_launch"]) is not bool:
        _fail(path, "allow_no_launch must be boolean")
    for name, value in schema["pinned"].items():
        if name not in leaves:
            _fail(path, "pinned input is absent from schema", input=name)
        _leaf_value(value, leaves[name], path)
    for name in _names(schema["runtime_i32"], path):
        if leaves.get(name) != {"kind": "runtime", "type": "i32"}:
            _fail(path, "runtime_i32 must name runtime i32 leaves", input=name)
    for name, rule in schema["hardware_rules"].items():
        _obj(rule, path, "kind name")
        _enum(rule["kind"], {"caller", "hardware"}, path)
        _source(rule, path, profile=profile)
        if name not in leaves:
            _fail(path, "hardware rule is absent from schema", input=name)
    if obj["input_schema_digest"] != canonical_digest(schema):
        _fail(path, "scheduler input schema digest differs from its body")
    inputs = _list(obj["inputs"], path)
    for item in inputs:
        _obj(item, path, "name source")
        name = _str(item["name"], path)
        if name not in leaves:
            _fail(path, "unknown scheduler input", input=name)
        _source(item["source"], path, leaves[name], profile=profile)
        _leaf_source(item["source"], leaves[name], path + ".inputs." + name)
        if name in schema["hardware_rules"] and item["source"] != schema["hardware_rules"][name]:
            _fail(path, "scheduler input differs from its hardware source rule", input=name)
        if name in schema["pinned"] and item["source"]["kind"] == "constant" and item["source"]["value"] != schema["pinned"][name]:
            _fail(path, "scheduler input differs from pinned value", input=name)
    _unique(inputs, path, ("name",))
    missing = set(schema["required"]) - {item["name"] for item in inputs}
    if missing:
        _fail(path, "scheduler required inputs are missing", missing=sorted(missing))
    runtime = set(_names(obj["runtime_inputs"], path))
    expected = {item["name"] for item in inputs if any(node.get("kind") in {"caller", "hardware"} for node in _walk(item["source"]))}
    if runtime != expected:
        _fail(path, "runtime inventory differs from runtime sources", expected=sorted(expected), actual=sorted(runtime))
    ids = _list(obj["workspace_ids"], path)
    for value in ids:
        _wire(value, path)
    _unique(ids, path, (None,))


def _public_declarations(obj: dict, path: str, profile: str) -> None:
    facts = _list(obj["published_facts"], path)
    for fact in facts:
        _obj(fact, path, "name type owner stage consumers", "value variants")
        _str(fact["name"], path, qualified=True)
        _type(fact["type"], path)
        _str(fact["owner"], path, qualified=True)
        _str(fact["stage"], path, qualified=True)
        _names(fact["consumers"], path, qualified=True)
        if ("value" in fact) == ("variants" in fact):
            _fail(path, "fact requires exactly one static value or finite variant table", "schema")
        if "value" in fact:
            _value(fact["value"], fact["type"], path)
        else:
            variants = _list(fact["variants"], path)
            if not variants:
                _fail(path, "fact variant table must be nonempty", "schema")
            for row in variants:
                _obj(row, path, "selection value")
                _selection(row["selection"], path)
                _value(row["value"], fact["type"], path)
    _unique(facts, path, ("name",))
    _scheduler(obj["scheduler_contract"], path + ".scheduler_contract", profile)


def parse_host(value: Any, stage_id: str, roles: dict, target: Any) -> dict:
    path = "$.host"
    if isinstance(value, dict) and set(value) == {"target"} and not roles:
        _validate_host_target(target, value, path + ".target")
        return deepcopy(value)
    obj = _obj(value, path, "target owner params_group provider residue scalars domains channels workspaces transfers published_facts scheduler_contract variants")
    _str(obj["owner"], path, qualified=True)
    variants = obj["variants"]
    if not isinstance(variants, dict):
        _fail(path + ".variants", "variant inventory must map stages to compiled rows")
    for stage, rows in variants.items():
        _str(stage, path + ".variants", qualified=True)
        if not _list(rows, path + ".variants"):
            _fail(path + ".variants", "compiled variant inventory must be nonempty")
        keys = set()
        for row in rows:
            _obj(row, path + ".variants", "key index entry")
            _int(row["index"], path + ".variants")
            if not isinstance(row["key"], dict):
                _fail(path + ".variants", "variant key must map bindings to integers")
            for binding, value in row["key"].items():
                _str(binding, path + ".variants")
                _int(value, path + ".variants")
            if row["entry"] is not None:
                _str(row["entry"], path + ".variants")
            key = canonical_digest(row["key"])
            if key in keys:
                _fail(path + ".variants", "duplicate compiled variant key")
            keys.add(key)
        _unique(rows, path + ".variants", ("index",))
    if obj["params_group"] is None:
        if variants:
            _fail(path + ".variants", "providerless declarations cannot own compiled variants")
        if roles or obj["provider"] is not None or obj["residue"] is not None or obj["scalars"] is not None or any(obj[key] != [] for key in ("domains", "channels", "workspaces", "transfers")):
            _fail(path, "providerless declarations cannot own runtime state or device roles")
        _public_declarations(obj, path, target.profile)
        if "strategy" in obj["scheduler_contract"]:
            _fail(path, "providerless declarations cannot be geometry query authorities", "geometry")
        _validate_host_target(target, obj, path + ".target")
        return deepcopy(obj)
    _str(obj["params_group"], path, qualified=True)
    provider = _obj(obj["provider"], path + ".provider", "abi query build release protocol args owner_contract_digest abi_types")
    if type(provider["protocol"]) is not int or provider["protocol"] != 2:
        _fail(path, "only host C protocol 2 is supported")
    suffix = None
    for export in ("abi", "query", "build", "release"):
        name = _str(provider[export], path)
        prefix = "pipelink_params_" + export + "_"
        if not _IDENT.fullmatch(name) or not name.startswith(prefix) or len(name) == len(prefix):
            _fail(path, "provider must publish the four exact C export forms", export=export)
        current = name[len(prefix):]
        if suffix is not None and current != suffix:
            _fail(path, "provider export owner suffixes differ")
        suffix = current
    args = _obj(provider["args"], path, "bytes align type_id")
    _int(args["bytes"], path)
    _align(args["align"], path)
    _wire(args["type_id"], path)
    _digest(provider["owner_contract_digest"], path)
    types = _list(provider["abi_types"], path)
    for typ in types:
        _obj(typ, path, "id name bytes align layout_digest fields")
        _wire(typ["id"], path)
        _str(typ["name"], path)
        if not _IDENT.fullmatch(typ["name"]):
            _fail(path, "ABI type name must be a public C identifier")
        _int(typ["bytes"], path)
        _align(typ["align"], path)
        _digest(typ["layout_digest"], path)
        fields = typ["fields"]
        if isinstance(fields, dict):
            for field, layout in fields.items():
                _str(field, path)
                _obj(layout, path, "offset bytes")
                _int(layout["offset"], path)
                _int(layout["bytes"], path)
                if layout["offset"] + layout["bytes"] > typ["bytes"]:
                    _fail(path, "ABI field lies outside its type")
        else:
            _names(fields, path)
    _unique(types, path)
    by_id = {typ["id"]: typ for typ in types}
    if args["type_id"] not in by_id or (args["bytes"] and any(args[k] != by_id[args["type_id"]][k] for k in ("bytes", "align"))):
        _fail(path, "owner args disagree with ABI type inventory")
    residue = obj["residue"]
    if residue is not None:
        residue = _obj(obj["residue"], path + ".residue", "id name bytes align type_id layout_digest read_roles")
        _wire(residue["id"], path)
        _str(residue["name"], path, qualified=True)
        _int(residue["bytes"], path)
        _align(residue["align"], path)
        _digest(residue["layout_digest"], path)
        _reads(residue["read_roles"], path)
        _wire(residue["type_id"], path)
        if residue["type_id"] not in by_id or residue["layout_digest"] != by_id[residue["type_id"]]["layout_digest"] or (residue["bytes"] and any(residue[k] != by_id[residue["type_id"]][k] for k in ("bytes", "align"))):
            _fail(path, "residue disagrees with ABI type inventory")
    from .profile import DTYPES
    scalars = obj["scalars"]
    if scalars is not None:
        _obj(scalars, path + ".scalars", "id read_roles fields")
        _wire(scalars["id"], path)
        _reads(scalars["read_roles"], path)
        if residue is not None and scalars["id"] == residue["id"]:
            _fail(path, "scalar image and residue IDs must differ")
        fields = _list(scalars["fields"], path)
        for field in fields:
            _obj(field, path, "id name type source")
            _wire(field["id"], path)
            _str(field["name"], path, qualified=True)
            _enum(field["type"], _SCALARS, path)
            _source(field["source"], path, field["type"], profile=target.profile)
        _unique(fields, path)
    domains = _list(obj["domains"], path)
    for domain in domains:
        _obj(domain, path + ".domains", "id name")
        _wire(domain["id"], path + ".domains")
        if not _IDENT.fullmatch(_str(domain["name"], path + ".domains")):
            _fail(path + ".domains", "domain name must be an identifier", "domain", actual=domain["name"])
    _unique(domains, path + ".domains")
    declared_domains = {domain["name"] for domain in domains}
    channels = _list(obj["channels"], path)
    for channel in channels:
        _obj(channel, path, "id name storage_dtype logical_dtype dtype_id rank access domain views source alias initializer lifetime")
        _wire(channel["id"], path)
        _str(channel["name"], path, qualified=True)
        for key in ("logical_dtype", "storage_dtype"):
            _enum(channel[key], DTYPES, path)
        storage, logical = (DTYPES[channel[key]] for key in ("storage_dtype", "logical_dtype"))
        if type(channel["dtype_id"]) is not int or channel["dtype_id"] != storage["id"]:
            _fail(path, "channel dtype wire ID differs from profile", "dtype")
        integers = {"signed", "unsigned"}
        if channel["storage_dtype"] != channel["logical_dtype"] and (
            storage["category"] not in integers or logical["category"] not in integers
            or max(storage["bits"], logical["bits"]) % min(storage["bits"], logical["bits"])
        ):
            _fail(path, "channel dtypes do not admit bit-preserving integer packing", "dtype")
        _wire(channel["rank"], path)
        _enum(channel["access"], {"read", "write", "read_write"}, path)
        source = _obj(channel["source"], path, "kind name")
        _enum(source["kind"], {"caller", "workspace"}, path)
        _str(source["name"], path, qualified=source["kind"] == "workspace")
        # A domain channel is one caller-lent allocation symmetric over a caller-named domain
        # (decision 32, option A); views are the remote addresses build receives.
        views = _names(channel["views"], path)
        if views != sorted(set(views)) or set(views) - {"multicast", "peer"}:
            _fail(path, "channel views must be a sorted subset of multicast, peer", "domain", actual=views)
        if channel["domain"] is None:
            if views:
                _fail(path, "channel views require a domain", "domain", channel=channel["name"])
        elif _str(channel["domain"], path) not in declared_domains:
            _fail(path, "channel domain is not declared by its owner", "domain", channel=channel["name"],
                  domain=channel["domain"])
        elif source["kind"] != "caller":
            _fail(path, "domain channels require a caller source", "domain", channel=channel["name"])
        _initializer(channel["initializer"], path)
        if channel["alias"] is not None:
            alias = _obj(channel["alias"], path, "channel byte_offset bit_offset")
            _str(alias["channel"], path, qualified=True)
            _expr(alias["byte_offset"], path)
            _expr(alias["bit_offset"], path)
        _enum(channel["lifetime"], {"release"}, path)
    _unique(channels, path)
    workspaces = _list(obj["workspaces"], path)
    for workspace in workspaces:
        _obj(workspace, path, "id name params_group align space channel bytes initializer reset")
        _wire(workspace["id"], path)
        _str(workspace["name"], path, qualified=True)
        if workspace["params_group"] != obj["params_group"]:
            _fail(path, "workspace must belong to its declaring params group")
        _align(workspace["align"], path)
        _enum(workspace["space"], {"gmem"}, path)
        if workspace["channel"] is not None:
            _str(workspace["channel"], path, qualified=True)
        _enum(workspace["bytes"], {"query"}, path)
        _initializer(workspace["initializer"], path)
        if workspace["initializer"].get("owner", obj["owner"]) != obj["owner"]:
            _fail(path, "workspace initialization must remain owner-local", "initializer")
        if _enum(workspace["reset"], {"open", "launch"}, path) == "launch" and workspace["initializer"]["kind"] != "zero":
            _fail(path, "per-launch reset requires zero initialization", "initializer")
    _unique(workspaces, path)
    transfers = _list(obj["transfers"], path)
    for transfer in transfers:
        _obj(transfer, path, "id name channel device_endpoint access")
        _wire(transfer["id"], path)
        _str(transfer["name"], path, qualified=True)
        _str(transfer["channel"], path, qualified=True)
        _enum(transfer["access"], {"read", "write"}, path)
        ref = _obj(transfer["device_endpoint"], path, "stage endpoint")
        _str(ref["stage"], path, qualified=True)
        _str(ref["endpoint"], path)
    _unique(transfers, path)
    _public_declarations(obj, path, target.profile)
    _validate_host_target(target, obj, path + ".target")
    return deepcopy(obj)


def role_view_name(stage: Any, role: str) -> str:
    return "plk_role_view_" + stage.symbol_prefix + role


def role_view_fields(host: dict, role: str) -> list[dict]:
    return [
        {"field": key, "kind": key, "id": record["id"] if record else None,
         "name": record.get("name", "pipelink.shared_scalars") if record else None,
         "read": record is not None and role in record["read_roles"]}
        for key in ("residue", "scalars")
        for record in (host[key],)
    ]


def role_view_declaration(name: str, fields: list[dict]) -> str:
    members = [f"  const void* {item['field']};" for item in fields] or ["  unsigned char reserved;"]
    return "\n".join([f"typedef struct {name} {{", *members, f"}} {name};"])


def _group_records(stages: list) -> tuple[dict, dict]:
    groups = {}
    members = {}
    for stage in sorted(stages, key=lambda item: item.id):
        if not stage.host or stage.host.get("params_group") is None:
            continue
        name = stage.host["params_group"]
        if name in groups and groups[name] != stage.host:
            _fail(f"stages.{stage.id}.host", "same-group references must agree with the complete defining record")
        groups[name] = stage.host
        members.setdefault(name, []).append(stage.id)
    return groups, members



def owner_contract_digest(host: dict, stage_variants: dict) -> str:
    declaration = deepcopy(host)
    del declaration["provider"]["owner_contract_digest"]
    variants = {
        stage: sorted(values, key=lambda value: canonical_digest(value))
        for stage, values in sorted(stage_variants.items())
    }
    return canonical_digest({"host": declaration, "stage_variants": variants})

def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)

def host_link(stages: Iterable[StageManifest]) -> bool:
    """Host declarations select the host link path independently of schema version."""
    return any(stage.host is not None for stage in stages)



def _schema_type_agrees(typ: Any, leaf: dict) -> bool:
    if leaf["kind"] == "tuple":
        return isinstance(typ, dict) and typ.get("kind") == "tuple" and len(typ["items"]) == len(leaf["items"]) and all(_schema_type_agrees(t, l) for t, l in zip(typ["items"], leaf["items"]))
    if leaf["kind"] == "enum":
        return isinstance(typ, dict) and typ.get("kind") == "enum" and typ["values"] == leaf["values"]
    if leaf["kind"] == "dtype":
        return typ == "dtype"
    return (typ.get("type") if isinstance(typ, dict) else typ) == leaf["type"]


def _scheduler_sources(source: dict, leaf: dict, facts: dict, scalars: dict, path: str) -> None:
    if source["kind"] == "tuple":
        for child, item in zip(source["items"], leaf["items"]):
            _scheduler_sources(child, item, facts, scalars, path)
    elif source["kind"] == "fact":
        fact = facts[source["name"]]
        if not _schema_type_agrees(fact["type"], leaf):
            _fail(path, "scheduler fact type differs from schema leaf", fact=fact["name"])
        values = [fact["value"]] if "value" in fact else [row["value"] for row in fact["variants"]]
        for value in values:
            _leaf_value(value, leaf, path)
    elif source["kind"] in {"caller", "hardware"}:
        if leaf["kind"] == "tuple":
            _fail(path, "runtime tuple requires per-leaf sources")
        projected = [field for field in scalars.values() if field["source"] == source]
        expected = {"i64"} if leaf.get("type") in {"i32", "i64"} else {"f64"} if leaf.get("type") == "f64" else {"i64", "u64"}
        if not projected or any(field["type"] not in expected for field in projected):
            _fail(path, "runtime scheduler source lacks a matching shared scalar projection", source=source)


def _validate_assembly(pipeline: Any, stages: list) -> None:
    from .abi import ABI_TYPES
    from .profile import DTYPES
    from ..profile import get_profile
    from ..layout import canonicalize
    if not host_link(stages):
        return
    for stage in stages:
        if stage.host is not None and (stage.entry_abi != "v2" or stage.params.bytes):
            _fail("host", "host-bearing stages require v2 entries and no stage parameter blob")
        if any(getattr(stage.target, key) != getattr(pipeline.target, key) for key in ("profile", "lto_arch", "toolkit")):
            _fail(f"stages.{stage.id}.target", "host target differs from pipeline")
        if stage.host is not None:
            _validate_host_target(stage.target, stage.host, f"stages.{stage.id}.host.target")
    groups, members = _group_records(stages)
    stages_by_id = {stage.id: stage for stage in stages}
    owners, role_owners = {}, {}
    for name, group in groups.items():
        owner = group["owner"]
        if owner in owners and owners[owner] != name:
            _fail("host.groups", "one owner must have exactly one params group", owner=owner)
        owners[owner] = name
        for stage_id in members[name]:
            for role in stages_by_id[stage_id].roles:
                role_owners[f"{stage_id}:{role}"] = owner
    facts, channels, workspaces, scalars, scalar_ids = {}, {}, {}, {}, {}
    scheduler = None
    strategy = None
    for stage in stages:
        if stage.host and stage.host.get("owner") is not None and stage.host["params_group"] is None:
            owners.setdefault(stage.host["owner"], None)
    for stage in stages:
        if not stage.host or stage.host.get("owner") is None or stage.host["params_group"] is not None:
            continue
        for fact in stage.host["published_facts"]:
            if fact["owner"] != stage.host["owner"] or fact["stage"] != stage.id or fact["name"] in facts:
                _fail(f"stages.{stage.id}.host.published_facts", "fact ownership or uniqueness mismatch")
            if set(fact["consumers"]) - owners.keys():
                _fail(f"stages.{stage.id}.host.published_facts", "fact has an unknown consumer")
            facts[fact["name"]] = fact
    for name, group in sorted(groups.items()):
        owner = group["owner"]
        path = f"host.groups.{name}"
        expected_roles = {role for role, role_owner in role_owners.items() if role_owner == owner}
        for record in (group["scalars"], group["residue"]):
            if record and set(record["read_roles"]) - expected_roles:
                _fail(path, "scalar or residue readers must remain within their owner", owner=owner)
        variants = {stage: [variant.to_json_obj() for variant in stages_by_id[stage].variants] for stage in members[name]}
        declared_variants = {stage: sorted(rows, key=lambda row: row["index"]) for stage, rows in group["variants"].items()}
        compiled_variants = {stage: sorted(rows, key=lambda row: row["index"]) for stage, rows in variants.items()}
        if declared_variants != compiled_variants:
            _fail(path, "owner variant inventory differs from linked member stages")
        if group["provider"]["owner_contract_digest"] != owner_contract_digest(group, variants):
            _fail(path, "owner contract digest does not cover its complete compiled declaration and variant inventory", "inventory", owner=owner)
        inventory = {item["name"]: item for item in group["provider"]["abi_types"]}
        missing = set(ABI_TYPES) - inventory.keys()
        if missing:
            _fail(path, "provider must declare the complete recursively reachable neutral ABI inventory", missing=sorted(missing))
        for type_name, typ in ABI_TYPES.items():
            if set(inventory[type_name]["fields"]) != {field[0] for field in typ._fields_}:
                _fail(path, "neutral ABI field inventory differs from public C protocol", type=type_name)
        for stage_id in members[name]:
            stage = stages_by_id[stage_id]
            for role in stage.roles:
                view_name = role_view_name(stage, role)
                fields = role_view_fields(group, f"{stage.id}:{role}")
                expected_fields = {field["field"] for field in fields} or {"reserved"}
                if view_name not in inventory or set(inventory[view_name]["fields"]) != expected_fields:
                    _fail(path, "role-local view is absent or differs from compiler ABI inventory", role=f"{stage.id}:{role}", type=view_name, expected=sorted(expected_fields))
        for field in (group["scalars"] or {}).get("fields", []):
            if field["name"] in scalars and scalars[field["name"]] != field:
                _fail(path, "shared scalar has incompatible IDs, types or sources", field=field["name"])
            if field["id"] in scalar_ids and scalar_ids[field["id"]] != field["name"]:
                _fail(path, "shared scalar wire ID names different fields", id=field["id"])
            scalars[field["name"]] = field
            scalar_ids[field["id"]] = field["name"]
        for channel in group["channels"]:
            channels.setdefault(channel["name"], []).append((owner, channel))
        for workspace in group["workspaces"]:
            if workspace["name"] in workspaces:
                _fail(path, "workspace has more than one allocating group", channel=workspace["name"])
            workspaces[workspace["name"]] = (owner, workspace)
        for fact in group["published_facts"]:
            if fact["name"] in facts and facts[fact["name"]] != fact:
                _fail(path, "published fact has conflicting defining records", "schema", fact=fact["name"])
            if fact["owner"] != owner or fact["stage"] not in members[name]:
                _fail(path, "fact must be published by its defining owner and stage", "schema", fact=fact["name"])
            if set(fact["consumers"]) - owners.keys():
                _fail(path, "published fact names an unknown consumer owner", "schema", fact=fact["name"])
            facts[fact["name"]] = fact
        contract = group["scheduler_contract"]
        if "strategy" in contract:
            identity = (contract["strategy"], contract["input_schema_digest"])
            if strategy is not None and strategy != identity:
                _fail(path, "scheduler strategy or schema body differs across stages")
            strategy = identity
            if contract["owner"] == owner:
                if scheduler is not None:
                    _fail(path, "exactly one owner may define scheduler geometry", "geometry")
                scheduler = contract
                if set(contract["workspace_ids"]) - {ws["id"] for ws in group["workspaces"]}:
                    _fail(path, "scheduler workspace inventory is not owned by its authority")
    if any(stage.roles for stage in stages) and scheduler is None:
        _fail("host.scheduler_contract", "device roles require one geometry query authority", "geometry")
    lender_domains: dict[str, tuple[str, str | None]] = {}
    for name, records in sorted(channels.items()):
        first = records[0][1]
        for owner, channel in records[1:]:
            for key in ("storage_dtype", "logical_dtype", "rank", "access", "domain", "views", "source", "alias", "initializer", "lifetime"):
                if first[key] != channel[key]:
                    _fail(f"host.channels.{name}.{key}", "cross-owner channel declarations disagree", "dtype" if "dtype" in key else "domain" if key == "views" else key, owner=owner)
        if first["source"]["kind"] == "caller":
            # Same-source channels may overlap; a lender is one local or one domain-symmetric allocation.
            # Views may differ per channel: the caller's one Resource carries their union.
            other, domain = lender_domains.setdefault(first["source"]["name"], (name, first["domain"]))
            if domain != first["domain"]:
                _fail(f"host.channels.{name}.domain", "channels sharing a caller source disagree on domain", "domain", channel=other)
        writers = [owner for owner, channel in records if channel["access"] != "read"]
        if len(writers) > 1:
            _fail(f"host.channels.{name}", "multiple writers require a registered disjoint/cooperative phase protocol", "access", channel=name, actual=writers)
        init = first["initializer"]
        if init["kind"] == "owner":
            if init["owner"] not in {owner for owner, _ in records}:
                _fail(f"host.channels.{name}", "initializer does not declare access to the channel", "initializer")
            if not any(owner == init["owner"] and channel["access"] != "read" for owner, channel in records):
                _fail(f"host.channels.{name}", "initializer lacks write access", "initializer")
        if first["source"]["kind"] == "workspace":
            ws_name = first["source"]["name"]
            if ws_name not in workspaces:
                _fail(f"host.channels.{name}", "workspace channel source does not resolve")
            ws_owner, ws = workspaces[ws_name]
            if ws["channel"] != name or ws["initializer"] != init:
                _fail(f"host.channels.{name}", "workspace export or initializer differs from its typed channel", "initializer")
            if init["kind"] == "caller":
                _fail(f"host.channels.{name}", "caller cannot initialize a not-yet-allocated workspace", "initializer")
        alias = first["alias"]
        if alias is not None:
            if alias["channel"] not in channels or alias["channel"] == name:
                _fail(f"host.channels.{name}", "alias target is absent or self-referential", "alias")
            base = channels[alias["channel"]][0][1]
            if first["source"] != base["source"] or first["lifetime"] != base["lifetime"]:
                _fail(f"host.channels.{name}", "aliases must identify the same lender/allocation and lifetime", "alias")
    for ws_name, (_owner, ws) in workspaces.items():
        if ws["channel"] is not None and ws["channel"] not in channels:
            _fail(f"host.workspaces.{ws_name}", "workspace export channel does not resolve")
    for name, records in channels.items():
        seen = {name}
        current = records[0][1]
        while current["alias"] is not None:
            target = current["alias"]["channel"]
            if target in seen:
                _fail(f"host.channels.{name}", "cyclic allocation alias", "alias")
            seen.add(target)
            current = channels[target][0][1]
    transfers_seen = set()
    for group_name, group in sorted(groups.items()):
        path = f"host.groups.{group_name}"
        local_channels = {channel["name"]: channel for channel in group["channels"]}
        for transfer in group["transfers"]:
            if transfer["name"] in transfers_seen:
                _fail(path, "transfer names must be globally unique")
            transfers_seen.add(transfer["name"])
            if transfer["channel"] not in local_channels:
                _fail(path, "transfer channel is absent from its owning group")
            channel = local_channels[transfer["channel"]]
            if transfer["access"] != channel["access"] and channel["access"] != "read_write":
                _fail(path, "transfer exceeds channel access", "access")
            ref = transfer["device_endpoint"]
            stage = stages_by_id.get(ref["stage"])
            candidates = [] if stage is None else [ep for ep in stage.endpoints if ep.name == ref["endpoint"]]
            if len(candidates) != 1:
                _fail(path, "device endpoint reference does not resolve", "layout")
            endpoint = candidates[0]
            direction = "produce" if transfer["access"] == "read" else "consume"
            if endpoint.direction != direction or endpoint.protocol is None or endpoint.protocol.kind != "mbarrier":
                _fail(path, "transfer requires an mbarrier endpoint with matching direction", "access")
            role = f"{stage.id}:{endpoint.owner_role}"
            if group["residue"] is None or role not in group["residue"]["read_roles"]:
                _fail(path, "transfer endpoint role must read the owner's residue")
            base = next((claim for claim in stage.claims if claim.handle == endpoint.binds.get("base")), None)
            physical = canonicalize(endpoint, base, get_profile(pipeline.target.profile))
            if physical.elem_bits != DTYPES[channel["logical_dtype"]]["bits"]:
                _fail(path, "endpoint element width differs from logical channel dtype", "dtype")
        contract = group["scheduler_contract"]
        if contract and contract["owner"] != (scheduler or {}).get("owner"):
            _fail(path, "scheduler reference does not resolve to geometry authority")
        for node in _walk(group):
            binding = node.get("binding") if set(node) == {"binding"} else node.get("name") if node.get("kind") == "binding" else None
            if binding is not None and binding not in pipeline.handles:
                _fail(path, "expression references an unknown binding", binding=binding)
            if set(node) == {"scalar"} and node["scalar"] not in scalars:
                _fail(path, "expression references an undeclared shared scalar", scalar=node["scalar"])
            fact_name = node.get("fact") if set(node) == {"fact"} else node.get("name") if node.get("kind") == "fact" else None
            if fact_name is not None:
                fact = facts.get(fact_name)
                if fact is None or (group["owner"] != fact["owner"] and group["owner"] not in fact["consumers"]):
                    _fail(path, "fact binding is missing or not authorized", fact=fact_name)
        for field in (group["scalars"] or {}).get("fields", []):
            if field["source"]["kind"] == "fact":
                fact = facts[field["source"]["name"]]
                widening = {"i32": {"i64"}, "u32": {"i64", "u64"}}
                if fact["type"] != field["type"] and field["type"] not in widening.get(str(fact["type"]), set()):
                    _fail(path, "scalar fact projection requires matching types or lossless integer widening", fact=fact["name"])
        if "strategy" in contract:
            leaves = {**contract["schema"]["required"], **contract["schema"]["optional"]}
            for item in contract["inputs"]:
                _scheduler_sources(item["source"], leaves[item["name"]], facts, scalars, path + ".scheduler." + item["name"])


def validate_host_assembly(pipeline: Any, stages: list) -> list[ClauseFailure]:
    try:
        _validate_assembly(pipeline, stages)
    except SchemaError as exc:
        return exc.failures
    return []


def _selected(selection: dict, bindings: dict) -> bool:
    return all(name in bindings and bindings[name] in values for name, values in selection.items())


@lru_cache(maxsize=32)
def _probe_views(declarations: str, inventory_json: str) -> dict:
    from .abi import probe_layouts
    with tempfile.TemporaryDirectory(prefix="plk-role-views-") as directory:
        header = Path(directory) / "role_views.h"
        header.write_text(declarations, encoding="utf-8")
        return probe_layouts(extra_headers=(header,), extra_types=json.loads(inventory_json))

def validate_scheduler_value(value: Any, leaf: dict, path: str) -> None:
    """Validate an invocation value, including enum wire indices, against a schema leaf."""
    try:
        if leaf["kind"] == "tuple":
            if not isinstance(value, (list, tuple)) or len(value) != len(leaf["items"]):
                _fail(path, "scheduler tuple has the wrong arity")
            for item, child in zip(value, leaf["items"]):
                validate_scheduler_value(item, child, path)
        elif leaf["kind"] == "enum" and type(value) is int:
            if not 0 <= value < len(leaf["values"]):
                _fail(path, "enum wire value is outside its declared inventory", actual=value)
        elif leaf.get("type") == "f64" and type(value) in (int, float):
            try:
                struct.pack("<d", value)
            except (OverflowError, struct.error):
                _fail(path, "scheduler f64 value is outside binary64 range", actual=value)
        else:
            _leaf_value(value, leaf, path)
    except SchemaError as exc:
        raise LinkError(exc.failures) from exc


def _resolved_transfers(stages: list, bindings: dict, profile: Any) -> dict:
    from ..layout import _expand_atom, canonicalize
    from ..model import BarrierSide, SharedMember, SolveError, TxSum
    from ..protocol import _endpoint_pairs, _shared_group, _tx_value
    pairs = _endpoint_pairs(stages)
    canon, endpoints = {}, {}
    for stage in stages:
        for endpoint in stage.endpoints:
            if endpoint.protocol is None or endpoint.protocol.kind != "mbarrier":
                continue
            base = next((claim for claim in stage.claims if claim.handle == endpoint.binds.get("base")), None)
            canon[f"{stage.id}:{endpoint.name}:{endpoint.direction}"] = canonicalize(endpoint, base, profile)
            endpoints[(stage.id, endpoint.name)] = endpoint
    roots, expected = set(), {}
    for endpoint in endpoints.values():
        full = endpoint.protocol.full
        if isinstance(full, SharedMember):
            roots.add(full.member_of)
        elif isinstance(full, BarrierSide) and isinstance(full.tx_bytes, TxSum):
            roots.add(endpoint.name)
    try:
        for root in sorted(roots):
            _, contributions, _ = _shared_group(root, pairs, stages, canon)
            expected.update(contributions)
        result = {}
        for stage in stages:
            for transfer in (stage.host or {}).get("transfers", []):
                ref = transfer["device_endpoint"]
                key = (ref["stage"], ref["endpoint"])
                if key in result:
                    continue
                endpoint = endpoints[key]
                physical = canon[f"{key[0]}:{key[1]}:{endpoint.direction}"]
                full = endpoint.protocol.full
                tx = expected[endpoint.name] if endpoint.name in expected else _tx_value(full.tx_bytes, endpoint, pairs, canon)
                ring = None
                if endpoint.ring is not None:
                    depth = endpoint.ring.depth
                    ring = {"depth": depth.fixed if depth.fixed is not None else bindings[depth.handle],
                            "stage_stride_bytes": endpoint.ring.stage_stride_bytes}
                layout = _expand_atom(endpoint.layout, profile, endpoint).to_json_obj()
                swizzle = layout.get("swizzle")
                if swizzle is not None:
                    swizzle = {**swizzle, "unit": swizzle.get("unit", "element")}
                result[key] = {
                    "endpoint": {"space": endpoint.space, "dtype": physical.dtype,
                                 "elem_bits": physical.elem_bits, "tile": endpoint.tile.to_json_obj(),
                                 "layout": layout, "swizzle": swizzle, "ring": ring,
                                 "align": physical.align_bits // 8 if physical.align_bits is not None else 1,
                                 "tile_bytes": physical.footprint_bits // 8, "access": endpoint.direction},
                    "expect_tx_bytes": tx,
                }
        return result
    except SolveError as exc:
        raise LinkError(exc.failures) from exc


def _source_value(source: dict, facts: dict, bindings: dict) -> Any:
    kind = source["kind"]
    if kind == "constant":
        return deepcopy(source["value"])
    if kind == "fact":
        return deepcopy(facts[source["name"]])
    if kind == "binding":
        return bindings[source["name"]]
    if kind == "tuple":
        return [_source_value(item, facts, bindings) for item in source["items"]]
    raise ValueError("runtime source has no link-time value")


def resolve_host(stages: list, solution: Any, pipeline: Any) -> dict | None:
    """Resolve selected host declarations into one shared image and owner residues."""
    if not host_link(stages):
        return None
    from ..link import _binding_values, _cluster_result
    from ..profile import get_profile
    from .profile import HOST_ABI, TOOLCHAIN_PROVENANCE, dtypes_digest
    profile = get_profile(pipeline.target.profile)
    bindings = {name: value[0] for name, value in _binding_values(pipeline, solution).items()}
    for stage in stages:
        if not any(variant.key == solution.variants.get(stage.id, {}) for variant in stage.variants):
            raise LinkError([ClauseFailure("L-HOSTCHAN", f"host.variants.{stage.id}", "selected variant is outside the compiled owner inventory", {})])
    groups, members = _group_records(stages)
    merged_fields = sorted({field["name"]: deepcopy(field) for group in groups.values()
                            for field in (group["scalars"] or {}).get("fields", [])}.values(), key=lambda item: item["id"])
    has_scalars = any(group["scalars"] is not None for group in groups.values())
    scalar_bytes = 24 + 16 * len(merged_fields) if has_scalars else 0
    cursor, alignment = scalar_bytes, profile.caps.param_align
    selected_groups, role_views, declarations, inventory = [], {}, [], {}
    stages_by_id = {stage.id: stage for stage in stages}
    facts, fact_inventory, scheduler = {}, {}, {}
    transfers = _resolved_transfers(stages, bindings, profile) if any(group["transfers"] for group in groups.values()) else {}
    for owner_id, (name, original) in enumerate(sorted(groups.items()), 1):
        group = deepcopy(original)
        group.update(owner_id=owner_id, stage_ids=members[name])
        for key in ("channels", "workspaces", "transfers"):
            group[key].sort(key=lambda item: item["id"])
        group["provider"]["abi_types"].sort(key=lambda item: item["id"])
        if group["scalars"] is not None:
            group["scalars"].update(offset=0, bytes=scalar_bytes, align=8, fields=deepcopy(merged_fields))
        residue = group["residue"]
        if residue is not None:
            cursor = (cursor + residue["align"] - 1) & -residue["align"]
            residue["offset"] = cursor
            cursor += residue["bytes"]
            alignment = max(alignment, residue["align"])
        for transfer in group["transfers"]:
            ref = transfer["device_endpoint"]
            transfer.update(deepcopy(transfers[(ref["stage"], ref["endpoint"])]))
        for stage_id in members[name]:
            stage = stages_by_id[stage_id]
            for role in sorted(stage.roles):
                role_key = f"{stage.id}:{role}"
                type_name = role_view_name(stage, role)
                view_fields = role_view_fields(original, role_key)
                declarations.append(role_view_declaration(type_name, view_fields))
                inventory[type_name] = [field["field"] for field in view_fields]
                for field in view_fields:
                    record = group[field["field"]] if field.pop("read") else None
                    field.update(offset=record["offset"] if record else None, bytes=record["bytes"] if record else 0,
                                 align=record["align"] if record else None)
                abi_type = next(typ for typ in group["provider"]["abi_types"] if typ["name"] == type_name)
                role_views[role_key] = {"owner": group["owner"], "params_group": name, "type": type_name,
                                        "type_id": abi_type["id"], "fields": view_fields}
        if group["scheduler_contract"].get("owner") == group["owner"] and "strategy" in group["scheduler_contract"]:
            scheduler = deepcopy(group["scheduler_contract"])
        selected_groups.append(group)
    declaration_records = [
        {"stage_id": stage.id, **deepcopy(stage.host)} for stage in sorted(stages, key=lambda item: item.id)
        if stage.host and stage.host.get("owner") is not None and stage.host["params_group"] is None
    ]
    for record in [*selected_groups, *declaration_records]:
        for fact in record["published_facts"]:
            selected_fact = deepcopy(fact)
            if "variants" in fact:
                rows = [row for row in fact["variants"] if _selected(row["selection"], bindings)]
                if len(rows) != 1:
                    raise LinkError([ClauseFailure("L-HOSTCHAN", f"host.facts.{fact['name']}", "selected fact must match exactly one declared variant", {"actual": len(rows)})])
                selected_fact["value"] = rows[0]["value"]
                del selected_fact["variants"]
            facts[fact["name"]] = selected_fact["value"]
            fact_inventory[fact["name"]] = selected_fact
    for item in scheduler.get("inputs", []):
        if any(node.get("kind") in {"caller", "hardware"} for node in _walk(item["source"])):
            continue
        item["value"] = _source_value(item["source"], facts, bindings)
        schema = scheduler["schema"]
        leaf = {**schema["required"], **schema["optional"]}[item["name"]]
        try:
            _leaf_value(item["value"], leaf, "host.scheduler_contract.inputs." + item["name"])
            if item["name"] in schema["pinned"] and item["value"] != schema["pinned"][item["name"]]:
                _fail("host.scheduler_contract.inputs." + item["name"], "selected value differs from pinned schema value")
        except SchemaError as exc:
            raise LinkError(exc.failures) from exc
    try:
        layouts = _probe_views("\n\n".join(declarations), json.dumps(inventory, sort_keys=True)) if groups else {}
        for group in selected_groups:
            for typ in group["provider"]["abi_types"]:
                if typ["name"] not in layouts:
                    continue
                actual = layouts[typ["name"]]
                if any(typ[key] != actual[key] for key in ("bytes", "align", "layout_digest")) or set(typ["fields"]) != set(actual["fields"]) or ("id" in actual and typ["id"] != actual["id"]):
                    _fail(f"host.groups.{group['params_group']}.provider.abi_types", "compiled public/role-view layout differs from declared ABI inventory", type=typ["name"], expected=typ, actual=actual)
            for view in role_views.values():
                if view["params_group"] == group["params_group"]:
                    view["layout"] = layouts[view["type"]]
    except SchemaError as exc:
        raise LinkError(exc.failures) from exc
    blobs = []
    for stage in sorted(stages, key=lambda item: item.id):
        if stage.host is not None:
            continue
        cursor = (cursor + stage.params.align - 1) & -stage.params.align
        blobs.append({"stage_id": stage.id, "offset": cursor, "bytes": stage.params.bytes, "align": stage.params.align})
        cursor += stage.params.bytes
        alignment = max(alignment, stage.params.align)
    total = (cursor + alignment - 1) & -alignment
    if total > profile.caps.param_limit_bytes:
        raise LinkError([ClauseFailure("L-HOSTCHAN", "host.params.bytes", "selected parameters exceed target limit", {"actual": total, "limit": profile.caps.param_limit_bytes})])
    cluster = _cluster_result(solution, stages)
    host = {
        "protocol": 2, "host_abi": HOST_ABI, "profile": pipeline.target.profile,
        "target": {"dtypes": dtypes_digest(pipeline.target.profile)},
        "groups": selected_groups, "params": {"bytes": total, "align": alignment, "blobs": blobs},
        "role_views": role_views, "role_view_header": "\n\n".join(declarations) + ("\n" if declarations else ""),
        "scheduler_contract": scheduler, "facts": facts, "fact_inventory": fact_inventory,
        "declarations": declaration_records, "bindings": bindings, "variants": deepcopy(solution.variants),
        "static_launch": {"block": [solution.launch.threads, 1, 1], "dynamic_smem_bytes": solution.launch.dyn_smem_bytes,
                          "regs_cap": solution.launch.regs_cap, "cluster": list(cluster) if isinstance(cluster, tuple) else cluster.to_json_obj()},
    }
    host["selection_digest"] = canonical_digest(host)
    host["toolchain_provenance"] = deepcopy(TOOLCHAIN_PROVENANCE)
    return host


def host_params_layout(host: dict) -> ParamsLayout:
    entries = []
    scalar_image = next((group["scalars"] for group in host["groups"] if group["scalars"] is not None), None)
    if scalar_image is not None:
        entries.append(("pipelink.shared_scalars", scalar_image["offset"], scalar_image["bytes"], scalar_image["align"]))
    for group in host["groups"]:
        residue = group["residue"]
        if residue is not None:
            entries.append((residue["name"], residue["offset"], residue["bytes"], residue["align"]))
    entries.extend((blob["stage_id"], blob["offset"], blob["bytes"], blob["align"]) for blob in host["params"]["blobs"])
    return ParamsLayout(host["params"]["bytes"], host["params"]["align"], sorted(entries, key=lambda entry: entry[0]))
