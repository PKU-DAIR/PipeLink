"""Assemble compiler-reflected host ABI records and semantic declarations.

The emit projection owns C facts; the public host parser owns the semantic
language. This module joins those facts and derives their existing digests,
without introducing an alternate host schema or a JSON overlay.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from ..canonjson import canonical_digest
from ..host.abi import ABI_TYPES, NEUTRAL_TYPE_ID_MIN, NEUTRAL_TYPE_ID_MAX, layout_digest
from ..host.contracts import owner_contract_digest, parse_host
from ..host.profile import DTYPES, HOST_ABI, dtypes_digest
from ..model import ClauseFailure, PackError, SchemaError, TargetStamp, register_clause


RECORD_KINDS = frozenset({
    "host_group", "host_abi_type", "host_residue", "host_scalars", "host_channel",
    "host_workspace", "host_transfer", "host_domain", "host_fact", "host_scheduler",
    "host_target", "host_declarations", "host_variants",
})
register_clause("P-HOST", phase="pack", description="Host annotations agree with compiler ABI facts and public host contracts.")


def _reject(path: str, message: str, **detail: Any) -> None:
    raise PackError([ClauseFailure("P-HOST", path, message, detail)])


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate semantic key: {key}")
        result[key] = value
    return result


def _semantic(record: dict, *, array: bool = False, forbidden: tuple[str, ...] = ()) -> Any:
    path = f"$.host.{record['kind']}.{record.get('name', '')}"
    try:
        value = json.loads(record["declaration"], object_pairs_hook=_object_pairs)
    except (ValueError, TypeError, KeyError) as exc:
        _reject(path, "invalid host semantic JSON", error=str(exc))
    if not isinstance(value, list if array else dict):
        _reject(path, "expected a semantic array" if array else "expected a semantic object")
    if not array and set(forbidden) & value.keys():
        _reject(path, "semantic declaration cannot restate compiled or derived fields",
                fields=sorted(set(forbidden) & value.keys()))
    return value


def _derived(value: dict, key: str, result: Any, path: str) -> None:
    if key in value:
        _reject(path, "semantic declaration cannot restate a derived field", field=key)
    value[key] = result




def _variants(values: list) -> list:
    result = []
    for value in values:
        if not isinstance(value, dict) or set(value) - {"key", "index", "entry"} or not {"key", "index"} <= value.keys():
            _reject("$.host.stage_variants", "expected complete stage variants with key, index and optional entry")
        if type(value["index"]) is not int or value["index"] < 0 or not isinstance(value["key"], dict):
            _reject("$.host.stage_variants", "invalid stage variant key or index")
        if any(type(key) is not str or type(item) is not int for key, item in value["key"].items()):
            _reject("$.host.stage_variants", "variant keys must map binding names to integers")
        if value.get("entry") is not None and type(value["entry"]) is not str:
            _reject("$.host.stage_variants", "variant entry must be a symbol or null")
        result.append({"key": value["key"], "index": value["index"], "entry": value.get("entry")})
    if not result or len({value["index"] for value in result}) != len(result):
        _reject("$.host.stage_variants", "variant inventory must be nonempty with unique indices")
    return sorted(result, key=lambda value: value["index"])


def _assemble(records: Mapping[str, list[dict]], manifest: dict) -> dict:
    def rows(kind: str) -> list[dict]:
        return records.get("host_" + kind, [])

    def single(kind: str, *, required: bool = False) -> dict | None:
        values = rows(kind)
        if len(values) > 1 or required and not values:
            _reject("$.host", f"expected {'exactly' if required else 'at most'} one PL_HOST_{kind.upper()} declaration")
        return values[0] if values else None

    group = single("group")
    declarations = single("declarations")
    single("target")
    if group is not None and declarations is not None:
        _reject("$.host", "PL_HOST_GROUP and PL_HOST_DECLARATIONS are mutually exclusive")
    profile = manifest["pipelink_abi"]["target"]["profile"]
    target = {"dtypes": dtypes_digest(profile)}
    if group is None and declarations is None:
        if any(records.get(kind) for kind in RECORD_KINDS - {"host_target"}):
            _reject("$.host", "host state and facts require PL_HOST_GROUP or PL_HOST_DECLARATIONS")
        return {"target": target}

    host = {
        "target": target,
        "owner": (group or declarations)["owner"],
        "params_group": group["params_group"] if group else None,
        "provider": None, "residue": None, "scalars": None, "domains": [], "channels": [],
        "workspaces": [], "transfers": [], "published_facts": [],
        "scheduler_contract": {}, "variants": {},
    }
    scheduler_record = single("scheduler")
    if scheduler_record is not None:
        host["scheduler_contract"] = _semantic(scheduler_record)
    for record in rows("fact"):
        host["published_facts"].append(_semantic(record))
    host["published_facts"].sort(key=lambda item: item["name"])
    scheduler = host["scheduler_contract"]
    if "strategy" in scheduler:
        _derived(scheduler, "input_schema_digest", canonical_digest(scheduler["schema"]), "$.host.scheduler_contract")

    if group is None:
        if any(rows(kind) for kind in ("abi_type", "residue", "scalars", "domain", "channel", "workspace", "transfer", "variants")):
            _reject("$.host", "providerless declarations cannot own runtime state or ABI inventory")
    else:
        inventory = []
        for record in rows("abi_type"):
            if record["name"] not in ABI_TYPES and NEUTRAL_TYPE_ID_MIN <= record["id"] <= NEUTRAL_TYPE_ID_MAX:
                _reject("$.host.provider.abi_types", "owner ABI type ID is reserved for the neutral inventory", type=record["name"], type_id=record["id"])
            members = {}
            for member in record["members"]:
                name = member["name"]
                if name in members:
                    _reject("$.host.provider.abi_types", "duplicate reflected ABI member", type=record["name"], member=name)
                if any(type(member[key]) is not int or member[key] < 0 for key in ("offset", "bytes")) or member["offset"] + member["bytes"] > record["bytes"]:
                    _reject("$.host.provider.abi_types", "reflected member lies outside its ABI type", type=record["name"], member=name)
                members[name] = {key: member[key] for key in ("offset", "bytes")}
            inventory.append({
                **{key: record[key] for key in ("id", "name", "bytes", "align")},
                "layout_digest": layout_digest(record["name"], record["bytes"], record["align"], members),
                "fields": sorted(members),
            })
        inventory.sort(key=lambda item: item["id"])
        by_id = {item["id"]: item for item in inventory}
        host["provider"] = {
            **{export: f"pipelink_params_{export}_{group['suffix']}" for export in ("abi", "query", "build", "release")},
            "protocol": 2, "args": {key: group[key] for key in ("bytes", "align", "type_id")},
            "abi_types": inventory, "owner_contract_digest": "",
        }
        residue = single("residue", required=True)
        for record, label in ((group, "owner args"), (residue, "residue")):
            typ = by_id.get(record["type_id"])
            if typ is None or typ["name"] != record["type"]:
                _reject("$.host.provider.abi_types", f"{label} type disagrees with ABI type inventory", type=record["type"], type_id=record["type_id"])
        host["residue"] = {
            **{key: residue[key] for key in ("id", "name", "bytes", "align", "type_id")},
            "layout_digest": by_id[residue["type_id"]]["layout_digest"],
            "read_roles": _semantic(residue, array=True),
        }
        scalar_record = single("scalars")
        if scalar_record is not None:
            host["scalars"] = {"id": scalar_record["id"], **_semantic(scalar_record, forbidden=("id",))}
        host["domains"] = sorted(({"id": record["id"], "name": record["name"]} for record in rows("domain")),
                                 key=lambda item: item["id"])
        for record in rows("channel"):
            channel = {"id": record["id"], **_semantic(record, forbidden=("id", "dtype_id"))}
            dtype = DTYPES[channel["storage_dtype"]]
            if record["storage_bits"] != dtype["bits"]:
                _reject("$.host.channels", "compiler storage element width differs from channel dtype", name=channel["name"], expected=dtype["bits"], actual=record["storage_bits"])
            channel["dtype_id"] = dtype["id"]
            channel.setdefault("logical_dtype", channel["storage_dtype"])
            host["channels"].append(channel)
        channels = {channel["name"]: channel for channel in host["channels"]}
        for record in rows("workspace"):
            workspace = {**_semantic(record, forbidden=("id", "align")),
                         **{key: record[key] for key in ("id", "align")}}
            workspace.setdefault("params_group", host["params_group"])
            host["workspaces"].append(workspace)
        for record in rows("transfer"):
            transfer = {"id": record["id"], **_semantic(record, forbidden=("id",))}
            if transfer["channel"] not in channels:
                _reject("$.host.transfers", "transfer channel must belong to this owner", name=transfer["name"], channel=transfer["channel"])
            host["transfers"].append(transfer)
        for key in ("channels", "workspaces", "transfers"):
            host[key].sort(key=lambda item: item["id"])

    if group is not None:
        current_stage = manifest["stage"]["id"]
        variants = {record["name"]: _variants(_semantic(record, array=True)) for record in rows("variants")}
        current = _variants(manifest["stage"]["variants"])
        if current_stage in variants and variants[current_stage] != current:
            _reject("$.host.stage_variants", "declared current-stage variants differ from PL_VARIANT inventory", stage=current_stage)
        variants[current_stage] = current
        host["variants"] = variants
        host["provider"]["owner_contract_digest"] = owner_contract_digest(host, variants)
    return host


def assemble_host(records: Mapping[str, list[dict]], manifest: dict) -> None:
    """Attach an optional validated host section, translating failures to P-HOST."""
    if not any(records.get(kind) for kind in RECORD_KINDS):
        return
    if manifest["params"]["bytes"]:
        _reject("$.params", "host annotations require PL_NO_PARAMS")
    stamp = manifest["pipelink_abi"]
    target = TargetStamp(stamp["schema"], stamp["target"]["profile"], stamp["target"]["lto_arch"], stamp["target"]["toolkit"], HOST_ABI)
    try:
        host = _assemble(records, manifest)
        host = parse_host(host, manifest["stage"]["id"], manifest["stage"]["roles"], target)
    except SchemaError as exc:
        raise PackError([ClauseFailure("P-HOST", failure.path, failure.message, failure.detail) for failure in exc.failures]) from exc
    except (KeyError, TypeError, ValueError) as exc:
        _reject("$.host", "invalid host annotation inventory or semantic expression", error=str(exc))
    manifest["host"] = host
    manifest["pipelink_abi"]["target"]["host_abi"] = HOST_ABI
    sections = manifest["provenance"]["sections"]
    sections["$.host"] = "declared"
    sections["$.host.target"] = "declared"
    if host.get("provider") is not None:
        sections["$.host.provider.args"] = "compiled"
        sections["$.host.provider.abi_types"] = "compiled"
        sections["$.host.provider.owner_contract_digest"] = "compiled"
        for key in ("bytes", "align"):
            sections[f"$.host.residue.{key}"] = "compiled"
        sections["$.host.residue.layout_digest"] = "compiled"
