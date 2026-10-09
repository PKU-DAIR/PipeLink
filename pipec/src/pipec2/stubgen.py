from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from .model import (
    ClauseFailure,
    LinkError,
    Pipeline,
    ProtocolExpansion,
    RegGroup,
    Solution,
    StageManifest,
    register_clause,
)


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXTERN_C_RE = re.compile(r'\bextern\s*"C"')
_ATTRIBUTE_RE = re.compile(r"__attribute__\s*\(\(.*?\)\)", re.DOTALL)
_NON_CODE_RE = re.compile(
    r"//[^\n]*|/\*.*?(?:\*/|\Z)|"
    r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(.*?(?:\)\1"|\Z)|'
    r'"(?:\\.|[^"\\])*(?:"|\\?\Z)|'
    r"'(?:\\.|[^'\\])*(?:'|\\?\Z)",
    re.DOTALL,
)
_PARAMETER_KEYWORDS = frozenset(
    {
        "alignas",
        "auto",
        "bool",
        "char",
        "const",
        "double",
        "enum",
        "float",
        "int",
        "long",
        "noexcept",
        "register",
        "restrict",
        "short",
        "signed",
        "struct",
        "typename",
        "union",
        "unsigned",
        "void",
        "volatile",
        "wchar_t",
        "__restrict",
        "__restrict__",
    }
)

register_clause(
    "L-STUB",
    phase="link",
    description="call declarations and selected bindings generate valid device stubs",
)


def _link_error(message: str) -> LinkError:
    return LinkError([ClauseFailure("L-STUB", "stubgen", message, {})])


def _require_identifier(value: str, what: str) -> str:
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise _link_error(f"{what} must be a C identifier, got {value!r}")
    return value


def _reg_group_name(group: RegGroup, pipeline: Pipeline) -> str:
    handle = pipeline.handles.get(group.handle)
    if handle is None or not handle.owner.startswith("reg_group:"):
        raise _link_error(
            f"register handle {group.handle!r} does not identify its reg_group owner"
        )
    return _require_identifier(handle.owner.removeprefix("reg_group:"), "reg_group name")


def _merge_binding(bindings: dict[str, int], name: str, value: int) -> None:
    _require_identifier(name, "binding handle")
    previous = bindings.get(name)
    if previous is not None and previous != value:
        raise _link_error(
            f"binding handle {name!r} has conflicting selected values {previous} and {value}"
        )
    bindings[name] = value


def _bindings(solution: Solution, pipeline: Pipeline) -> dict[str, int]:
    result: dict[str, int] = {}
    for mapping in (solution.depths, solution.native_bases, pipeline.forced_bindings):
        for name in sorted(mapping):
            _merge_binding(result, name, mapping[name])
    for group in sorted(
        solution.reg_groups,
        key=lambda item: (item.warp_lo, item.warp_hi, item.handle, tuple(sorted(item.roles))),
    ):
        _merge_binding(result, group.handle, group.value)
    for stage_id in sorted(solution.variants):
        for name in sorted(solution.variants[stage_id]):
            _merge_binding(result, name, solution.variants[stage_id][name])
    if "pipelink_launch_regs" in pipeline.handles:
        _merge_binding(result, "pipelink_launch_regs", solution.launch.regs_cap)
    return result


def _code_mask(text: str) -> str:
    """Mask comments and literals for parsing, preserving indices and newlines."""
    return _NON_CODE_RE.sub(
        lambda match: re.sub(r"[^\n]", " ", match.group()), text
    )


def _declarator(decl: str, identifier: str) -> tuple[str, re.Match[str]]:
    mask = _code_mask(decl)
    matches = list(re.finditer(rf"\b{re.escape(identifier)}\b(?=\s*\()", mask))
    if len(matches) != 1:
        raise _link_error(f"call declaration must declare {identifier!r} exactly once")
    return mask, matches[0]


def _top_level_positions(mask: str) -> Iterator[int]:
    """Visit un-nested code, including template-aware parameter separators."""
    round_depth = square_depth = brace_depth = angle_depth = 0
    for index, char in enumerate(mask):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif not char.isspace() and round_depth == square_depth == brace_depth == angle_depth == 0:
            yield index


def _matching_paren(mask: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(mask)):
        if mask[index] == "(":
            depth += 1
        elif mask[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise _link_error("call declaration has an unterminated parameter list")


def declaration_parts(decl: str, export_symbol: str) -> tuple[str, str, str]:
    mask, symbol_match = _declarator(decl, export_symbol)
    opening = mask.index("(", symbol_match.end())
    closing = _matching_paren(mask, opening)

    terminator = closing + 1
    nested = 0
    while terminator < len(mask):
        char = mask[terminator]
        if char in "([{":
            nested += 1
        elif char in ")]}":
            nested -= 1
        elif char == ";" and nested == 0:
            break
        terminator += 1
    if terminator == len(mask):
        raise _link_error(f"call declaration for {export_symbol!r} is not semicolon-terminated")

    extern_matches = [
        match for match in _EXTERN_C_RE.finditer(decl, 0, symbol_match.start())
    ]
    if not extern_matches:
        raise _link_error(f"call declaration for {export_symbol!r} has no extern \"C\"")
    prefix = decl[extern_matches[-1].end() : symbol_match.start()]
    prefix = _ATTRIBUTE_RE.sub(" ", prefix)
    for qualifier in ("__device__", "__forceinline__", "__noinline__"):
        prefix = re.sub(rf"\b{qualifier}\b", " ", prefix)
    prefix = " ".join(prefix.split())
    if not prefix:
        raise _link_error(f"call declaration for {export_symbol!r} has no return type")
    params = decl[opening : closing + 1]
    suffix = decl[closing + 1 : terminator].strip()
    return prefix, params, suffix


def _split_parameters(parameters: str) -> list[str]:
    body = parameters[1:-1]
    if not body.strip() or body.strip() == "void":
        return []
    parts: list[str] = []
    start = 0
    for index in _top_level_positions(_code_mask(body)):
        if body[index] == ",":
            parts.append(body[start:index])
            start = index + 1
    parts.append(body[start:])
    return parts


def _without_default(parameter: str) -> str:
    for index in _top_level_positions(_code_mask(parameter)):
        if parameter[index] == "=":
            return parameter[:index]
    return parameter


def _parameter_name(parameter: str) -> str:
    parameter = _ATTRIBUTE_RE.sub(" ", _without_default(parameter)).strip()
    if parameter == "...":
        raise _link_error("variadic call declarations cannot be forwarded")
    nested_name = re.search(
        r"\(\s*[*&]+\s*(?:const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        parameter,
    )
    if nested_name:
        return nested_name.group(1)

    mask = _code_mask(parameter)
    name = None
    name_start = name_end = 0
    end = 0
    for index in _top_level_positions(mask):
        if index < end or not (mask[index].isalpha() or mask[index] == "_"):
            continue
        match = _WORD_RE.match(mask, index)
        assert match is not None
        word = match.group()
        if word not in _PARAMETER_KEYWORDS:
            name_start, name_end = match.span()
            name = word
        end = match.end()
    if name is None:
        raise _link_error(f"call parameter has no forwardable name: {parameter!r}")
    before = mask[:name_start].strip()
    after = mask[name_end:].lstrip()
    qualifiers = {"const", "volatile", "struct", "class", "typename", "__restrict", "__restrict__"}
    if (not before or before.endswith("::") or after.startswith(("*", "&", "::", "<"))
            or not any(word not in qualifiers for word in _WORD_RE.findall(before))):
        raise _link_error(f"call parameter has no forwardable name: {parameter!r}")
    return name


def _parameter_has_name(parameter: str) -> bool:
    try:
        _parameter_name(parameter)
    except LinkError:
        return False
    return True


@dataclass(frozen=True)
class ViewCall:
    """A call into a role-less host-group stage. The caller passes its own params view
    (`{residue, scalars}`) first; the trampoline recovers the params image from the
    view's `source_field`, which points `source_offset` bytes into it, and passes the
    callee its owner's view (`None`: record absent) instead, built from the same image
    so the pointers keep kernel-parameter provenance after inlining."""

    source_field: str
    source_offset: int
    residue: int | None
    scalars: int | None


def first_parameter_type(decl: str, export_symbol: str) -> str | None:
    """The first parameter's type with its name removed and pointer spacing canonical."""
    items = _split_parameters(declaration_parts(decl, export_symbol)[1])
    if not items:
        return None
    item = _ATTRIBUTE_RE.sub(" ", _without_default(items[0])).strip()
    if _parameter_has_name(item):
        item = item[: item.rindex(_parameter_name(item))]
    text = re.sub(r"\s*\*\s*", "*", " ".join(item.split()))
    return "const void*" if text in {"const void*", "void const*"} else text


def call_symbol(channel: str) -> str:
    return f"pipelink_call_{_require_identifier(channel, 'call channel')}"


def reg_config_symbol(group: str) -> str:
    return f"pipelink_regs_config_{_require_identifier(group, 'reg_group name')}"


def _trampoline(channel: str, callable_export: tuple[str, str, str], view: ViewCall | None = None) -> str:
    _require_identifier(channel, "call channel")
    export_symbol, decl, digest = callable_export
    _require_identifier(export_symbol, "call export symbol")
    return_type, parameters, suffix = declaration_parts(decl, export_symbol)
    items = _split_parameters(parameters)
    if items and not all(_parameter_has_name(item) for item in items):
        # Typed (name-less) declarations printed from the export's function type: the
        # trampoline names its own parameters and forwards them positionally.
        named = [f"{item.strip()} pl_a{index}" for index, item in enumerate(items)]
        parameters = "(" + ", ".join(named) + ")"
        names = [f"pl_a{index}" for index in range(len(items))]
    else:
        names = [_parameter_name(item) for item in items]
    body = []
    if view is not None:
        source = (f"static_cast<const unsigned char*>("
                  f"static_cast<const pipelink_host_view*>({names[0]})->{view.source_field})")
        image = source if not view.source_offset else f"{source} - {view.source_offset}u"
        members = ["nullptr" if offset is None else "pl_image" if not offset else f"pl_image + {offset}u"
                   for offset in (view.residue, view.scalars)]
        body = [f"  const unsigned char* const pl_image = {image};",
                f"  const pipelink_host_view pl_view = {{{', '.join(members)}}};"]
        names = ["&pl_view", *names[1:]]
    suffix_text = f" {suffix}" if suffix else ""
    decl_text = decl if decl.endswith("\n") else decl + "\n"
    return "".join([
        f"// call channel {channel}; decl-sha256 {digest}\n",
        decl_text,
        'extern "C" __attribute__((used)) __forceinline__ __device__\n',
        f"{return_type} {call_symbol(channel)}{parameters}{suffix_text} {{\n",
        *(line + "\n" for line in body),
        f"  return {export_symbol}({', '.join(names)});\n",
        "}\n",
    ])


def _reg_groups(solution: Solution, pipeline: Pipeline) -> list[tuple[str, RegGroup]]:
    groups = [(_reg_group_name(group, pipeline), group) for group in solution.reg_groups]
    groups.sort(key=lambda item: item[0])
    names = [name for name, _ in groups]
    if len(names) != len(set(names)):
        raise _link_error("solution contains duplicate reg_group names")
    return groups


def replace_declarator_identifier(decl: str, identifier: str, placeholder: str) -> str:
    """Replace only the expected function declarator, preserving all other text.

    Declaration identity normalization is owned by pipec2.decls, not this parser.
    """
    _, match = _declarator(decl, identifier)
    return decl[: match.start()] + placeholder + decl[match.end() :]


def generate_stub(
    solution: Solution,
    pipeline: Pipeline,
    stages: Iterable[StageManifest],
    expansions: Iterable[ProtocolExpansion],
    view_calls: Mapping[str, ViewCall] | None = None,
) -> str:
    """Generate the deterministic device binding/trampoline CUDA translation unit."""

    view_calls = view_calls or {}
    ordered_expansions = sorted(expansions, key=lambda item: item.channel)
    lines = ["// Generated by pipec2.stubgen; deterministic ABI v2 source.", "#include <cstdint>", ""]
    if any(item.callable is not None and re.search(r"\bpipelink_tensor_source\b", item.callable[1])
           for item in ordered_expansions):
        lines.append("#include <pipelink/tensor_source.h>")
    if view_calls:
        lines.append("#include <pipelink/host_view.h>")

    bindings = _bindings(solution, pipeline)
    for name in sorted(bindings):
        lines.append(
            f'extern "C" __device__ int {name}() {{ return {bindings[name]}; }}'
        )

    calls = [item for item in ordered_expansions if item.callable is not None]
    if bindings and (calls or solution.reg_groups):
        lines.append("")
    for index, expansion in enumerate(calls):
        if index:
            lines.append("")
        lines.extend(_trampoline(expansion.channel, expansion.callable, view_calls.get(expansion.channel))
                     .rstrip("\n").splitlines())

    groups = _reg_groups(solution, pipeline)
    if calls and groups:
        lines.append("")
    for index, (name, group) in enumerate(groups):
        if index:
            lines.append("")
        lines.extend(
            [
                'extern "C" __attribute__((used)) __forceinline__ __device__',
                f"void {reg_config_symbol(name)}() {{",
            ]
        )
        if group.op != "none":
            lines.append(
                f'  asm volatile("setmaxnreg.{group.op}.sync.aligned.u32 %0;" '
                f': : "n"({group.value}));'
            )
        lines.append("}")

    return "\n".join(lines).rstrip() + "\n"


__all__ = ["ViewCall", "call_symbol", "declaration_parts", "first_parameter_type", "generate_stub",
           "reg_config_symbol", "replace_declarator_identifier"]
