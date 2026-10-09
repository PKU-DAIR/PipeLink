"""Captured facts, reflected types and explicit declarations/conventions."""
import ast
import re
from collections.abc import Mapping


class AnnotationError(ValueError):
    def __init__(self, field, detail="missing compiler-fact source"):
        self.field = field
        super().__init__(f"P-ANNOTATION {field}: {detail}")


def fact(facts, path):
    """Resolve a JSON-pointer path; dots in compiler attribute names are literal."""
    value = facts
    try:
        for part in path.strip("/").split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise AnnotationError(path) from error
    return value


def symbol_prefix(stage):
    return stage.stage_id.replace(".", "_").replace("-", "_").lower() + "_"


class Ledger:
    def __init__(self):
        self.sources = {}

    def record(self, field, source, *, kind):
        if kind not in ("captured", "reflected", "declared", "convention", "contract"):
            raise AnnotationError(field, "unknown source class")
        if not isinstance(source, str) or not source.strip() or re.fullmatch(r"[+-]?\d+", source):
            raise AnnotationError(field, "expected a named fact, declaration, convention or C++ expression")
        entry = {"class": kind, "source": source}
        if field in self.sources and self.sources[field] != entry:
            raise AnnotationError(field, "conflicting sources")
        self.sources[field] = entry
        return source

    def require(self, field):
        if field not in self.sources:
            raise AnnotationError(field)
        return self.sources[field]["source"]

    def alias(self, field, source_field):
        source = self.require(source_field)
        return self.record(field, source, kind=self.sources[source_field]["class"])

    def reference(self, field, facts, path, namespace):
        if type(fact(facts, path)) not in (int, bool):
            raise AnnotationError(field, f"not an integer fact: {path}")
        self.record(field, "facts.json#" + path, kind="captured")
        self.require(field)
        return namespace + "::" + FactsHeader.symbol(path)


class FactsHeader:
    @staticmethod
    def symbol(path):
        return "tir_" + re.sub(r"[^a-zA-Z0-9_]", "_", path.strip("/"))

    @staticmethod
    def render(facts: dict, namespace: str) -> str:
        if not re.fullmatch(r"[a-zA-Z_]\w*", namespace):
            raise AnnotationError("namespace", "invalid C++ identifier")
        lines = ["#pragma once", "#include <array>", f"namespace {namespace} {{"]
        seen = set()

        def scalars(path, value):
            if isinstance(value, (int, bool)):
                name = FactsHeader.symbol(path)
                if name in seen:
                    raise AnnotationError(path, "colliding C++ fact identifiers")
                seen.add(name)
                lines.append(f"inline constexpr long long {name} = {int(value)};")
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    scalars(path + "/" + str(key).replace("~", "~0").replace("/", "~1"), item)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    scalars(path + "/" + str(index), item)

        scalars("", facts)
        # These internal maps are provenance, not WS endpoint placement. In
        # particular, keep role-local and CTA-thread coordinates distinct.
        for name, layout in facts.get("layouts", {}).items():
            variables = {name: f"x{index}" for index, name in enumerate(layout["forward_vars"])}
            params = ", ".join(f"int {name}" for name in variables.values())
            exprs = list(layout["forward_index"])
            if "thread" in layout:
                exprs = [layout["thread"], f"({layout['thread']}) + {layout['thread_min']}", *exprs]
            fn = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            for index, expression in enumerate(exprs):
                body = _cpp_expr(ast.parse(expression, mode="eval").body, variables)
                lines.append(f"constexpr int {fn}_{index}({params}) {{ return {body}; }}")
        return "\n".join([*lines, "}", ""])


def _cpp_expr(node, variables):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return str(node.value)
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "/", ast.Mod: "%"}
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return f"({_cpp_expr(node.left, variables)} {operators[type(node.op)]} {_cpp_expr(node.right, variables)})"
    raise AnnotationError("layouts", f"unsupported captured map expression: {ast.dump(node)}")


def emit_annotations(stage, facts, ledger):
    if all(role.branch in ("then", "else") for role in stage.roles.values()):
        from .ws import emit_annotations as emit_ws
        return emit_ws(stage, facts, ledger)
    from .f1 import emit_annotations as emit_f1
    return emit_f1(stage, facts, ledger)
