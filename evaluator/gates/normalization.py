"""Strict SASS extraction and one-to-one first-use register normalization.

Every opcode and every decimal/hexadecimal operand remains byte-exact. Only
address and scheduler-control dump fields and register names are normalized.
"""
from __future__ import annotations

import re

FUNCTION = re.compile(r"^\s*Function\s*:\s*(.+?)\s*$")
INSTRUCTION = re.compile(r"/\*[0-9a-f]+\*/\s+(.*?)\s+/\*")
REGISTER = re.compile(
    r"(?<![A-Za-z0-9_])(URZ|RZ|UPT|PT|UR\d+|R\d+|UP\d+|P\d+)(?![A-Za-z0-9_])"
)


def sections(text: str) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    name: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        function = FUNCTION.match(line)
        if function:
            if name is not None:
                result.append((name, body))
            name = function.group(1)
            body = []
            continue
        instruction = INSTRUCTION.search(line)
        if name is not None and instruction is not None:
            body.append(instruction.group(1).strip())
    if name is not None:
        result.append((name, body))
    return result


def canonicalize(body: list[str]) -> list[str]:
    registers: dict[str, str] = {}
    next_ids = {"R": 0, "UR": 0, "P": 0, "UP": 0}

    def rename(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in {"RZ", "URZ", "PT", "UPT"}:
            return token
        if token not in registers:
            kind = "UR" if token.startswith("UR") else "UP" if token.startswith("UP") else token[0]
            registers[token] = f"{kind}<{next_ids[kind]}>"
            next_ids[kind] += 1
        return registers[token]

    return [REGISTER.sub(rename, instruction) for instruction in body]
