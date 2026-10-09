from __future__ import annotations

import hashlib as _hashlib


def normalize_declaration(text: str) -> str:
    """Ignore comments and whitespace runs without changing declaration tokens or literals."""
    pieces: list[str] = []
    index = 0
    space = False
    brace_depth = paren_depth = 0
    declaration_end = False
    while index < len(text):
        char = text[index]
        if char.isspace():
            space = True
            index += 1
            continue
        if char == "#":
            if pieces and pieces[-1] != "\n":
                pieces.append("\n")
            end = text.find("\n", index)
            if end < 0:
                end = len(text)
            pieces.extend((text[index:end].strip(), "\n"))
            index = end
            space = declaration_end = False
            continue
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            index = len(text) if end < 0 else end + 1
            space = True
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
            space = True
            continue
        if pieces and declaration_end and char != ";":
            pieces.append("\n")
        elif space and pieces and pieces[-1] != "\n":
            pieces.append(" ")
        declaration_end = False
        space = False
        start = index
        if text.startswith('R"', index):
            opening = text.find("(", index + 2, index + 19)
            if opening >= 0:
                delimiter = text[index + 2 : opening]
                if not any(item.isspace() or item in "()\\" for item in delimiter):
                    closing = ")" + delimiter + '"'
                    end = text.find(closing, opening + 1)
                    index = len(text) if end < 0 else end + len(closing)
                    pieces.append(text[start:index])
                    continue
        index += 1
        if char in {'"', "'"}:
            while index < len(text):
                current = text[index]
                index += 1
                if current == "\\" and index < len(text):
                    index += 1
                elif current == char:
                    break
        pieces.append(text[start:index])
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            declaration_end = brace_depth == paren_depth == 0
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == ";":
            declaration_end = brace_depth == paren_depth == 0
    return "".join(pieces).rstrip()


def declaration_digest(text: str) -> str:
    """Return the SHA-256 identity of a normalized UTF-8 declaration."""
    return _hashlib.sha256(normalize_declaration(text).encode("utf-8")).hexdigest()


__all__ = ["normalize_declaration", "declaration_digest"]
