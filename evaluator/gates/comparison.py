"""Pure ordered SASS comparison for tl-scoring-v2.

Callers account for each integer operand delta and decide whether absolute
local traffic/CALL counts are diagnostic or part of their acceptance policy.
"""
from collections import Counter
import re

IMMEDIATE = re.compile(r"-?(?:0x[0-9a-fA-F]+|[0-9]+)\Z")
REGISTER = re.compile(r"(?<![A-Za-z0-9_])(?:[U]?R<[0-9]+>|[U]?P<[0-9]+>|URZ|RZ|UPT|PT)(?![A-Za-z0-9_])")


def instruction_parts(line):
    """Keep exact operand spans; commas nested in SASS operands do not split them."""
    head = re.match(r"(?:(@\S+)\s+)?([^\s;]+)", line)
    if head is None or not line.endswith(";"):
        raise ValueError(f"unsupported normalized instruction: {line!r}")
    spans = []
    start = head.end()
    brackets = []
    closing = {"[": "]", "{": "}", "(": ")"}
    for index in range(start, len(line)):
        char = line[index]
        if char in closing:
            brackets.append(closing[char])
        elif char in "]})":
            if not brackets or brackets.pop() != char:
                raise ValueError(f"unbalanced operand: {line!r}")
        elif char in ",;" and not brackets:
            left, right = start, index
            while left < right and line[left].isspace():
                left += 1
            while right > left and line[right - 1].isspace():
                right -= 1
            if left < right:
                spans.append((left, right))
            start = index + 1
    if brackets:
        raise ValueError(f"unbalanced operand: {line!r}")
    return head.group(1), head.group(2), spans


def compare_streams(control, linked):
    """Exact ordered comparison, exempting only individually located integer operands.

    The caller must separately account for every exempted delta. This is not an
    opcode-histogram comparison or a broadened normalization policy: predicates,
    modifiers, registers, punctuation and all non-integer operands stay exact.
    """
    checks = {
        "function_identity_order_equal": [name for name, _ in control] == [name for name, _ in linked],
        "instruction_counts_equal": [len(body) for _, body in control] == [len(body) for _, body in linked],
        "opcode_sequence_including_modifiers_equal": True,
        "predicates_equal": True,
        "register_relationships_equal_under_frozen_renaming": True,
        "all_non_enumerated_instruction_text_equal": True,
    }
    deltas = []
    normalized_index = 0
    memory = {"control": [], "linked": []}
    for label, sections in (("control", control), ("linked", linked)):
        for function, body in sections:
            opcodes = Counter(instruction_parts(line)[1].split(".")[0] for line in body)
            memory[label].append({
                "function": function,
                "exact_opcode_counts": {op: opcodes[op] for op in ("STL", "LDL", "STS", "LDS", "CALL")},
                "counts": {op: sum(count for name, count in opcodes.items() if name.startswith(op))
                           for op in ("STL", "LDL", "STS", "LDS", "CALL")},
            })
    checks["memory_op_class_counts_equal"] = memory["control"] == memory["linked"]
    checks["no_local_traffic_or_calls"] = all(
        row["counts"][op] == 0
        for rows in memory.values() for row in rows for op in ("STL", "LDL", "CALL")
    )
    for function_index, ((function, left), (_, right)) in enumerate(zip(control, linked)):
        normalized_index += 1  # The frozen stream includes each function header.
        for index, (old, new) in enumerate(zip(left, right)):
            old_pred, old_op, old_spans = instruction_parts(old)
            new_pred, new_op, new_spans = instruction_parts(new)
            checks["opcode_sequence_including_modifiers_equal"] &= old_op == new_op
            checks["predicates_equal"] &= old_pred == new_pred
            checks["register_relationships_equal_under_frozen_renaming"] &= REGISTER.findall(old) == REGISTER.findall(new)
            replacements = []
            for operand_index, ((a, b), (c, d)) in enumerate(zip(old_spans, new_spans)):
                old_value, new_value = old[a:b], new[c:d]
                if old_value == new_value:
                    continue
                integer_encoding = bool(IMMEDIATE.fullmatch(old_value) and IMMEDIATE.fullmatch(new_value))
                deltas.append({
                    "function": function, "function_index_zero_based": function_index,
                    "instruction_index_zero_based": index,
                    "normalized_stream_index_zero_based": normalized_index + index,
                    "control_instruction": old, "linked_instruction": new,
                    "operand_index_zero_based": operand_index,
                    "operand_location": {
                        "description": "comma-separated operand, destination included in indexing",
                        "control_character_span_zero_based_half_open": [a, b],
                        "linked_character_span_zero_based_half_open": [c, d],
                    },
                    "old_value": old_value, "new_value": new_value,
                    "integer_operand_encoding": integer_encoding,
                })
                if integer_encoding:
                    replacements.append((c, d, old_value))
            restored = new
            for a, b, old_value in reversed(replacements):
                restored = restored[:a] + old_value + restored[b:]
            checks["all_non_enumerated_instruction_text_equal"] &= restored == old
        normalized_index += len(left)
    return {
        "checks": checks, "seam_structure_clean": all(checks.values()),
        "instruction_counts": {label: sum(len(body) for _, body in sections)
                               for label, sections in (("control", control), ("linked", linked))},
        "memory_op_class_counts": memory, "deltas": deltas,
    }
