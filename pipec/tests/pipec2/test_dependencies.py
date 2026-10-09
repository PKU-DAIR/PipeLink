import ast
from pathlib import Path


def test_pipec2_has_no_pipec_mini_dependency() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "pipec2"
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            else:
                continue
            if any(module.split(".", 1)[0] == "pipec_mini" for module in modules):
                offenders.append(f"{path.relative_to(source_root)}:{node.lineno}")
    assert not offenders, f"pipec2 imports the historical implementation: {offenders}"
