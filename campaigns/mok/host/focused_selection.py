"""Select unchanged upstream test-loop entries for a reproducible diagnostic."""
import os


def pytest_collection_modifyitems(items):
    selectors = {"shapes": os.environ["MOK_TEST_SHAPE"],
                 "mok_params": os.environ.get("MOK_TEST_PARAMETER", "Default"),
                 "swiglu_params": os.environ.get("MOK_TEST_SWIGLU", "Unclamped SwiGLU")}
    for module in {item.module for item in items if item.module.__name__.endswith(("test_functional", "test_ops"))}:
        for name, selected in selectors.items():
            original = getattr(module, name, None)
            if original is None:
                continue
            def focused(*args, original=original, selected=selected):
                rows = tuple(row for row in original(*args) if row[0] == selected)
                if len(rows) != 1:
                    raise ValueError(f"expected one upstream selection: {selected}")
                return rows
            setattr(module, name, focused)
