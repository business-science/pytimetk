#!/usr/bin/env python3
"""
Generate stub overlays for pandas so static analyzers can discover
the accessor methods registered by pytimetk.

The script inspects the pytimetk source for functions decorated with
`pandas_flavor.register_dataframe_method` and for assignments that
attach callables to `pd.core.groupby.generic.DataFrameGroupBy`.  It then
materialises `.pyi` files under `typings/pandas/...` (used for in-repo
development) and copies them into the stub package at
`stub-packages/pandas-stubs-pytimetk` ready for distribution.

Run this whenever accessor functions are added, renamed, or removed:

    poetry run python scripts/generate_pandas_stubs.py
"""

from __future__ import annotations

import ast
import re
from inspect import cleandoc, getdoc, isfunction, signature
from pathlib import Path
from typing import Dict, Iterable, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "pytimetk"
TYPINGS_ROOT = REPO_ROOT / "typings"
STUB_PACKAGE_ROOT = (
    REPO_ROOT
    / "stub-packages"
    / "pandas-stubs-pytimetk"
    / "pandas-stubs-pytimetk"
    / "pandas"
)

# Methods that return plotting objects rather than DataFrames.
PLOT_METHODS_DF: Set[str] = {
    "plot_anomalies",
    "plot_anomalies_cleaned",
    "plot_anomalies_decomp",
    "plot_correlation_funnel",
    "plot_timeseries",
}

PLOT_METHODS_GB: Set[str] = {
    "plot_anomalies",
    "plot_anomalies_cleaned",
    "plot_anomalies_decomp",
    "plot_timeseries",
}


def read_python_files(paths: Iterable[Path]) -> Dict[Path, str]:
    """Return a mapping of path -> text for the provided Python files."""
    return {path: path.read_text(encoding="utf-8") for path in paths}


def _clean_doc(doc: str | None) -> str:
    if not doc:
        return ""
    cleaned = cleandoc(doc)
    return cleaned.replace('"""', r'\"\"\"')


def collect_dataframe_methods(files: Dict[Path, str]) -> Dict[str, str]:
    """Extract function names decorated with register_dataframe_method."""
    result: Dict[str, str] = {}
    for path, text in files.items():
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                for deco in node.decorator_list:
                    if (
                        isinstance(deco, ast.Attribute)
                        and deco.attr == "register_dataframe_method"
                    ):
                        result[node.name] = _clean_doc(ast.get_docstring(node))
                        break
    return result


def collect_groupby_methods(files: Dict[Path, str]) -> Dict[str, str]:
    """Find methods assigned to DataFrameGroupBy.*."""
    pattern = re.compile(r"DataFrameGroupBy\.([a-zA-Z0-9_]+)")
    result: Dict[str, str] = {}
    for path, text in files.items():
        matches = pattern.findall(text)
        if not matches:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        doc_map: Dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                doc_map[node.name] = _clean_doc(ast.get_docstring(node))
        for name in matches:
            result.setdefault(name, doc_map.get(name, ""))
    return result


def collect_runtime_dataframe_methods() -> Dict[str, str]:
    """Inspect pandas.DataFrame to capture method docs."""
    import pandas as pd  # local import to keep script fast when unused

    result: Dict[str, str] = {}
    for name in dir(pd.DataFrame):
        if name.startswith("_"):
            continue
        attr = getattr(pd.DataFrame, name, None)
        if attr is None or not callable(attr):
            continue
        result[name] = _clean_doc(getdoc(attr))
    return result


def collect_runtime_groupby_methods() -> Dict[str, str]:
    from pandas.core.groupby.generic import DataFrameGroupBy

    result: Dict[str, str] = {}
    for name in dir(DataFrameGroupBy):
        if name.startswith("_"):
            continue
        attr = getattr(DataFrameGroupBy, name, None)
        if attr is None or not callable(attr):
            continue
        result[name] = _clean_doc(getdoc(attr))
    return result


def render_methods(
    class_name: str,
    methods: Dict[str, str],
    plot_methods: Set[str],
) -> str:
    """Render method signatures for either DataFrame or GroupBy classes."""
    base_import = (
        "from pandas import DataFrame as _BaseDataFrame"
        if class_name == "DataFrame"
        else "from pandas.core.groupby.generic import DataFrameGroupBy as _BaseGroupBy"
    )

    return_type = "DataFrame" if class_name == "DataFrame" else "pd.DataFrame"

    lines = [
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "import pandas as pd",
        base_import,
        "",
        f"class {class_name}(",
        f"    {'_BaseDataFrame' if class_name == 'DataFrame' else '_BaseGroupBy'},",
        "):",
    ]
    indent = "    "
    lines.append("")
    for name in sorted(methods):
        ret = "Any" if name in plot_methods else return_type
        lines.append(f"{indent}def {name}(self, *args: Any, **kwargs: Any) -> {ret}:")
        doc = methods.get(name, "")
        if doc:
            doc_lines = doc.splitlines()
            if len(doc_lines) == 1:
                lines.append(f'{indent}    """{doc_lines[0]}"""')
            else:
                lines.append(f'{indent}    """')
                for doc_line in doc_lines:
                    lines.append(f"{indent}    {doc_line}")
                lines.append(f'{indent}    """')
        lines.append(f"{indent}    ...")
    lines.append("")
    return "\n".join(lines)


def _write_to_base(base: Path, frame_stub: str, groupby_stub: str) -> None:
    (base / "core" / "groupby").mkdir(parents=True, exist_ok=True)
    (base / "core" / "frame.pyi").write_text(frame_stub, encoding="utf-8")
    (base / "core" / "groupby" / "generic.pyi").write_text(
        groupby_stub, encoding="utf-8"
    )


def write_stubs(
    dataframe_methods: Dict[str, str], groupby_methods: Dict[str, str]
) -> None:
    """Write the .pyi overlay files."""
    frame_stub = render_methods("DataFrame", dataframe_methods, PLOT_METHODS_DF)
    groupby_stub = render_methods(
        "DataFrameGroupBy", groupby_methods, PLOT_METHODS_GB
    )

    _write_to_base(TYPINGS_ROOT / "pandas", frame_stub, groupby_stub)

    if STUB_PACKAGE_ROOT.exists():
        _write_to_base(STUB_PACKAGE_ROOT, frame_stub, groupby_stub)


def main() -> None:
    files = read_python_files(SRC_ROOT.rglob("*.py"))
    dataframe_methods = collect_runtime_dataframe_methods()
    dataframe_methods.update(collect_dataframe_methods(files))

    groupby_methods = collect_runtime_groupby_methods()
    groupby_methods.update(collect_groupby_methods(files))

    if not dataframe_methods:
        raise SystemExit("No dataframe methods discovered; aborting.")

    write_stubs(dataframe_methods, groupby_methods)
    locations = [TYPINGS_ROOT]
    if STUB_PACKAGE_ROOT.exists():
        locations.append(STUB_PACKAGE_ROOT)

    print(
        "Generated stub overlays for "
        f"{len(dataframe_methods)} DataFrame methods and "
        f"{len(groupby_methods)} GroupBy methods in "
        + ", ".join(str(loc) for loc in locations)
    )


if __name__ == "__main__":
    main()
