"""
Utilities to help editors locate pytimetk's pandas stub overlays.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Optional


def get_typing_stub_path() -> Optional[pathlib.Path]:
    """
    Return the path to the installed pandas stub overlay if available.

    This looks for the ``pandas-stubs-pytimetk`` wheel first. If that
    package is not installed but the stubs have been copied directly into
    the pandas package (e.g. during development), it falls back to the
    pandas module location.
    """

    spec = importlib.util.find_spec("pandas_stubs_pytimetk")
    if spec and spec.origin:
        # Stub-only package: spec.origin points at .../pandas/__init__.pyi
        return pathlib.Path(spec.origin).parent

    spec = importlib.util.find_spec("pandas.core.frame")
    if spec and spec.origin:
        # pandas/core/frame.py -> go back to the top-level pandas package
        return pathlib.Path(spec.origin).resolve().parents[2]

    return None


def _cli_vscode_config() -> int:
    """
    Print guidance for configuring VS Code / Pyright with the stub path.
    """

    stub_path = get_typing_stub_path()
    if not stub_path:
        print("pytimetk: pandas stub overlay not found.", file=sys.stderr)
        return 1

    print(
        "Add the following entry to your VS Code settings.json (or pyrightconfig.json):\n\n"
        '    "python.analysis.typeshedPaths": [\n'
        f'        "{stub_path.as_posix()}"\n'
        "    ]\n\n"
        "Then restart the Python language server."
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point: ``python -m pytimetk._typing_support [command]``.
    """

    cmd = argv[0] if argv else None
    if cmd in (None, "path"):
        stub_path = get_typing_stub_path()
        if not stub_path:
            print("pytimetk: pandas stub overlay not found.", file=sys.stderr)
            return 1
        print(stub_path)
        return 0

    if cmd == "vscode-config":
        return _cli_vscode_config()

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
