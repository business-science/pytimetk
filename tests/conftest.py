import sys
import importlib
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def _stub_run_ray_tasks(monkeypatch):
    """
    Replace Ray executions with a deterministic local fallback so tests do not
    require the optional `ray` dependency.
    """

    def fake_run(func, args_list, **_):
        return [func(*args) for args in args_list]

    modules = [
        "pytimetk.utils.parallel_helpers",
        "pytimetk.feature_engineering.expanding_apply",
        "pytimetk.feature_engineering.expanding",
        "pytimetk.feature_engineering.rolling_apply",
        "pytimetk.feature_engineering.rolling",
        "pytimetk.core.future",
        "pytimetk.core.ts_features",
    ]

    for module_path in modules:
        module = importlib.import_module(module_path)
        if hasattr(module, "run_ray_tasks"):
            monkeypatch.setattr(module, "run_ray_tasks", fake_run)
