import pandas as pd
import pytest

from pytimetk.feature_store import (
    FeatureStore,
    build_features_with_mlflow,
    load_features_from_mlflow,
)

try:
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore


pytestmark = pytest.mark.skipif(
    mlflow is None,
    reason="MLflow is not installed; skip integration tests.",
)


@pytest.fixture()
def tracking_uri(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    return tracking_dir


def test_build_with_mlflow_logs_metadata(tmp_path, tracking_uri):
    store = FeatureStore(root_path=tmp_path / "catalog")

    def make_features(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["value_squared"] = result["value"] ** 2
        return result

    store.register("example", make_features, default_key_columns=("id",))

    df = pd.DataFrame({"id": [1, 2, 3], "value": [2.0, 3.0, 4.0]})

    with mlflow.start_run() as run:
        result = build_features_with_mlflow(
            store,
            "example",
            df,
            return_engine="pandas",
            log_feature_artifact=True,
        )
        run_id = run.info.run_id

    logged_run = mlflow.get_run(run_id)

    params = logged_run.data.params
    assert params["example_feature_version"] == result.metadata.version
    assert params["example_storage_backend"] == result.metadata.storage_backend
    assert params["example_cache_key"] == result.metadata.cache_key

    loaded = load_features_from_mlflow(
        store,
        "example",
        run_id=run_id,
        return_engine="pandas",
    )

    pd.testing.assert_frame_equal(result.data, loaded.data)
