import pandas as pd
import polars as pl

from pytimetk.feature_store import FeatureStore


def test_feature_store_build_and_cache(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 12, 14]})

    store = FeatureStore(root_path=tmp_path)

    def signature_transform(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["double_value"] = result["value"] * 2
        return result

    store.register("signature", signature_transform, default_key_columns=("id",))

    first = store.build("signature", df, return_engine="pandas")
    assert not first.from_cache
    assert "double_value" in first.data.columns
    assert first.metadata.storage_backend == "local"
    assert first.metadata.artifact_uri.startswith(str(tmp_path))

    second = store.build("signature", df, return_engine="pandas")
    assert second.from_cache
    pd.testing.assert_frame_equal(first.data, second.data)

    catalog = store.list_feature_sets()
    assert len(catalog) == 1
    assert catalog.iloc[0]["name"] == "signature"

    lock_dir = tmp_path / ".locks"
    assert lock_dir.exists()
    assert not any(lock_dir.iterdir())


def test_feature_store_polars_accessor(tmp_path):
    df = pl.DataFrame({"id": [1, 2], "value": [5.0, 7.0]})

    accessor = df.tk.feature_store(root_path=tmp_path)

    def make_features(data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(
            (pl.col("value") - pl.col("value").mean()).alias("value_centered"),
        )

    accessor.register("centered", make_features, default_key_columns=("id",))

    result = accessor.build("centered")
    assert isinstance(result.data, pl.DataFrame)
    assert not result.from_cache
    assert "value_centered" in result.data.columns
    assert result.metadata.storage_backend == "local"

    cached = accessor.build("centered")
    assert cached.from_cache
    assert "value_centered" in cached.data.columns


def test_feature_store_assemble(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "value": [3.0, 4.0]})
    store = FeatureStore(root_path=tmp_path)

    def feat_one(data: pd.DataFrame) -> pd.DataFrame:
        res = data.copy()
        res["feat_one"] = res["value"] + 1
        return res[["id", "feat_one"]]

    def feat_two(data: pd.DataFrame) -> pd.DataFrame:
        res = data.copy()
        res["feat_two"] = res["value"] * 2
        return res[["id", "feat_two"]]

    store.register("one", feat_one, default_key_columns=("id",))
    store.register("two", feat_two, default_key_columns=("id",))

    store.build("one", df, return_engine="pandas")
    store.build("two", df, return_engine="pandas")

    assembled = store.assemble(["one", "two"], return_engine="pandas")
    assert set(assembled.data.columns) == {"id", "feat_one", "feat_two"}


def test_feature_store_remote_backend(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "value": [3.0, 4.0]})
    catalog_root = tmp_path / "catalog"
    artifact_uri = tmp_path.as_uri() + "/artifacts"

    store = FeatureStore(root_path=catalog_root, artifact_uri=artifact_uri)

    def make_features(data: pd.DataFrame) -> pd.DataFrame:
        res = data.copy()
        res["scaled"] = res["value"] / res["value"].sum()
        return res

    store.register("remote", make_features, default_key_columns=("id",))

    built = store.build("remote", df, return_engine="pandas")
    assert built.metadata.storage_backend == "pyarrow"
    assert built.metadata.storage_path.startswith(artifact_uri)

    loaded = store.load("remote", return_engine="pandas")
    pd.testing.assert_frame_equal(built.data, loaded.data)

    pl_df = pl.from_pandas(df)
    pl_accessor = pl_df.tk.feature_store(root_path=catalog_root, artifact_uri=artifact_uri)

    def polars_features(data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns((pl.col("value") * 10).alias("value_x10"))

    pl_accessor.register("remote_polars", polars_features, default_key_columns=("id",))
    polars_result = pl_accessor.build("remote_polars")
    assert polars_result.metadata.storage_backend == "pyarrow"
    assert polars_result.metadata.storage_path.startswith(artifact_uri)
