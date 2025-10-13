from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import polars as pl

from .store import FeatureSetResult, FeatureStore


__all__ = [
    "build_features_with_mlflow",
    "log_feature_metadata_to_mlflow",
    "load_features_from_mlflow",
]


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - handled through tests
        raise ImportError(
            "MLflow integration requires the `mlflow` package. "
            "Install it with `pip install mlflow` or add it to your environment."
        ) from exc
    return mlflow


def _require_active_run(mlflow_module, run=None):
    if run is not None:
        return run
    active = mlflow_module.active_run()
    if active is None:
        raise RuntimeError(
            "No active MLflow run found. Start a run with `mlflow.start_run()` "
            "before invoking the feature store MLflow helpers."
        )
    return active


def _parameter_prefix(prefix: Optional[str], name: str) -> str:
    if prefix is None or prefix == "":
        return name
    return f"{prefix.rstrip('_')}_{name}"


def build_features_with_mlflow(
    store: FeatureStore,
    name: str,
    data: Union[pd.DataFrame, pl.DataFrame],
    *,
    params: Optional[Mapping[str, Any]] = None,
    refresh: bool = False,
    version: Optional[str] = None,
    key_columns: Optional[Sequence[str]] = None,
    storage_format: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    return_engine: str = "auto",
    writer_options: Optional[Mapping[str, Any]] = None,
    params_prefix: Optional[str] = None,
    metadata_artifact_path: str = "feature_store",
    log_metadata_artifact: bool = True,
    log_feature_artifact: Optional[bool] = None,
    run=None,
    **build_kwargs: Any,
) -> FeatureSetResult:
    """
    Build (or reuse) a feature set while recording versioning metadata to MLflow.
    """

    mlflow = _import_mlflow()
    active_run = _require_active_run(mlflow, run=run)

    result = store.build(
        name=name,
        data=data,
        params=params,
        refresh=refresh,
        version=version,
        key_columns=key_columns,
        storage_format=storage_format,
        tags=tags,
        extra_metadata=extra_metadata,
        return_engine=return_engine,
        writer_options=writer_options,
        **build_kwargs,
    )

    log_feature_artifact = (
        result.metadata.storage_backend == "local"
        if log_feature_artifact is None
        else log_feature_artifact
    )

    log_feature_metadata_to_mlflow(
        result=result,
        name=name,
        params_prefix=params_prefix,
        metadata_artifact_path=metadata_artifact_path,
        log_metadata_artifact=log_metadata_artifact,
        log_feature_artifact=log_feature_artifact,
    )

    # Expose refresh/cache outcome for downstream logging if desired.
    mlflow.log_metric(
        _parameter_prefix(params_prefix, f"{name}_cache_hit"),
        1.0 if result.from_cache else 0.0,
    )

    return result


def log_feature_metadata_to_mlflow(
    *,
    result: FeatureSetResult,
    name: str,
    params_prefix: Optional[str] = None,
    metadata_artifact_path: str = "feature_store",
    log_metadata_artifact: bool = True,
    log_feature_artifact: bool = False,
) -> None:
    """
    Log feature metadata and optional artifacts for a previously built feature set.
    """

    mlflow = _import_mlflow()
    _require_active_run(mlflow)

    meta = result.metadata
    prefix_name = _parameter_prefix(params_prefix, name)

    mlflow.log_param(f"{prefix_name}_feature_version", meta.version)
    mlflow.log_param(f"{prefix_name}_cache_key", meta.cache_key)
    mlflow.log_param(f"{prefix_name}_storage_backend", meta.storage_backend)
    mlflow.log_param(f"{prefix_name}_artifact_uri", meta.artifact_uri)
    if meta.key_columns:
        mlflow.log_param(
            f"{prefix_name}_key_columns",
            ",".join(meta.key_columns),
        )

    # Always log a compact set of metadata as JSON when requested.
    if log_metadata_artifact:
        artifact_name = os.path.join(
            metadata_artifact_path.rstrip("/"),
            f"{prefix_name}_metadata.json",
        )
        mlflow.log_dict(asdict(meta), artifact_name)

    # Optional: include the concrete artifact if locally accessible.
    if log_feature_artifact and meta.storage_backend == "local":
        if os.path.exists(meta.storage_path):
            mlflow.log_artifact(
                meta.storage_path,
                artifact_path=os.path.join(
                    metadata_artifact_path.rstrip("/"),
                    f"{prefix_name}_data",
                ),
            )


def load_features_from_mlflow(
    store: FeatureStore,
    name: str,
    *,
    run_id: Optional[str] = None,
    params_prefix: Optional[str] = None,
    version_param: Optional[str] = None,
    return_engine: str = "auto",
    strict: bool = True,
) -> FeatureSetResult:
    """
    Load a feature set using the version recorded in an MLflow run.
    """

    mlflow = _import_mlflow()

    if run_id is None:
        active = mlflow.active_run()
        if active is None:
            raise RuntimeError(
                "No run_id provided and no active MLflow run found. "
                "Pass an explicit run_id when loading outside a tracking context."
            )
        run_id = active.info.run_id

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    param_key = version_param or f"{_parameter_prefix(params_prefix, name)}_feature_version"
    version_value = run.data.params.get(param_key)
    if version_value is None:
        message = (
            f"MLflow run '{run_id}' did not log a parameter '{param_key}'. "
            "Ensure that `build_features_with_mlflow` (or the logging helper) "
            "was invoked during training."
        )
        if strict:
            raise KeyError(message)
        return store.load(name, return_engine=return_engine)

    return store.load(name, version=version_value, return_engine=return_engine)
