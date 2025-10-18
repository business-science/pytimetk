from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import posixpath
from pathlib import Path
import time
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    BinaryIO,
)

import pandas as pd
import polars as pl

import pyarrow.fs as pa_fs

from pytimetk.utils.dataframe_ops import identify_frame_kind, resolve_pandas_groupby_frame

_FEATURE_STORE_BETA_MESSAGE = (
    "Feature Store & Caching is currently in beta. APIs and storage formats may change before general availability."
)


def _warn_feature_store_beta() -> None:
    warnings.warn(_FEATURE_STORE_BETA_MESSAGE, UserWarning, stacklevel=2)

Jsonable = Union[str, int, float, bool, None, Mapping[str, Any], Sequence[Any]]
TransformCallable = Callable[[Any], Any]


@dataclass(frozen=True)
class FeatureSetMetadata:
    """
    Immutable metadata describing a single materialised feature set.
    """

    name: str
    version: str
    cache_key: str
    storage_path: str
    storage_format: str
    storage_backend: str
    artifact_uri: str
    created_at: str
    data_fingerprint: str
    transform_fingerprint: str
    transform_module: str
    transform_name: str
    transform_kwargs: Mapping[str, Any]
    pytimetk_version: str
    package_versions: Mapping[str, str]
    tags: Tuple[str, ...] = field(default_factory=tuple)
    description: Optional[str] = None
    key_columns: Optional[Tuple[str, ...]] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSetResult:
    """
    Wrapper returned by the feature store for build/load operations.
    """

    data: Union[pd.DataFrame, pl.DataFrame]
    metadata: FeatureSetMetadata
    from_cache: bool


@dataclass
class RegisteredTransform:
    name: str
    function: TransformCallable
    default_kwargs: MutableMapping[str, Any]
    description: Optional[str]
    tags: Tuple[str, ...]
    default_key_columns: Optional[Tuple[str, ...]]
    extra_metadata: Mapping[str, Any]

    def fingerprint(self, runtime_kwargs: Mapping[str, Any]) -> str:
        hasher = sha256()
        hasher.update(self._callable_signature().encode())
        kwargs_payload = _normalise_for_hash(runtime_kwargs)
        hasher.update(json.dumps(kwargs_payload, sort_keys=True).encode())
        return hasher.hexdigest()

    def _callable_signature(self) -> str:
        module = getattr(self.function, "__module__", "unknown")
        qualname = getattr(self.function, "__qualname__", repr(self.function))
        signature = f"{module}:{qualname}"
        try:
            source = inspect.getsource(self.function)
        except (OSError, TypeError):
            source = repr(self.function)
        return f"{signature}:{source}"


class FileLockManager:
    """
    Lightweight file-based locking to coordinate concurrent writers.
    """

    def __init__(
        self,
        lock_dir: Path,
        *,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
    ) -> None:
        self.lock_dir = lock_dir
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquire(self, key: str):
        if not key:
            raise ValueError("Lock key must be a non-empty string.")

        lock_path = self.lock_dir / f"{key}.lock"
        start_time = time.time()

        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if (time.time() - start_time) >= self.timeout:
                    raise TimeoutError(
                        f"Timeout waiting for lock on '{key}'. Existing lock file: {lock_path}"
                    )
                time.sleep(self.poll_interval)

        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


class ArtifactBackend:
    name: str = "base"

    def __init__(self, artifact_uri: str) -> None:
        self.artifact_uri = artifact_uri.rstrip("/")

    def write(
        self,
        frame: pl.DataFrame,
        *,
        name: str,
        version: str,
        storage_format: str,
        writer_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    def read(self, storage_path: str, storage_format: str) -> pl.DataFrame:
        raise NotImplementedError

    def remove(self, storage_path: str) -> None:
        raise NotImplementedError

    def exists(self, storage_path: str) -> bool:
        raise NotImplementedError


class LocalArtifactBackend(ArtifactBackend):
    name = "local"

    def __init__(self, artifact_path: Path) -> None:
        self.base_path = artifact_path.expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        super().__init__(str(self.base_path))

    def write(
        self,
        frame: pl.DataFrame,
        *,
        name: str,
        version: str,
        storage_format: str,
        writer_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        version_dir = self.base_path / name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = version_dir / f"features.{_extension_for_format(storage_format)}"
        _write_frame(frame, artifact_path, storage_format, writer_options)
        return str(artifact_path)

    def read(self, storage_path: str, storage_format: str) -> pl.DataFrame:
        return _read_frame(Path(storage_path), storage_format)

    def remove(self, storage_path: str) -> None:
        path = Path(storage_path)
        if path.exists():
            path.unlink()
        parent = path.parent
        while parent != self.base_path and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    def exists(self, storage_path: str) -> bool:
        return Path(storage_path).exists()


class PyArrowArtifactBackend(ArtifactBackend):
    name = "pyarrow"

    def __init__(self, artifact_uri: str) -> None:
        fs, base_path = pa_fs.FileSystem.from_uri(artifact_uri)
        if base_path:
            fs = pa_fs.SubTreeFileSystem(base_path, fs)
        self._fs = fs
        super().__init__(artifact_uri)
        self._clean_base = self.artifact_uri.rstrip("/")

    def write(
        self,
        frame: pl.DataFrame,
        *,
        name: str,
        version: str,
        storage_format: str,
        writer_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        relative_path = posixpath.join(name, version, f"features.{_extension_for_format(storage_format)}")
        directory = posixpath.dirname(relative_path)
        if directory:
            self._fs.create_dir(directory, recursive=True)
        with self._fs.open_output_stream(relative_path) as sink:
            _write_frame(frame, sink, storage_format, writer_options)
        return _join_uri(self.artifact_uri, relative_path)

    def read(self, storage_path: str, storage_format: str) -> pl.DataFrame:
        relative_path = self._relative_path(storage_path)
        with self._fs.open_input_stream(relative_path) as source:
            return _read_frame(source, storage_format)

    def remove(self, storage_path: str) -> None:
        relative_path = self._relative_path(storage_path)
        try:
            self._fs.delete_file(relative_path)
        except FileNotFoundError:
            return

    def exists(self, storage_path: str) -> bool:
        relative_path = self._relative_path(storage_path)
        info = self._fs.get_file_info(relative_path)
        return info.type != pa_fs.FileType.NotFound

    def _relative_path(self, storage_path: str) -> str:
        base = self.artifact_uri
        if storage_path.startswith(base.rstrip("/") + "/"):
            return storage_path[len(base.rstrip("/") + "/") :]
        if storage_path == base.rstrip("/"):
            return ""
        raise ValueError(
            f"Storage path '{storage_path}' is not within artifact root '{base}'."
        )
class FeatureStore:
    """
    Lightweight on-disk feature store with metadata cataloguing.
    """

    def __init__(
        self,
        root_path: Optional[Union[str, os.PathLike[str]]] = None,
        catalog_filename: str = "catalog.json",
        default_storage_format: str = "parquet",
        artifact_uri: Optional[str] = None,
        artifact_backend: Optional[ArtifactBackend] = None,
        enable_locking: bool = True,
        lock_timeout: float = 30.0,
        lock_poll_interval: float = 0.2,
    ) -> None:
        _warn_feature_store_beta()
        self.root_path = _resolve_root_path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root_path / catalog_filename
        self.default_storage_format = default_storage_format
        self._registry: Dict[str, RegisteredTransform] = {}
        self._catalog: List[Dict[str, Any]] = self._load_catalog()
        artifact_uri = artifact_uri or str(self.root_path)
        self._artifact_backend = artifact_backend or _create_artifact_backend(artifact_uri)
        self.enable_locking = enable_locking
        self._lock_manager = (
            FileLockManager(
                self.root_path / ".locks",
                timeout=lock_timeout,
                poll_interval=lock_poll_interval,
            )
            if enable_locking
            else None
        )

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(
        self,
        name: str,
        transform: TransformCallable,
        *,
        default_kwargs: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        default_key_columns: Optional[Sequence[str]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "FeatureStore":
        if not name or not isinstance(name, str):
            raise ValueError("`name` must be a non-empty string.")
        if not callable(transform):
            raise TypeError("`transform` must be callable.")

        normalised_tags = tuple(sorted({str(tag) for tag in (tags or ())}))
        default_kwargs = dict(default_kwargs or {})
        default_key_columns = (
            tuple(default_key_columns) if default_key_columns else None
        )
        extra_metadata = dict(extra_metadata or {})

        self._registry[name] = RegisteredTransform(
            name=name,
            function=transform,
            default_kwargs=default_kwargs,
            description=description,
            tags=normalised_tags,
            default_key_columns=default_key_columns,
            extra_metadata=extra_metadata,
        )
        return self

    # ------------------------------------------------------------------ #
    # Build / Load
    # ------------------------------------------------------------------ #
    def build(
        self,
        name: str,
        data: Union[pd.DataFrame, pl.DataFrame],
        *,
        params: Optional[Mapping[str, Any]] = None,
        refresh: bool = False,
        version: Optional[str] = None,
        key_columns: Optional[Sequence[str]] = None,
        storage_format: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        return_engine: str = "auto",
        writer_options: Optional[Mapping[str, Any]] = None,
    ) -> FeatureSetResult:
        transform = self._get_registered_transform(name)

        combined_kwargs = {**transform.default_kwargs, **(params or {})}
        data_fingerprint = _dataframe_fingerprint(data)
        transform_fingerprint = transform.fingerprint(combined_kwargs)
        cache_key = _make_cache_key(name, data_fingerprint, transform_fingerprint)

        entry = self._find_entry(name=name, cache_key=cache_key, version=version)
        if entry and not refresh:
            return self._load_entry(
                entry,
                return_engine=_resolve_return_engine(return_engine, data),
                from_cache=True,
            )

        lock_cm = self._lock_manager.acquire(name) if self._lock_manager else nullcontext()
        with lock_cm:
            entry = self._find_entry(name=name, cache_key=cache_key, version=version)
            if entry and not refresh:
                return self._load_entry(
                    entry,
                    return_engine=_resolve_return_engine(return_engine, data),
                    from_cache=True,
                )

            result_frame = transform.function(data, **combined_kwargs)
            storage_df = _ensure_polars_df(result_frame)

            storage_format_resolved = (
                storage_format
                or (entry["storage_format"] if entry else None)
                or self.default_storage_format
            )
            version_id = version or (
                entry["version"] if entry and not refresh else _derive_version(cache_key)
            )

            storage_path = self._artifact_backend.write(
                storage_df,
                name=name,
                version=version_id,
                storage_format=storage_format_resolved,
                writer_options=writer_options,
            )

            combined_tags = tuple(sorted({*transform.tags, *(tags or ())}))
            resolved_key_columns = tuple(key_columns or transform.default_key_columns or ())
            metadata_map: Dict[str, Any] = {
                "name": name,
                "version": version_id,
                "cache_key": cache_key,
                "storage_path": storage_path,
                "storage_format": storage_format_resolved,
                "storage_backend": self._artifact_backend.name,
                "artifact_uri": self._artifact_backend.artifact_uri,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_fingerprint": data_fingerprint,
                "transform_fingerprint": transform_fingerprint,
                "transform_module": getattr(transform.function, "__module__", "unknown"),
                "transform_name": getattr(transform.function, "__qualname__", repr(transform.function)),
                "transform_kwargs": combined_kwargs,
                "pytimetk_version": _pytimetk_version(),
                "package_versions": _package_versions(),
                "tags": combined_tags,
                "description": transform.description,
                "key_columns": resolved_key_columns or None,
                "row_count": storage_df.height,
                "column_count": storage_df.width,
                "extra_metadata": {
                    **transform.extra_metadata,
                    **dict(extra_metadata or {}),
                },
            }

            if entry:
                self._catalog.remove(entry)
            self._catalog.append(metadata_map)
            self._persist_catalog()

            metadata = _metadata_from_dict(metadata_map)
            return FeatureSetResult(
                data=_coerce_return_frame(
                    storage_df,
                    return_engine=_resolve_return_engine(return_engine, data),
                    base_input=data,
                ),
                metadata=metadata,
                from_cache=False,
            )

    def load(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        return_engine: str = "polars",
    ) -> FeatureSetResult:
        entry = self._find_entry(name=name, version=version)
        if not entry:
            raise KeyError(f"No feature set named '{name}' found (version={version or 'latest'}).")
        return self._load_entry(entry, return_engine=return_engine, from_cache=True)

    # ------------------------------------------------------------------ #
    # Catalog inspection
    # ------------------------------------------------------------------ #
    def list_feature_sets(self, name: Optional[str] = None) -> pd.DataFrame:
        rows = [
            _metadata_from_dict(entry).__dict__
            for entry in self._catalog
            if name is None or entry["name"] == name
        ]
        if not rows:
            return pd.DataFrame(columns=FeatureSetMetadata.__dataclass_fields__.keys())
        return pd.DataFrame(rows).sort_values(["name", "created_at"])

    def describe(self, name: str, version: Optional[str] = None) -> FeatureSetMetadata:
        entry = self._find_entry(name=name, version=version)
        if not entry:
            raise KeyError(f"No feature set named '{name}' found (version={version or 'latest'}).")
        return _metadata_from_dict(entry)

    def drop(self, name: str, version: Optional[str] = None, *, delete_artifact: bool = True) -> None:
        entry = self._find_entry(name=name, version=version)
        if not entry:
            raise KeyError(f"No feature set named '{name}' found (version={version or 'latest'}).")
        self._catalog.remove(entry)
        self._persist_catalog()

        if delete_artifact:
            backend = _backend_from_metadata(entry, self._artifact_backend)
            backend.remove(entry["storage_path"])

    def assemble(
        self,
        feature_specs: Sequence[Union[str, Tuple[str, str]]],
        *,
        join_keys: Optional[Sequence[str]] = None,
        how: str = "left",
        return_engine: str = "polars",
    ) -> FeatureSetResult:
        if not feature_specs:
            raise ValueError("`feature_specs` must contain at least one feature set identifier.")

        loaded: List[FeatureSetResult] = []
        for spec in feature_specs:
            name, version = _parse_feature_spec(spec)
            loaded.append(self.load(name, version=version, return_engine="polars"))

        join_columns = tuple(join_keys or loaded[0].metadata.key_columns or ())
        if not join_columns:
            raise ValueError("`join_keys` must be provided when feature metadata does not specify default keys.")

        combined = loaded[0].data
        for result in loaded[1:]:
            suffix = f"_{result.metadata.version}"
            combined = combined.join(
                result.data,
                on=list(join_columns),
                how=how,
                suffix=suffix,
            )

        merged_metadata = FeatureSetMetadata(
            name="+".join(result.metadata.name for result in loaded),
            version=datetime.now(timezone.utc).strftime("assembled-%Y%m%d%H%M%S"),
            cache_key="",
            storage_path="",
            storage_format="",
            storage_backend="assembled",
            artifact_uri="",
            created_at=datetime.now(timezone.utc).isoformat(),
            data_fingerprint="",
            transform_fingerprint="",
            transform_module="",
            transform_name="",
            transform_kwargs={},
            pytimetk_version=_pytimetk_version(),
            package_versions=_package_versions(),
            tags=tuple(),
            description="Assembled feature set",
            key_columns=join_columns,
            row_count=combined.height,
            column_count=combined.width,
            extra_metadata={},
        )

        return FeatureSetResult(
            data=_coerce_return_frame(combined, return_engine=return_engine),
            metadata=merged_metadata,
            from_cache=True,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_registered_transform(self, name: str) -> RegisteredTransform:
        if name not in self._registry:
            raise KeyError(
                f"Transform '{name}' has not been registered. "
                "Call `FeatureStore.register` before building."
            )
        return self._registry[name]

    def _find_entry(
        self,
        *,
        name: str,
        cache_key: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates = [entry for entry in self._catalog if entry["name"] == name]
        if not candidates:
            return None

        if cache_key:
            candidates = [entry for entry in candidates if entry["cache_key"] == cache_key]

        if version:
            for entry in candidates:
                if entry["version"] == version:
                    return entry
            return None

        if not candidates:
            return None

        return max(candidates, key=lambda entry: entry["created_at"])

    def _load_entry(
        self,
        entry: Mapping[str, Any],
        *,
        return_engine: str,
        from_cache: bool,
    ) -> FeatureSetResult:
        backend = _backend_from_metadata(entry, self._artifact_backend)
        storage_path = entry["storage_path"]
        if not backend.exists(storage_path):
            raise FileNotFoundError(
                f"Stored artifact missing at '{storage_path}'. Run `build(..., refresh=True)` to recreate it."
            )

        frame = backend.read(storage_path, entry["storage_format"])
        metadata = _metadata_from_dict(entry)

        return FeatureSetResult(
            data=_coerce_return_frame(frame, return_engine=return_engine),
            metadata=metadata,
            from_cache=from_cache,
        )

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        with self.catalog_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        entries = payload.get("feature_sets", [])
        if not isinstance(entries, list):
            raise ValueError("Catalog file is malformed.")
        return entries

    def _persist_catalog(self) -> None:
        tmp_path = self.catalog_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump({"feature_sets": self._catalog}, fh, indent=2, default=_json_serialize)
        tmp_path.replace(self.catalog_path)


class FeatureStoreAccessor:
    """
    Polars `.tk` accessor helper that operates against a feature store instance.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        store: Optional[FeatureStore] = None,
        store_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._frame = frame
        if store is not None:
            self._store = store
        else:
            store_kwargs = dict(store_kwargs or {})
            self._store = FeatureStore(**store_kwargs)

    @property
    def store(self) -> FeatureStore:
        return self._store

    def register(self, *args: Any, **kwargs: Any) -> FeatureStore:
        return self._store.register(*args, **kwargs)

    def build(self, name: str, **kwargs: Any) -> FeatureSetResult:
        return self._store.build(name=name, data=self._frame, **kwargs)

    def load(self, *args: Any, **kwargs: Any) -> FeatureSetResult:
        return self._store.load(*args, **kwargs)

    def assemble(self, *args: Any, **kwargs: Any) -> FeatureSetResult:
        return self._store.assemble(*args, **kwargs)


def feature_store(
    root_path: Optional[Union[str, os.PathLike[str]]] = None,
    **kwargs: Any,
) -> FeatureStore:
    """
    Convenience factory mirroring the OO constructor.
    """
    return FeatureStore(root_path=root_path, **kwargs)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _resolve_root_path(
    root_path: Optional[Union[str, os.PathLike[str]]],
) -> Path:
    env_override = os.getenv("PYTIMETK_FEATURE_STORE")
    if root_path:
        return Path(root_path).expanduser().resolve()
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path.home().expanduser() / ".pytimetk" / "feature_store"


def _pytimetk_version() -> str:
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        from importlib_metadata import version  # type: ignore
    try:
        return version("pytimetk")
    except Exception:
        return "unknown"


def _package_versions() -> Dict[str, str]:
    versions = {}
    for package in ("pandas", "polars"):
        try:
            mod = __import__(package)
            versions[package] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover
            versions[package] = "unavailable"
    return versions


def _dataframe_fingerprint(data: Union[pd.DataFrame, pl.DataFrame]) -> str:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        pandas_df = resolve_pandas_groupby_frame(data)
    elif isinstance(data, pd.DataFrame):
        pandas_df = data
    elif isinstance(data, pl.dataframe.group_by.GroupBy):
        pandas_df = data.df.to_pandas()
    elif isinstance(data, pl.DataFrame):
        pandas_df = data.to_pandas()
    else:
        raise TypeError("`data` must be a pandas or polars DataFrame (or GroupBy).")

    pandas_df = pandas_df.reset_index(drop=True)
    hasher = sha256()
    hasher.update(str(pandas_df.shape).encode())
    hasher.update(",".join(map(str, pandas_df.columns)).encode())
    try:
        import pandas.util

        hashed = pd.util.hash_pandas_object(pandas_df, index=False).to_numpy()
        hasher.update(hashed.tobytes())
    except Exception:
        sample = pandas_df.head(25).to_json().encode()
        hasher.update(sample)
    return hasher.hexdigest()


def _ensure_polars_df(data: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    raise TypeError("Expected pandas or polars DataFrame result from transform.")


def _read_frame(target: Union[Path, str, BinaryIO], storage_format: str) -> pl.DataFrame:
    storage_format = storage_format.lower()
    if storage_format == "parquet":
        return pl.read_parquet(target)
    if storage_format in {"ipc", "arrow", "feather"}:
        return pl.read_ipc(target)
    raise ValueError(f"Unsupported storage format '{storage_format}'.")


def _write_frame(
    frame: pl.DataFrame,
    target: Union[Path, str, BinaryIO],
    storage_format: str,
    writer_options: Optional[Mapping[str, Any]],
) -> None:
    storage_format = storage_format.lower()
    writer_options = dict(writer_options or {})
    if storage_format == "parquet":
        frame.write_parquet(target, **writer_options)
        return
    if storage_format in {"ipc", "arrow", "feather"}:
        frame.write_ipc(target, **writer_options)
        return
    raise ValueError(f"Unsupported storage format '{storage_format}'.")


def _metadata_from_dict(entry: Mapping[str, Any]) -> FeatureSetMetadata:
    return FeatureSetMetadata(
        name=entry["name"],
        version=entry["version"],
        cache_key=entry["cache_key"],
        storage_path=entry["storage_path"],
        storage_format=entry["storage_format"],
        storage_backend=entry.get("storage_backend", "local"),
        artifact_uri=entry.get("artifact_uri", ""),
        created_at=entry["created_at"],
        data_fingerprint=entry["data_fingerprint"],
        transform_fingerprint=entry["transform_fingerprint"],
        transform_module=entry["transform_module"],
        transform_name=entry["transform_name"],
        transform_kwargs=entry["transform_kwargs"],
        pytimetk_version=entry["pytimetk_version"],
        package_versions=entry["package_versions"],
        tags=tuple(entry.get("tags", ())),
        description=entry.get("description"),
        key_columns=tuple(entry["key_columns"]) if entry.get("key_columns") else None,
        row_count=entry.get("row_count"),
        column_count=entry.get("column_count"),
        extra_metadata=entry.get("extra_metadata", {}),
    )


def _resolve_return_engine(return_engine: str, base_input: Any = None) -> str:
    if return_engine == "auto":
        if base_input is None:
            return "polars"
        kind = identify_frame_kind(base_input)
        return "polars" if kind.startswith("polars") else "pandas"
    if return_engine not in {"pandas", "polars"}:
        raise ValueError("`return_engine` must be 'auto', 'pandas', or 'polars'.")
    return return_engine


def _coerce_return_frame(
    frame: pl.DataFrame,
    *,
    return_engine: str,
    base_input: Any = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    if return_engine == "polars":
        return frame
    if return_engine == "pandas":
        pandas_df = frame.to_pandas()
        if base_input is not None and isinstance(base_input, pd.DataFrame):
            pandas_df.index = base_input.index if len(base_input.index) == len(pandas_df) else pandas_df.index
        return pandas_df
    raise ValueError("Unsupported `return_engine`.")


def _normalise_for_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    def serialise(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(k): serialise(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [serialise(item) for item in value]
        return repr(value)

    return {str(key): serialise(value) for key, value in payload.items()}


def _make_cache_key(name: str, data_fingerprint: str, transform_fingerprint: str) -> str:
    hasher = sha256()
    hasher.update(name.encode())
    hasher.update(data_fingerprint.encode())
    hasher.update(transform_fingerprint.encode())
    return hasher.hexdigest()


def _derive_version(cache_key: str) -> str:
    return cache_key[:12]


def _extension_for_format(storage_format: str) -> str:
    storage_format = storage_format.lower()
    if storage_format == "parquet":
        return "parquet"
    if storage_format in {"ipc", "arrow", "feather"}:
        return "ipc"
    raise ValueError(f"Unsupported storage format '{storage_format}'.")


def _create_artifact_backend(artifact_uri: str) -> ArtifactBackend:
    artifact_uri = artifact_uri.rstrip("/")
    if "://" in artifact_uri and not artifact_uri.startswith("file://"):
        return PyArrowArtifactBackend(artifact_uri)
    if artifact_uri.startswith("file://"):
        return PyArrowArtifactBackend(artifact_uri)
    return LocalArtifactBackend(Path(artifact_uri))


def _create_artifact_backend_with_name(name: str, artifact_uri: str) -> ArtifactBackend:
    if name == "local":
        return LocalArtifactBackend(Path(artifact_uri))
    if name == "pyarrow":
        return PyArrowArtifactBackend(artifact_uri)
    return _create_artifact_backend(artifact_uri)


def _backend_from_metadata(entry: Mapping[str, Any], default_backend: ArtifactBackend) -> ArtifactBackend:
    backend_name = entry.get("storage_backend")
    artifact_uri = entry.get("artifact_uri")

    if not backend_name or not artifact_uri:
        return default_backend

    if (
        backend_name == default_backend.name
        and artifact_uri.rstrip("/") == default_backend.artifact_uri.rstrip("/")
    ):
        return default_backend

    try:
        return _create_artifact_backend_with_name(backend_name, artifact_uri)
    except Exception:
        return default_backend


def _join_uri(base: str, *segments: str) -> str:
    base = base.rstrip("/")
    tail = "/".join(segment.strip("/") for segment in segments if segment)
    if not tail:
        return base
    return f"{base}/{tail}"


def _parse_feature_spec(spec: Union[str, Tuple[str, str]]) -> Tuple[str, Optional[str]]:
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, Sequence) and len(spec) == 2:
        name, version = spec
        return str(name), str(version) if version is not None else None
    raise TypeError("Feature spec must be a string name or a (name, version) tuple.")


def _json_serialize(obj: Any) -> Jsonable:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return {str(k): _json_serialize(v) for k, v in obj.items()}
    return repr(obj)
