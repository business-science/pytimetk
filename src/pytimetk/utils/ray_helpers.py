from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pytimetk.utils.parallel_helpers import conditional_tqdm

try:  # pragma: no cover - optional dependency
    import ray  # type: ignore
except ImportError:  # pragma: no cover - ray optional
    ray = None  # type: ignore


def ensure_ray_initialized(num_cpus: Optional[int] = None):
    """
    Initialize Ray on-demand and return the module.

    Parameters
    ----------
    num_cpus : Optional[int]
        Optional CPU limit to pass to ``ray.init``.
    """

    if ray is None:  # pragma: no cover - guarded at runtime
        raise ImportError(
            "Ray is required for parallel execution. Install it with `pip install ray` "
            "or set `threads=1` to disable parallel processing."
        )

    if not ray.is_initialized():
        init_kwargs = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "log_to_driver": False,
        }
        if num_cpus is not None and num_cpus > 0:
            init_kwargs["num_cpus"] = num_cpus
        ray.init(**init_kwargs)

    return ray


def run_ray_tasks(
    func: Callable,
    args_list: Sequence[Tuple],
    *,
    num_cpus: Optional[int],
    desc: str,
    show_progress: bool,
) -> List:
    """
    Execute ``func`` across ``args_list`` using Ray.

    Parameters
    ----------
    func : Callable
        Function to execute remotely. Must be picklable.
    args_list : Sequence[Tuple]
        Positional arguments for each invocation.
    num_cpus : Optional[int]
        Desired CPU count for the Ray cluster. ``None`` lets Ray decide.
    desc : str
        Description for the progress iterator.
    show_progress : bool
        Whether to display progress via tqdm.
    """

    if not args_list:
        return []

    from pytimetk.utils.parallel_helpers import conditional_tqdm

    ray_module = ensure_ray_initialized(num_cpus)
    remote_worker = ray_module.remote(func)
    jobs = [remote_worker.remote(*args) for args in args_list]
    index_map = {job: idx for idx, job in enumerate(jobs)}
    pending = list(jobs)
    results: List = [None] * len(jobs)

    iterator = conditional_tqdm(
        range(len(jobs)),
        total=len(jobs),
        display=show_progress,
        desc=desc,
    )

    for _ in iterator:
        ready, pending = ray_module.wait(pending, num_returns=1)
        ref = ready[0]
        idx = index_map[ref]
        results[idx] = ray_module.get(ref)

    return results


__all__ = ["ensure_ray_initialized", "run_ray_tasks"]
