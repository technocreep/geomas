"""Model cache helpers for Hugging Face repositories."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

# In-memory mapping of (repo_id, cache_dir) -> resolved Path to avoid repeated
# snapshot downloads within a single process.
_CACHE: Dict[Tuple[str, Optional[Path]], Path] = {}


def get_cached_model(
    repo_id: str,
    *,
    allow_network: bool,
    cache_dir: Path | None = None,
    max_workers: int | None = None,
) -> Path:
    """Return local path to cached *repo_id*, downloading if permitted.

    The result is memoized in-process so that multiple adapters requesting the
    same model reuse the resolved path without triggering additional snapshot
    downloads.

    Parameters
    ----------
    repo_id:
        Hugging Face repository identifier.
    allow_network:
        Whether network downloads are permitted when the model is missing.
    cache_dir:
        Optional explicit cache directory; defaults to Hugging Face's standard
        cache location.
    max_workers:
        Optional worker count used for ``snapshot_download`` when network access
        is allowed.

    Returns
    -------
    Path
        Directory containing the resolved model snapshot.

    Raises
    ------
    LocalEntryNotFoundError
        If ``allow_network`` is ``False`` and the model is not present in the
        local cache.
    """

    key = (repo_id, cache_dir)
    if key in _CACHE:
        return _CACHE[key]

    path = Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=not allow_network,
            resume_download=True,
            max_workers=max_workers,
        )
    )
    _CACHE[key] = path
    return path


def model_is_cached(repo_id: str, *, cache_dir: Path | None = None) -> bool:
    """Return ``True`` if *repo_id* exists in the local cache."""
    try:
        get_cached_model(repo_id, allow_network=False, cache_dir=cache_dir)
        return True
    except LocalEntryNotFoundError:
        return False


__all__ = ["get_cached_model", "model_is_cached"]
