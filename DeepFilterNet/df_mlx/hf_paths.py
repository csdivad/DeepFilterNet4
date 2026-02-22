"""Helpers for normalizing Hugging Face cache paths."""

from __future__ import annotations


def normalize_hf_dataset_cache_dir(cache_dir: str) -> str:
    """Normalize hf:// cache paths to dataset-qualified repo form.

    Examples:
        hf://user/repo -> hf://datasets/user/repo
        hf://datasets/user/repo -> hf://datasets/user/repo
        /local/path -> /local/path
    """
    raw = str(cache_dir).strip()
    if not raw.startswith("hf://"):
        return raw

    repo_path = raw[5:].lstrip("/")
    if repo_path.startswith(("datasets/", "models/", "spaces/")):
        return f"hf://{repo_path}"
    return f"hf://datasets/{repo_path}"


def hf_dataset_fsspec_path(cache_dir: str) -> str:
    """Return a normalized path for HfFileSystem calls.

    For hf:// inputs this returns the scheme-less path expected by HfFileSystem.
    Non-hf inputs are returned as-is.
    """
    normalized = normalize_hf_dataset_cache_dir(cache_dir)
    if not normalized.startswith("hf://"):
        return normalized
    return normalized[5:]
