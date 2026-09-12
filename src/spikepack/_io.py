"""On-disk containers for encoded spike trains: a minimal Blosc directory shard, and Zarr."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spikepack._core import DEFAULT_QUANTIZATION_US, compress_array, decode_times, decompress_array, encode_times

FORMAT_VERSION = "ibl_ai_agent_spike_shard_v2"


def _build_arrays(
    times_seconds: np.ndarray,
    labels: np.ndarray | None,
    quantization_us: int,
    cluster_ids: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    deltas, time_meta = encode_times(times_seconds, quantization_us=quantization_us)
    arrays: dict[str, np.ndarray] = {"spike_times_delta_ticks": deltas}
    has_labels = labels is not None
    has_cluster_ids = cluster_ids is not None and has_labels
    if has_labels:
        labels = np.asarray(labels)
        if labels.shape[0] != deltas.shape[0]:
            raise ValueError(f"labels length {labels.shape[0]} does not match times length {deltas.shape[0]}")
        arrays["spike_clusters"] = labels
    if has_cluster_ids:
        sorted_ids = np.sort(cluster_ids).astype(np.int32)
        arrays["cluster_ids"] = sorted_ids
        arrays["cluster_spike_counts"] = np.bincount(
            labels, minlength=len(sorted_ids)
        ).astype(np.int32)
    meta = {
        "format": FORMAT_VERSION,
        "n_events": int(deltas.size),
        "has_labels": has_labels,
        "has_cluster_ids": has_cluster_ids,
        **time_meta,
    }
    return arrays, meta


def _restore(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, Any]:
    times = decode_times(arrays["spike_times_delta_ticks"], meta)
    out: dict[str, Any] = {"times": times, "meta": meta}
    if meta.get("has_labels"):
        out["labels"] = arrays["spike_clusters"]
    if meta.get("has_cluster_ids"):
        out["cluster_ids"] = arrays["cluster_ids"]
        out["cluster_spike_counts"] = arrays["cluster_spike_counts"]
    return out


# --- Blosc directory shard: one directory per train, one .blosc file per array + meta.json ---


def write_blosc(
    path: Path,
    *,
    times_seconds: np.ndarray,
    labels: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    quantization_us: int = DEFAULT_QUANTIZATION_US,
    extra_meta: dict[str, Any] | None = None,
) -> int:
    """Write a spike train to a Blosc directory shard.

    Output is compatible with ``ibl_ai_agent_spike_shard_v2`` and can be
    loaded directly by ``ibl-ai-agent``'s ``load_spike_shard``.

    Parameters
    ----------
    path : Path
        Output directory (created if missing).
    times_seconds : np.ndarray
        Sorted event times in seconds.
    labels : np.ndarray, optional
        Per-event dense cluster index (0..N-1), same length as `times_seconds`.
        Stored as ``spike_clusters`` in the shard.
    cluster_ids : np.ndarray, optional
        Original cluster IDs (e.g. kilosort integers), length N.  When provided
        alongside ``labels``, also writes ``cluster_ids`` and
        ``cluster_spike_counts`` arrays so that label *i* maps to
        ``cluster_ids[i]``.  Requires ``labels`` to be set.
    quantization_us : int
        Tick size in microseconds.
    extra_meta : dict, optional
        Extra fields merged into ``meta.json`` (e.g. ``pid``, ``eid``).

    Returns
    -------
    int
        Total bytes written to disk.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays, meta = _build_arrays(times_seconds, labels, quantization_us, cluster_ids)
    meta = {**meta, **(extra_meta or {})}
    manifest: dict[str, Any] = dict(meta)
    manifest["arrays"] = {}
    for name, arr in arrays.items():
        payload, spec = compress_array(arr)
        spec["entry"] = f"{name}.blosc"
        manifest["arrays"][name] = spec
        (path / spec["entry"]).write_bytes(payload)
    (path / "meta.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def read_blosc(path: Path) -> dict[str, Any]:
    """Read a spike train written by `write_blosc`.

    Returns
    -------
    dict
        `times` (seconds, float64), `labels` (if present), `meta`.
    """
    path = Path(path)
    manifest = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    arrays = {
        name: decompress_array((path / spec["entry"]).read_bytes(), spec) for name, spec in manifest["arrays"].items()
    }
    meta = {k: v for k, v in manifest.items() if k != "arrays"}
    return _restore(arrays, meta)


# --- Zarr store: one group per train, standard Zarr arrays/attrs, Blosc(zstd, shuffle) codec ---


def write_zarr(
    path: Path,
    *,
    times_seconds: np.ndarray,
    labels: np.ndarray | None = None,
    quantization_us: int = DEFAULT_QUANTIZATION_US,
    group: str | None = None,
    chunk_size: int | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Write a spike train to a Zarr group.

    Parameters
    ----------
    path : Path
        Zarr store path (a single store may hold multiple trains as sibling groups).
    times_seconds : np.ndarray
        Sorted event times in seconds.
    labels : np.ndarray, optional
        Per-event integer label (e.g. unit/cluster id), same length as `times_seconds`.
    quantization_us : int
        Tick size in microseconds.
    group : str, optional
        Subgroup name within the store (default: store root).
    chunk_size : int, optional
        Chunk length in events (default: whole array in one chunk).
    extra_meta : dict, optional
        Extra fields merged into the group's Zarr attributes.
    """
    import zarr
    from zarr.codecs import BloscCodec

    arrays, meta = _build_arrays(times_seconds, labels, quantization_us)
    meta = {**meta, **(extra_meta or {})}
    root = zarr.open_group(str(path), mode="a")
    target = root.require_group(group) if group else root
    codec = BloscCodec(cname="zstd", clevel=7, shuffle="shuffle")
    for name, arr in arrays.items():
        if name in target:
            del target[name]
        chunks = (min(chunk_size, arr.shape[0]),) if chunk_size else arr.shape
        target.create_array(name, shape=arr.shape, chunks=chunks, dtype=arr.dtype, compressors=[codec])
        target[name][:] = arr
    target.attrs.update(meta)


def read_zarr(path: Path, *, group: str | None = None) -> dict[str, Any]:
    """Read a spike train written by `write_zarr`.

    Returns
    -------
    dict
        `times` (seconds, float64), `labels` (if present), `meta`.
    """
    import zarr

    root = zarr.open_group(str(path), mode="r")
    target = root[group] if group else root
    meta = dict(target.attrs)
    arrays = {"spike_times_delta_ticks": np.asarray(target["spike_times_delta_ticks"][:])}
    if meta.get("has_labels"):
        arrays["spike_clusters"] = np.asarray(target["spike_clusters"][:])
    if meta.get("has_cluster_ids"):
        arrays["cluster_ids"] = np.asarray(target["cluster_ids"][:])
        arrays["cluster_spike_counts"] = np.asarray(target["cluster_spike_counts"][:])
    return _restore(arrays, meta)
