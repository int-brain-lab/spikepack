"""Delta-quantization codec for sorted event-time arrays (spike trains).

Encoding pipeline
------------------
1. subtract the first timestamp (origin normalization) so tick values stay small
2. quantize to integer ticks at a configurable resolution
3. delta-encode (times are sorted, so successive differences are small and low-entropy)
4. pick the narrowest integer dtype that holds the largest delta (uint16 -> uint32 -> uint64)
5. compress with Blosc(zstd, shuffle)

This loses at most half a quantization step per timestamp and is lossless otherwise
(no spikes are dropped or reordered).
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_QUANTIZATION_US = 100
CANDIDATE_DTYPES = (np.uint16, np.uint32, np.uint64)
_FALLBACK_DTYPE = np.uint32  # first dtype that requires fallback_used=True


class SpikepackError(ValueError):
    """Raised when a spike train cannot be encoded under the requested settings."""


def encode_times(
    times_seconds: np.ndarray, *, quantization_us: int = DEFAULT_QUANTIZATION_US
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantize and delta-encode a sorted array of event times.

    Parameters
    ----------
    times_seconds : np.ndarray
        Sorted (ascending), 1-D array of event times in seconds.
    quantization_us : int
        Tick size in microseconds. `100` means one tick = 0.1 ms.

    Returns
    -------
    deltas : np.ndarray
        Delta-encoded integer ticks, dtype chosen to be as narrow as possible.
    meta : dict
        Fields required to decode. Keys are compatible with ibl-ai-agent
        ``ibl_ai_agent_spike_shard_v2``: ``time_quantization_us``,
        ``time_origin_seconds``, ``time_origin_ticks``, ``storage_dtype``,
        ``time_dtype``, ``delta_encoded``, ``overflow_checked``,
        ``fallback_used``, ``n_events``, ``max_delta_ticks``.

    Raises
    ------
    SpikepackError
        If ``quantization_us`` is not positive, times are not sorted, or the
        largest delta overflows the widest candidate dtype (``uint64``).
    """
    if quantization_us <= 0:
        raise SpikepackError(f"quantization_us must be positive, got {quantization_us}")
    times = np.asarray(times_seconds, dtype=np.float64)
    if times.ndim != 1:
        raise SpikepackError(f"times_seconds must be 1-D, got shape {times.shape}")
    if times.size == 0:
        dtype0 = CANDIDATE_DTYPES[0]
        return np.asarray([], dtype=dtype0), {
            "time_quantization_us": int(quantization_us),
            "time_origin_seconds": 0.0,
            "time_origin_ticks": 0,
            "storage_dtype": np.dtype(dtype0).name,
            "time_dtype": np.dtype(dtype0).str,
            "delta_encoded": True,
            "overflow_checked": True,
            "fallback_used": False,
            "n_events": 0,
        }
    if np.any(np.diff(times) < 0):
        raise SpikepackError("times_seconds must be sorted ascending")

    origin_seconds = float(times[0])
    origin_ticks = int(np.rint(origin_seconds * 1_000_000.0 / quantization_us))
    shifted = np.maximum(times - origin_seconds, 0.0)
    ticks = np.rint(shifted * 1_000_000.0 / quantization_us).astype(np.uint64)
    deltas = np.empty_like(ticks)
    deltas[0] = 0
    if ticks.size > 1:
        deltas[1:] = np.diff(ticks)
    max_delta = int(deltas.max(initial=0))

    for dtype in CANDIDATE_DTYPES:
        if max_delta <= int(np.iinfo(dtype).max):
            fallback = dtype != CANDIDATE_DTYPES[0]
            return deltas.astype(dtype), {
                "time_quantization_us": int(quantization_us),
                "time_origin_seconds": origin_seconds,
                "time_origin_ticks": origin_ticks,
                "storage_dtype": np.dtype(dtype).name,
                "time_dtype": np.dtype(dtype).str,
                "delta_encoded": True,
                "overflow_checked": True,
                "fallback_used": fallback,
                "n_events": int(deltas.size),
                "max_delta_ticks": max_delta,
            }
    raise SpikepackError(f"max delta ticks={max_delta} overflows uint64 at quantization_us={quantization_us}")


def decode_times(deltas: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Invert `encode_times`.

    Parameters
    ----------
    deltas : np.ndarray
        Delta-encoded integer ticks, as returned by `encode_times`.
    meta : dict
        Metadata dict returned by `encode_times`. Accepts both the current
        keys (``time_quantization_us``, ``time_origin_ticks``) and the legacy
        keys (``quantization_us``, ``origin_seconds``) for backward compatibility.

    Returns
    -------
    np.ndarray
        Reconstructed event times in seconds (``float64``), equal to the input
        times to within half a quantization step.
    """
    if deltas.size == 0:
        return np.asarray([], dtype=np.float64)
    quantization_us = meta.get("time_quantization_us", meta.get("quantization_us"))
    ticks = np.cumsum(deltas.astype(np.int64), dtype=np.int64)
    # Prefer float origin (best precision, ≤½ tick error).
    # Fall back to tick-based origin only for files written by ibl-ai-agent
    # that do not store time_origin_seconds.
    if "time_origin_seconds" in meta or "origin_seconds" in meta:
        origin_s = meta.get("time_origin_seconds", meta.get("origin_seconds", 0.0))
        return ticks.astype(np.float64) * (quantization_us / 1_000_000.0) + origin_s
    ticks += int(meta.get("time_origin_ticks", 0))
    return ticks.astype(np.float64) * (quantization_us / 1_000_000.0)


def compress_array(arr: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    """Compress a numpy array with Blosc(zstd, shuffle).

    Returns
    -------
    payload : bytes
        Compressed byte payload.
    spec : dict
        `dtype`, `shape`, `nbytes`, `compressed_nbytes` needed to decompress.
    """
    from numcodecs import Blosc

    array = np.ascontiguousarray(arr)
    codec = Blosc(cname="zstd", clevel=7, shuffle=Blosc.SHUFFLE)
    payload = codec.encode(array)
    return payload, {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "order": "C",
        "nbytes": int(array.nbytes),
        "compressed_nbytes": int(len(payload)),
        "typesize": int(array.dtype.itemsize),
        "codec": {"name": "blosc", "cname": "zstd", "clevel": 7, "shuffle": "shuffle"},
    }


def decompress_array(payload: bytes, spec: dict[str, Any]) -> np.ndarray:
    """Invert `compress_array`."""
    if int(spec.get("nbytes", 0)) == 0:
        return np.asarray([], dtype=np.dtype(spec["dtype"])).reshape(tuple(spec["shape"]))
    from numcodecs import Blosc

    raw = Blosc().decode(payload)
    arr = np.frombuffer(raw, dtype=np.dtype(spec["dtype"]))
    return arr.reshape(tuple(spec["shape"]))
