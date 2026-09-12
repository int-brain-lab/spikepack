import numpy as np
import pytest

from spikepack import SpikepackError, compress_array, decode_times, decompress_array, encode_times


def synthetic_train(n=200_000, rate_hz=50.0, seed=0):
    rng = np.random.default_rng(seed)
    isis = rng.exponential(scale=1.0 / rate_hz, size=n)
    return np.cumsum(isis)


@pytest.mark.parametrize("quantization_us", [25, 50, 100, 250])
def test_round_trip_within_half_tick(quantization_us):
    times = synthetic_train()
    deltas, meta = encode_times(times, quantization_us=quantization_us)
    recovered = decode_times(deltas, meta)
    assert np.max(np.abs(recovered - times)) <= (quantization_us * 1e-6) / 2 + 1e-12


def test_empty_array_round_trips():
    times = np.asarray([], dtype=np.float64)
    deltas, meta = encode_times(times, quantization_us=100)
    assert deltas.size == 0
    assert decode_times(deltas, meta).size == 0


def test_single_event_round_trips():
    times = np.asarray([12.345], dtype=np.float64)
    deltas, meta = encode_times(times, quantization_us=100)
    recovered = decode_times(deltas, meta)
    assert np.max(np.abs(recovered - times)) <= 50e-6 + 1e-12


def test_dtype_narrows_to_uint16_for_dense_train():
    times = synthetic_train(n=10_000, rate_hz=100.0)
    deltas, meta = encode_times(times, quantization_us=100)
    assert meta["storage_dtype"] == "uint16"
    assert meta["fallback_used"] is False
    assert deltas.dtype == np.uint16


def test_dtype_falls_back_to_uint32_on_large_gap():
    # gap of 100 ms at 1 us ticks = 100_000 ticks, overflows uint16 (max 65535)
    times = np.asarray([0.0, 0.1, 0.100_5])
    deltas, meta = encode_times(times, quantization_us=1)
    assert meta["storage_dtype"] == "uint32"
    assert meta["fallback_used"] is True
    recovered = decode_times(deltas, meta)
    assert np.max(np.abs(recovered - times)) <= 1e-6 + 1e-12


def test_dtype_falls_back_to_uint64_on_huge_gap():
    # ~5 days at 100 us ticks overflows uint32 (max ~4.29e9)
    times = np.asarray([0.0, 5 * 86_400.0])
    deltas, meta = encode_times(times, quantization_us=100)
    assert meta["storage_dtype"] == "uint64"
    assert meta["fallback_used"] is True
    recovered = decode_times(deltas, meta)
    assert np.max(np.abs(recovered - times)) <= 50e-6 + 1e-3  # float64 tick math, generous tolerance


def test_rejects_unsorted_times():
    times = np.asarray([1.0, 0.5, 2.0])
    with pytest.raises(SpikepackError):
        encode_times(times, quantization_us=100)


def test_rejects_non_positive_quantization():
    with pytest.raises(SpikepackError):
        encode_times(synthetic_train(n=10), quantization_us=0)


def test_compress_decompress_array_round_trip():
    arr = np.arange(10_000, dtype=np.uint16)
    payload, spec = compress_array(arr)
    assert len(payload) < arr.nbytes  # narrow monotonic array should compress
    out = decompress_array(payload, spec)
    np.testing.assert_array_equal(out, arr)


def test_compress_decompress_empty_array():
    arr = np.asarray([], dtype=np.uint16)
    payload, spec = compress_array(arr)
    out = decompress_array(payload, spec)
    assert out.size == 0
    assert out.dtype == arr.dtype
