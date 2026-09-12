import numpy as np
import pytest

from spikepack import read_blosc, write_blosc


def synthetic_train(n=50_000, rate_hz=50.0, seed=1):
    rng = np.random.default_rng(seed)
    isis = rng.exponential(scale=1.0 / rate_hz, size=n)
    times = np.cumsum(isis)
    labels = rng.integers(0, 30, size=n).astype(np.int32)
    return times, labels


def test_blosc_round_trip_times_only(tmp_path):
    times, _ = synthetic_train()
    write_blosc(tmp_path / "train", times_seconds=times, quantization_us=100)
    out = read_blosc(tmp_path / "train")
    assert np.max(np.abs(out["times"] - times)) <= 50e-6 + 1e-12
    assert "labels" not in out
    assert out["meta"]["n_events"] == times.size


def test_blosc_round_trip_with_labels(tmp_path):
    times, labels = synthetic_train()
    write_blosc(tmp_path / "train", times_seconds=times, labels=labels, quantization_us=100)
    out = read_blosc(tmp_path / "train")
    np.testing.assert_array_equal(out["labels"], labels)


def test_blosc_layout_is_directory_of_blosc_files_plus_meta(tmp_path):
    times, labels = synthetic_train(n=1_000)
    write_blosc(tmp_path / "train", times_seconds=times, labels=labels, quantization_us=100)
    files = {p.name for p in (tmp_path / "train").iterdir()}
    assert files == {"meta.json", "spike_times_delta_ticks.blosc", "spike_clusters.blosc"}


def test_blosc_cluster_ids_written_and_recovered(tmp_path):
    times, labels = synthetic_train(n=5_000)
    n_units = int(labels.max()) + 1
    cluster_ids = np.arange(100, 100 + n_units, dtype=np.int32)  # arbitrary original IDs
    write_blosc(tmp_path / "train", times_seconds=times, labels=labels, cluster_ids=cluster_ids, quantization_us=100)
    out = read_blosc(tmp_path / "train")
    np.testing.assert_array_equal(out["cluster_ids"], np.sort(cluster_ids))
    assert out["cluster_spike_counts"].sum() == times.size
    files = {p.name for p in (tmp_path / "train").iterdir()}
    assert "cluster_ids.blosc" in files
    assert "cluster_spike_counts.blosc" in files


def test_blosc_extra_meta_is_preserved(tmp_path):
    times, _ = synthetic_train(n=1_000)
    write_blosc(
        tmp_path / "train",
        times_seconds=times,
        quantization_us=100,
        extra_meta={"source": "unit-test", "recording_id": "abc"},
    )
    out = read_blosc(tmp_path / "train")
    assert out["meta"]["source"] == "unit-test"
    assert out["meta"]["recording_id"] == "abc"


def test_blosc_mismatched_labels_length_raises(tmp_path):
    times, labels = synthetic_train(n=1_000)
    with pytest.raises(ValueError):
        write_blosc(tmp_path / "train", times_seconds=times, labels=labels[:-1], quantization_us=100)


def test_blosc_compresses_smaller_than_raw_float64(tmp_path):
    times, labels = synthetic_train(n=200_000)
    raw_bytes = times.nbytes + labels.nbytes
    shard_bytes = write_blosc(tmp_path / "train", times_seconds=times, labels=labels, quantization_us=100)
    assert shard_bytes < raw_bytes / 3


@pytest.mark.filterwarnings("ignore")
def test_zarr_round_trip_matches_blosc(tmp_path):
    pytest.importorskip("zarr")
    from spikepack import read_zarr, write_zarr

    times, labels = synthetic_train()
    write_blosc(tmp_path / "train", times_seconds=times, labels=labels, quantization_us=100)
    write_zarr(tmp_path / "train.zarr", times_seconds=times, labels=labels, quantization_us=100)

    blosc_out = read_blosc(tmp_path / "train")
    zarr_out = read_zarr(tmp_path / "train.zarr")

    np.testing.assert_array_equal(blosc_out["times"], zarr_out["times"])
    np.testing.assert_array_equal(blosc_out["labels"], zarr_out["labels"])


def test_zarr_multiple_groups_in_one_store(tmp_path):
    pytest.importorskip("zarr")
    from spikepack import read_zarr, write_zarr

    times_a, labels_a = synthetic_train(n=1_000, seed=2)
    times_b, labels_b = synthetic_train(n=1_000, seed=3)
    store = tmp_path / "multi.zarr"
    write_zarr(store, times_seconds=times_a, labels=labels_a, quantization_us=100, group="rec_a")
    write_zarr(store, times_seconds=times_b, labels=labels_b, quantization_us=100, group="rec_b")

    out_a = read_zarr(store, group="rec_a")
    out_b = read_zarr(store, group="rec_b")
    np.testing.assert_array_equal(out_a["labels"], labels_a)
    np.testing.assert_array_equal(out_b["labels"], labels_b)
