# spikepack — codec for sorted spike-time arrays

Lossy-in-timing, lossless-in-order codec for sorted spike (or any sorted event-time)
arrays. Quantizes to integer ticks, delta-encodes, and compresses with
Blosc(zstd, shuffle) — routinely **>7-10× compression** over raw `float64` seconds at
a `100 µs` quantization (≤ 50 µs worst-case timing error per spike).

Extracted and generalized from the `bwm_ephys` spike-store builder in
[`int-brain-lab/ibl-ai-agent`](https://github.com/int-brain-lab/ibl-ai-agent), where
this exact recipe (`delta_int_ticks + 100us + shuffle_zstd`) is the frozen default for
the IBL Brain-Wide Map ephys release.

```bash
pip install spikepack        # add the `zarr` extra for Zarr container support
```

```python
import numpy as np
from spikepack import write_blosc, read_blosc

times = np.sort(np.random.default_rng(0).exponential(0.02, 1_000_000).cumsum())
labels = np.random.default_rng(0).integers(0, 200, times.size)  # e.g. unit/cluster id

write_blosc("spikes", times_seconds=times, labels=labels, quantization_us=100)
out = read_blosc("spikes")
# out["times"], out["labels"], out["meta"]
```

## Documentation

Full documentation is at **https://int-brain-lab.github.io/spikepack/**.

| Section | Contents |
| --- | --- |
| [Tutorial](https://int-brain-lab.github.io/spikepack/tutorials/first-compression.html) | Encode, write, and read your first spike train |
| [How-To: quantization](https://int-brain-lab.github.io/spikepack/how-to/choose-quantization.html) | Trading fidelity for size |
| [How-To: Blosc vs Zarr](https://int-brain-lab.github.io/spikepack/how-to/blosc-vs-zarr.html) | Picking a container format |
| [API reference](https://int-brain-lab.github.io/spikepack/reference/) | Full public API |
| [On-disk format](https://int-brain-lab.github.io/spikepack/reference/format.html) | Blosc-shard and Zarr layout specs |
| [Codec design](https://int-brain-lab.github.io/spikepack/explanation/encoding.html) | Why quantize + delta-encode + shuffle |
| [Benchmark](https://int-brain-lab.github.io/spikepack/explanation/benchmark.html) | Blosc vs Zarr, compression ratios across 11 real IBL insertions |

## How it works

1. subtract the first timestamp (origin normalization)
2. quantize to integer ticks at a configurable resolution (default `100 µs`)
3. delta-encode — sorted times mean small, low-entropy successive differences
4. pick the narrowest safe integer dtype (`uint16 -> uint32 -> uint64`)
5. compress with Blosc(zstd, shuffle)

Two container formats ship the same codec: a minimal dependency-free directory-of-
`.blosc`-files shard, and a standard Zarr group for cross-tool interoperability and
chunked access. See [Blosc vs Zarr](https://int-brain-lab.github.io/spikepack/how-to/blosc-vs-zarr.html).

## Help and Feedback

Have a question, found an issue, or want to share feedback? Please open an issue on
[GitHub](https://github.com/int-brain-lab/spikepack/issues).
