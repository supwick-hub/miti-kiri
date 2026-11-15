# miti-kiri

By Supriya Wickrematillake. Part of the kiri family: [kiri.terrific.life](https://kiri.terrific.life/).

Splits a song into drums, bass, other and vocals on the Mac's GPU. Rust, no Python at runtime; the only
external program is `ffmpeg`, which decodes anything that is not already a WAV. This repository holds the
runtime and the 1.0 model. A 5-minute track takes 11 to 18 s, depending on what else the Mac is doing.
Memory stays at about 300 MB whatever the track's length: 223 to 303 MB measured on tracks of 3 to 60 minutes
(M1 MacBook Air, 8 GB).

| | |
|---|---|
| `crates/kiri-core` | decode, FFT, windowing, the STFT stage, the network interface (`backend::Backend`) and Core ML |
| `crates/miti-kiri` | the separator: pipeline, consistency stage, stem writing and the `miti-kiri` command |
| `model/` | miti-kiri 1.0, compiled for Core ML (and a folded variant); see [MODEL_CARD.md](MODEL_CARD.md) |
| `tools/golden.sh` | the 1.0 model's outputs, hashed on a synthetic mix; [tests/golden.sha256](tests/golden.sha256) |

`kiri-core` depends on nothing of ours; `miti-kiri` depends on `kiri-core` alone (a test reads `cargo metadata`
and fails on any other edge).

## Build

    git clone https://github.com/supwick-hub/miti-kiri && cd miti-kiri
    cargo build --release

macOS only (Core ML). The binary is `target/release/miti-kiri`.

## Run

    miti-kiri TRACK -o OUTDIR -m model/miti-kiri-1.0.mlmodelc [--overlap 0|0.25|0.5] [--consist tf|none] [--pcm16] [--progress]

Writes `OUTDIR/{drums,bass,other,vocals}.wav` (32-bit float, 44.1 kHz stereo; `--pcm16`: 16-bit with TPDF
dither). Defaults: overlap 0.25, consist `tf`. The model must be fp16 or fp32 with input `mix_spec`
(B,4,2049,T) and output `stems_spec` (B,4,4,2049,T). The window (T frames, (T-1)*1024 samples) and the
windows per network call (B) are read from the model's signature in its `model.mil` before it loads;
`--frames` and `--batch` only check them. Windows go to the network B at a time; a short last batch is padded
with zero windows whose output is dropped.

The network runs on the GPU (`CPUAndGPU`): it loads in under a second and matches CPU fp32 to 46-61 dB. The
Neural Engine (`KIRI_UNITS=ane`; also `all`, `cpu`, `gpu`) was faster per window but took minutes to prepare,
again and again, and its fp16 output drifted to 19 dB on vocals.

`--progress` prints to stderr, one line each: `preparing` before the model loads, then `progress <done> <total>`
after each window leaves the network. `total` is estimated before decoding (a WAV's header, else ffprobe's
duration; 0 when neither answers) and the last line has the exact count.

## Signal path

ffmpeg decode (f32le, stereo, 44.1 kHz) -> T-frame windows ((T-1)*1024 samples, raised-cosine cross-fade, no
fade-in on the first; the audio past the file's end is the file mirrored, not silence, so a model does not
fade whatever sounds at the end) -> STFT (4096/1024, Hann, reflect, normalised, then each window's own mean and
std) -> Core ML -> de-normalise, iSTFT, overlap-add -> optional `tf` consistency (the leftover of the mix is
shared among the stems by each stem's power share per bin, over the whole track) -> WAV.

Threads (plain, bounded channels): decode + spectrum in; network (main thread); spectrum out + overlap-add;
consistency + WAV writing. Consistency runs frame by frame as the overlap-added stems become final, so it
overlaps the network too. fp16 network output is widened while the spectrum stage reads it, not on the
network's thread. Model load overlaps decoding the first windows. Nothing is held whole: a track is never
decoded into memory.

The FFT sits behind `fft::FftBackend` (rustfft today), the network behind `backend::Backend` (Core ML today;
an ONNX backend would sit beside it).

## Extension points

The separator knows nothing about what a host wants beside the stems; it offers `blocks::BlockSink`: the final
mix and stem blocks, as the files hold them, to any consumer (`cli::run(&args, Some(sink))`).

## Tests

    cargo test --workspace --release        # unit tests; tests/batching.rs: batch 2 == batch 1 bit for bit
                                            # (pass-through backend); tests/boundaries.rs
    tools/golden.sh check                   # the 1.0 model's outputs, hashed (needs the release build)

Tests that decode need `ffmpeg` on the PATH. `tools/golden.sh` hashes four runs (default, overlap 0 without
consistency, overlap 0.5 with 16-bit output, the folded model) on a synthetic mix; Core ML's GPU output is
machine-specific, so the hashes hold on the Mac that recorded them and another machine records its own.

## Numbers

SDR against true stems and the Mac's time per track: [MODEL_CARD.md](MODEL_CARD.md).

## Licence

The code is MIT ([LICENSE](LICENSE)). The model weights in `model/` are Creative Commons
Attribution-NonCommercial 4.0 International ([model/LICENSE](model/LICENSE)). A larger model is available under a
commercial licence from sup@terrific.life.

## Author and citation

miti-kiri is written and maintained by Supriya Wickrematillake (sup@terrific.life). If you use it, credit
"miti-kiri 1.0 by Supriya Wickrematillake" and link to this repository; [CITATION.cff](CITATION.cff) has the
citation in a form GitHub and reference managers read.
