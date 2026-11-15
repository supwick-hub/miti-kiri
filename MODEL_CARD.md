# miti-kiri 1.0

Splits stereo music into four stems: drums, bass, other, vocals.

## The model

| | |
|---|---|
| Author | Supriya Wickrematillake (sup@terrific.life) |
| Input | 44.1 kHz stereo audio |
| Network input | `mix_spec` (1, 4, 2049, 128): the real and imaginary parts of the left and right STFT (n_fft 4096, hop 1024, Hann), one window of 128 frames = 127 x 1024 samples = 2.95 s, each window normalised by its own mean and standard deviation |
| Network output | `stems_spec` (1, 4, 4, 2049, 128): four stems, the same layout |
| Parameters | 4.67 M (plain); 4.28 M (folded) |
| Files | `model/miti-kiri-1.0.mlmodelc` (9.4 MB, fp16, Core ML); `model/miti-kiri-1.0-fold.mlmodelc` (8.7 MB) |
| Runs on | the Mac's GPU through Core ML (the Neural Engine is not used; see the README) |
| Memory | about 300 MB at any track length (peak footprint 223 to 303 MB on tracks of 3 to 60 minutes, M1 MacBook Air 8 GB) |

**The folded variant** is the same network with one repeated fusion step in the decoder folded into its weights:
the same function with half the multiply-adds. Against the plain export it agrees to 65.6 dB (fp16 rounding;
not bit-identical); the plain export agrees with the fp32 network to 61.9 dB and the folded one to 61.8 dB.
It is about 10-15 % faster.

**Post-processing.** The runtime overlaps windows (0.25 by default) with raised-cosine cross-fades, and with
`--consist tf` shares the leftover of the mix among the stems by each stem's power share per bin, so the stems
add up to the mix.

## Numbers

SDR against true stems: MUSDB18-HQ test set (50 songs) and six multitrack songs from MoisesDB. "uSDR" is the
whole-track energy ratio, 10 log10(sum ref^2 / sum (ref - est)^2), median over tracks, in dB, scored on the
fp32 network with the reference implementation (the Core ML build agrees with it to 61.9 dB). The other rows
are other published separators scored the same way, for scale.

| MUSDB18-HQ test, 50 songs | drums | bass | other | vocals | mean |
|---|---|---|---|---|---|
| **miti-kiri 1.0** | 7.47 | 5.10 | 4.81 | 7.42 | 6.20 |
| htdemucs_ft | 9.95 | 9.10 | 6.82 | 9.65 | 8.88 |
| SCNet (MSST settings) | 10.66 | 9.23 | 7.42 | 10.31 | 9.41 |
| BS-RoFormer | 11.38 | 8.76 | 7.99 | 11.55 | 9.92 |

| MoisesDB, 6 songs | drums | bass | other | vocals | mean |
|---|---|---|---|---|---|
| **miti-kiri 1.0** | 8.81 | 9.41 | 2.39 | 5.94 | 6.64 |
| htdemucs_ft | 12.46 | 10.70 | 4.44 | 7.60 | 8.80 |
| SCNet (MSST settings) | 11.06 | 9.80 | 2.96 | 5.88 | 7.43 |
| BS-RoFormer | 11.59 | 9.58 | 3.07 | 6.69 | 7.73 |

The MUSDB18-HQ test set is for evaluation only (its licence is non-commercial). miti-kiri 1.0 is a small, fast
model: it trails the large separators by 2-4 dB on MUSDB18 and is nearer them on drums and bass of the music it
was built for.

## Mac time

One track, 512.6 s long (stereo m4a), overlap 0.25, `tf` consistency, GPU:

| Mac | network | wall | per second of audio |
|---|---|---|---|
| M1 MacBook Air, 8 GB (quiet, recorded 2026-10-09) | 18.2 s | 18.5 s | 36 ms (28x realtime) |
| M1 MacBook Air, 8 GB, this release's build, other work running (load average about 4), 342 s and 360 s tracks | 20.2 s, 20.5 s | 20.5 s, 20.7 s | 60 ms, 58 ms (17x realtime) |

Time per track is the wall time to write four WAVs from an m4a; decoding overlaps the network and adds
about 0.25 s. The model loads in under a second. Memory stays small because nothing is decoded whole: windows
stream through bounded queues.

## Limits

- Built for electronic music: on MUSDB18 (rock and pop) it trails the large separators by 2-4 dB, as the
  table shows, and is nearer them on drums and bass of electronic tracks.
- On instrumentals the vocal stem sits about 50 dB below the mix, and in quiet passages (a share of 3-second
  chunks above -40 dB) it is audible in roughly one chunk in five.
- The training code and data are not part of this repository.
