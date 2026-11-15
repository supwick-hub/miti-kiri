# Changelog

## 1.0.0 (2026-10-19)

First release of the runtime in its own repository, with the 1.0 model.

- `kiri-core`: decode (ffmpeg), FFT, windowing with raised-cosine cross-fades and a mirrored read past the
  file's end, the STFT stage, the `Backend` interface, Core ML on the GPU.
- `miti-kiri`: the pipelined separation (decode, network, overlap-add, `tf` consistency, WAV writing), overlap
  0 / 0.25 / 0.5, 32-bit float or 16-bit dithered output, progress on stderr.
- `model/miti-kiri-1.0.mlmodelc` and the folded variant `model/miti-kiri-1.0-fold.mlmodelc`.
- Tests: unit tests, batch 2 == batch 1 bit for bit, the dependency direction, and `tools/golden.sh` (the model's
  outputs hashed on a synthetic mix).
