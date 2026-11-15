// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The core under miti-kiri: decode, FFT, windowing, the STFT stage and the network backend interface
//! (Core ML today; ONNX would sit beside it). Nothing here knows a model's purpose.
pub mod backend;
#[cfg(target_os = "macos")]
pub mod coreml;
pub mod decode;
pub mod fft;
pub mod spec;
pub mod windowing;
