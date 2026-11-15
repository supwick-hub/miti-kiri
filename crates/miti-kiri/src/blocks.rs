// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The final audio, block by block, offered to an optional consumer. The separator knows only this trait;
//! whoever hosts it decides what consumes the blocks (a waveform overview, say).
use anyhow::Result;

/// Blocks of the mix (source 0) and of each stem (source 1 + stem index: drums, bass, other, vocals) in
/// time order per source, from any thread. The stem blocks are what the WAV files hold (16-bit with dither
/// when `--pcm16`), the mix block what was decoded. Dropping the sink ends the stream.
pub trait BlockSink: Send + Sync {
    fn push(&self, source: usize, l: Vec<f32>, r: Vec<f32>) -> Result<()>;
}
