// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The separator's command line, as a library so a host can add flags and a sink without repeating the rest. `miti-kiri` itself is `Args` and `run` with no sink.
use crate::blocks::BlockSink;
use crate::consist::Mode;
use crate::pipeline::{separate, Options, Stats};
use crate::probe;
use anyhow::{bail, Result};
use clap::{Args as ClapArgs, ValueEnum};
use kiri_core::spec::Geometry;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, ValueEnum)]
pub enum ConsistArg {
    None,
    Tf,
}

#[derive(ClapArgs)]
pub struct Args {
    /// Audio file (anything ffmpeg reads)
    pub input: PathBuf,
    /// Output directory; gets drums.wav, bass.wav, other.wav, vocals.wav
    #[arg(short, long)]
    pub out: PathBuf,
    /// Compiled Core ML model (.mlmodelc). Its input shape (B,4,2049,T) sets the window
    #[arg(short, long)]
    pub model: PathBuf,
    /// Frames per window; read from the model, so only a check (or the window for --passthrough, default 128)
    #[arg(long)]
    pub frames: Option<usize>,
    /// Windows per network call; read from the model, so only a check (or the batch for --passthrough, default 1)
    #[arg(long)]
    pub batch: Option<usize>,
    /// Window overlap: 0, 0.25 or 0.5
    #[arg(long, default_value_t = 0.25)]
    pub overlap: f64,
    /// Make the stems add up to the mix
    #[arg(long, value_enum, default_value = "tf")]
    pub consist: ConsistArg,
    /// Run in macOS's background state (library work); without it, leave any inherited one
    #[arg(long)]
    pub background: bool,
    /// 16-bit WAV with TPDF dither instead of 32-bit float
    #[arg(long)]
    pub pcm16: bool,
    /// Print `preparing`, then `progress <done> <total>` per window, to stderr
    #[arg(long)]
    pub progress: bool,
    /// Use a pass-through backend instead of Core ML (tests the signal path only)
    #[arg(long, hide = true)]
    pub passthrough: bool,
}

#[cfg(target_os = "macos")]
extern "C" {
    fn setpriority(which: i32, who: u32, prio: i32) -> i32;
}

/// Out of macOS's background state, which a throttled parent passes on: there it ran
/// ~1.5x slower on a quiet Mac, more under load (2026-10-06). Into it with `--background`, for
/// library work that must not starve playback.
pub fn set_background(on: bool) {
    #[cfg(target_os = "macos")]
    unsafe {
        const PRIO_DARWIN_PROCESS: i32 = 4;
        const PRIO_DARWIN_BG: i32 = 0x1000;
        setpriority(PRIO_DARWIN_PROCESS, 0, if on { PRIO_DARWIN_BG } else { 0 });
    }
}

/// One file in, four stems out, printing the timing line at the end. `blocks` also receives the final
/// mix and stem blocks; it is dropped when this returns, so the caller can finish what it feeds.
pub fn run(a: &Args, blocks: Option<Arc<dyn BlockSink>>) -> Result<Stats> {
    set_background(a.background);
    crate::wav::PCM16.store(a.pcm16, std::sync::atomic::Ordering::Relaxed);
    if ![0.0, 0.25, 0.5].contains(&a.overlap) {
        bail!("--overlap must be 0, 0.25 or 0.5");
    }
    let passthrough = a.passthrough;
    let geometry = if passthrough {
        Geometry { frames: a.frames.unwrap_or(128), batch: a.batch.unwrap_or(1) }
    } else {
        let g = Geometry::from_model(&a.model)?;
        if a.frames.is_some_and(|f| f != g.frames) || a.batch.is_some_and(|b| b != g.batch) {
            bail!("{} takes {} frames x batch {}; --frames/--batch disagree", a.model.display(), g.frames, g.batch);
        }
        g
    };
    if geometry.frames < 2 || geometry.batch < 1 {
        bail!("--frames must be >= 2 and --batch >= 1");
    }
    let progress = a.progress.then(|| probe::input_frames(&a.input).map_or(0, |f| probe::windows_for(f, a.overlap, geometry.seg())));
    let opt = Options {
        geometry,
        overlap: a.overlap,
        consist: match a.consist { ConsistArg::None => Mode::None, ConsistArg::Tf => Mode::Tf },
        progress,
        blocks,
    };
    let model = a.model.clone();
    let make = move || -> Result<Box<dyn kiri_core::backend::Backend>> {
        if passthrough {
            return Ok(Box::new(kiri_core::backend::Passthrough(geometry)));
        }
        #[cfg(target_os = "macos")]
        {
            Ok(Box::new(kiri_core::coreml::CoreMl::load(&model, geometry)?))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = model;
            bail!("only the Core ML backend exists so far")
        }
    };
    let st = separate(&a.input, &a.out, &opt, make)?;
    let audio = st.samples as f64 / 44100.0;
    eprintln!(
        "{:.1} s of audio, {} windows; model load {:.2} s; network {:.2} s ({:.1} ms/window); wall {:.2} s",
        audio, st.windows, st.load.as_secs_f64(), st.network.as_secs_f64(), 1000.0 * st.network.as_secs_f64() / st.windows.max(1) as f64, st.total.as_secs_f64()
    );
    Ok(st)
}
