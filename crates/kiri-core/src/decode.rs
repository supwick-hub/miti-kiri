// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! ffmpeg as a subprocess: f32le stereo 44.1 kHz on stdout, read in chunks.
use anyhow::{bail, Result};
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};

pub struct Decoder {
    child: Child,
    out: std::process::ChildStdout,
    rest: Vec<u8>,
}

impl Decoder {
    pub fn spawn(path: &Path) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-f", "f32le", "-ac", "2", "-ar", "44100", "-"])
            .stdout(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()?;
        let out = child.stdout.take().unwrap();
        Ok(Self { child, out, rest: Vec::new() })
    }

    /// Next chunk of up to `frames` stereo frames as (left, right); None at end of stream.
    pub fn next_chunk(&mut self, frames: usize) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let want = frames * 8;
        let mut buf = std::mem::take(&mut self.rest);
        let mut tmp = vec![0u8; 1 << 16];
        while buf.len() < want {
            let n = self.out.read(&mut tmp)?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
        }
        let take = buf.len().min(want) / 8 * 8;
        if take == 0 {
            return Ok(None);
        }
        self.rest = buf.split_off(take);
        let n = take / 8;
        let (mut l, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for f in buf.chunks_exact(8) {
            l.push(f32::from_le_bytes([f[0], f[1], f[2], f[3]]));
            r.push(f32::from_le_bytes([f[4], f[5], f[6], f[7]]));
        }
        Ok(Some((l, r)))
    }

    pub fn finish(mut self) -> Result<()> {
        let st = self.child.wait()?;
        if !st.success() {
            bail!("ffmpeg failed: {st}");
        }
        Ok(())
    }
}
