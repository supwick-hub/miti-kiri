// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! Streaming WAV writer, 32-bit float or (with `--pcm16`) 16-bit with TPDF dither; the header is
//! patched when the length is known.
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// 16-bit output, for readers that take 16-bit stems without converting.
pub static PCM16: AtomicBool = AtomicBool::new(false);

pub struct WavWriter {
    w: BufWriter<File>,
    frames: u64,
    buf: Vec<u8>,
    pcm16: bool,
    rng: u64,
    /// What the file now holds, as a reader sees it, when asked for (`--lanes`).
    echo: Option<(Vec<f32>, Vec<f32>)>,
}

impl WavWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        w.write_all(&[0u8; 44])?;
        // Seeded from the name, so a file's noise is the same run to run.
        let rng = path.file_name().map_or(0, |n| n.as_encoded_bytes().iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3)));
        Ok(Self { w, frames: 0, buf: Vec::new(), pcm16: PCM16.load(Ordering::Relaxed), rng: rng | 1, echo: None })
    }

    /// Uniform in [0, 1), xorshift64*.
    fn uniform(&mut self) -> f64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// TPDF at 1 LSB peak-to-peak, round half away, clamped.
    fn q16(&mut self, x: f32) -> i16 {
        let d = self.uniform() - self.uniform();
        (x as f64 * 32767.0 + d + 0.5).floor().clamp(-32768.0, 32767.0) as i16
    }

    /// From now on keep each block as written (quantised and dithered, if 16-bit), for `take_echo`.
    pub fn keep_echo(&mut self) {
        self.echo = Some((Vec::new(), Vec::new()));
    }

    /// The last block's samples as the file holds them, as f32 (i16 / 32768, as a reader scales).
    pub fn take_echo(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        self.echo.take().map(|e| {
            self.echo = Some((Vec::new(), Vec::new()));
            e
        })
    }

    /// Append stereo frames from two channel slices.
    pub fn write(&mut self, l: &[f32], r: &[f32]) -> Result<()> {
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        let mut echo = self.echo.take();
        for i in 0..l.len() {
            if self.pcm16 {
                let (a, b) = (self.q16(l[i]), self.q16(r[i]));
                if let Some((el, er)) = echo.as_mut() {
                    el.push(a as f32 / 32768.0);
                    er.push(b as f32 / 32768.0);
                }
                buf.extend_from_slice(&a.to_le_bytes());
                buf.extend_from_slice(&b.to_le_bytes());
            } else {
                if let Some((el, er)) = echo.as_mut() {
                    el.push(l[i]);
                    er.push(r[i]);
                }
                buf.extend_from_slice(&l[i].to_le_bytes());
                buf.extend_from_slice(&r[i].to_le_bytes());
            }
        }
        self.echo = echo;
        self.w.write_all(&buf)?;
        self.buf = buf;
        self.frames += l.len() as u64;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        let (fmt, bytes): (u16, u32) = if self.pcm16 { (1, 2) } else { (3, 4) };
        let data = (self.frames * 2 * bytes as u64) as u32;
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + data).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&fmt.to_le_bytes()); // 1 PCM, 3 IEEE float
        h.extend_from_slice(&2u16.to_le_bytes());
        h.extend_from_slice(&44100u32.to_le_bytes());
        h.extend_from_slice(&(44100u32 * 2 * bytes).to_le_bytes());
        h.extend_from_slice(&((2 * bytes) as u16).to_le_bytes());
        h.extend_from_slice(&((8 * bytes) as u16).to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&data.to_le_bytes());
        self.w.seek(SeekFrom::Start(0))?;
        self.w.write_all(&h)?;
        self.w.flush()?;
        Ok(())
    }
}
