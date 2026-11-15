// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! How many windows a separation will run, before decoding: the `--progress` total.
use kiri_core::windowing::Plan;
use std::io::Read;
use std::path::Path;
use std::process::Command;

/// Length in 44.1 kHz frames: a WAV's header when the input is one, else ffprobe's duration.
pub fn input_frames(path: &Path) -> Option<u64> {
    wav_frames(path).or_else(|| ffprobe_frames(path))
}

/// Windows the pipeline runs over `frames` samples (windows of `seg` samples): one per step until the step
/// passes the end.
pub fn windows_for(frames: u64, overlap: f64, seg: usize) -> usize {
    let step = Plan::new(overlap, seg).step as u64;
    frames.div_ceil(step) as usize
}

fn wav_frames(path: &Path) -> Option<u64> {
    let mut head = vec![0u8; 4096];
    let n = std::fs::File::open(path).ok()?.read(&mut head).ok()?;
    let head = &head[..n];
    if n < 12 || &head[0..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return None;
    }
    let u16_at = |i: usize| u16::from_le_bytes([head[i], head[i + 1]]) as u64;
    let u32_at = |i: usize| u32::from_le_bytes([head[i], head[i + 1], head[i + 2], head[i + 3]]) as u64;
    let (mut at, mut fmt) = (12usize, None);
    while at + 8 <= n {
        let size = u32_at(at + 4);
        match &head[at..at + 4] {
            b"fmt " if at + 24 <= n => fmt = Some((u16_at(at + 10), u32_at(at + 12), u16_at(at + 22))),
            b"data" => {
                let (channels, rate, bits) = fmt?;
                let frame_bytes = channels * bits / 8;
                if frame_bytes == 0 || rate == 0 {
                    return None;
                }
                return Some(size / frame_bytes * 44100 / rate);
            }
            _ => {}
        }
        at += 8 + size as usize + (size as usize & 1);
    }
    None
}

fn ffprobe_frames(path: &Path) -> Option<u64> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
        .arg(path)
        .output()
        .ok()?;
    let secs: f64 = String::from_utf8(out.stdout).ok()?.trim().parse().ok()?;
    Some((secs * 44100.0).round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_the_input() {
        for seg in [127 * 1024, 254 * 1024] {
            let step = Plan::new(0.25, seg).step as u64;
            assert_eq!(windows_for(1, 0.25, seg), 1);
            assert_eq!(windows_for(step, 0.25, seg), 1);
            assert_eq!(windows_for(step + 1, 0.25, seg), 2);
        }
    }

    #[test]
    fn reads_a_wav_header() {
        let dir = std::env::temp_dir().join(format!("kiri-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wav");
        let mut w = crate::wav::WavWriter::create(&p).unwrap();
        w.write(&[0.0; 1000], &[0.0; 1000]).unwrap();
        w.finish().unwrap();
        assert_eq!(wav_frames(&p), Some(1000));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
