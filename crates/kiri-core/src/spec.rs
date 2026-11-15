// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The spectrum in and out of the network: `student.cpu_mix_to_input` / `cpu_output_to_wave`.
use crate::fft::*;

pub const STEMS: [&str; 4] = ["drums", "bass", "other", "vocals"];
const SCALE: f32 = 1.0 / 64.0; // 1/sqrt(4096): torch's normalized=True

/// torch's reflect padding index for center=True.
#[inline]
pub fn reflect(j: isize, n: usize) -> usize {
    let n = n as isize;
    let mut j = j;
    if j < 0 {
        j = -j;
    }
    if j >= n {
        j = 2 * (n - 1) - j;
    }
    j as usize
}

/// A network output element: f32, or fp16 bits (u16) as Core ML produces them.
pub trait Sample: Copy {
    fn to_f32(self) -> f32;
}
impl Sample for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}
impl Sample for u16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f16_bits_to_f32(self)
    }
}

#[inline(always)]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let normal = ((e + 112) << 23) | (m << 13);
    let sub = ((m as f32) * (1.0 / 16777216.0)).to_bits();
    let special = 0x7f80_0000 | (m << 13);
    let bits = if e == 0 { sub } else if e == 31 { special } else { normal };
    f32::from_bits(sign | bits)
}

/// The model's window: frames per window and windows per network call. miti-kiri 1.0: 128 x 1 (130048
/// samples). Read from the model (`Geometry::from_model`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub frames: usize,
    pub batch: usize,
}

impl Geometry {
    pub const V1: Geometry = Geometry { frames: 128, batch: 1 };

    /// Samples per window: (frames - 1) * HOP (center=True STFT gives frames = seg / HOP + 1).
    pub fn seg(&self) -> usize {
        (self.frames - 1) * HOP
    }
    /// f32 values of one window's input, layout (4, BINS, frames).
    pub fn in_len(&self) -> usize {
        4 * BINS * self.frames
    }
    /// f32 values of one window's output, layout (4 stems, 4, BINS, frames).
    pub fn out_len(&self) -> usize {
        4 * self.in_len()
    }

    /// The input shape of a compiled Core ML model, from its `model.mil` signature
    /// (`func main<..>(tensor<fp16, [B, 4, 2049, T]> mix_spec)`), without loading it, so decoding can start
    /// while the model loads.
    pub fn from_model(path: &std::path::Path) -> anyhow::Result<Geometry> {
        let mil = std::fs::read_to_string(path.join("model.mil"))
            .map_err(|e| anyhow::anyhow!("reading {}/model.mil: {e}", path.display()))?;
        Self::from_mil(&mil).ok_or_else(|| anyhow::anyhow!("{}: no `mix_spec` input of shape [B, 4, 2049, T] in model.mil", path.display()))
    }

    pub fn from_mil(mil: &str) -> Option<Geometry> {
        let line = mil.lines().find(|l| l.contains("func main") && l.contains("> mix_spec"))?;
        let at = line.find("> mix_spec")?;
        let open = line[..at].rfind('[')?;
        let close = open + line[open..].find(']')?;
        let dims: Vec<usize> = line[open + 1..close].split(',').map(|d| d.trim().parse().ok()).collect::<Option<_>>()?;
        match dims[..] {
            [b, 4, BINS, t] if b >= 1 && t >= 2 => Some(Geometry { frames: t, batch: b }),
            _ => None,
        }
    }
}

pub struct WindowDsp {
    frames: usize,
    seg: usize,
    fft: PairFft,
    win: Vec<f32>,
    env: Vec<f32>,   // window envelope for the inverse, padded coordinates
    tmp: Vec<C32>,   // [4][T][BINS] scratch for the transposes
    wl: Vec<f32>,
    wr: Vec<f32>,
    xl: Vec<C32>,
    xr: Vec<C32>,
    ola: Vec<f32>,
}

impl WindowDsp {
    pub fn new(frames: usize) -> Self {
        let seg = (frames - 1) * HOP;
        let win = hann();
        let padded = seg + N;
        let mut env = vec![0.0f32; padded];
        for t in 0..frames {
            for i in 0..N {
                env[t * HOP + i] += win[i] * win[i];
            }
        }
        Self {
            frames,
            seg,
            fft: PairFft::new(),
            win,
            env,
            tmp: vec![C32::new(0.0, 0.0); 2 * frames * BINS],
            wl: vec![0.0; N],
            wr: vec![0.0; N],
            xl: vec![C32::new(0.0, 0.0); BINS],
            xr: vec![C32::new(0.0, 0.0); BINS],
            ola: vec![0.0; 2 * padded],
        }
    }

    /// l, r: seg samples. Fills `out` (in_len, layout [4][BINS][frames]) normalised; returns (mean, std).
    pub fn mix_to_input(&mut self, l: &[f32], r: &[f32], out: &mut [f32]) -> (f32, f32) {
        let (frames, seg) = (self.frames, self.seg);
        debug_assert!(l.len() == seg && r.len() == seg && out.len() == 4 * BINS * frames);
        let mut ol = vec![C32::new(0.0, 0.0); BINS];
        let mut or = vec![C32::new(0.0, 0.0); BINS];
        for t in 0..frames {
            for i in 0..N {
                let j = reflect((t * HOP + i) as isize - (N / 2) as isize, seg);
                self.wl[i] = l[j] * self.win[i];
                self.wr[i] = r[j] * self.win[i];
            }
            self.fft.forward(&self.wl, &self.wr, &mut ol, &mut or);
            for k in 0..BINS {
                out[(0 * BINS + k) * frames + t] = ol[k].re * SCALE;
                out[(1 * BINS + k) * frames + t] = ol[k].im * SCALE;
                out[(2 * BINS + k) * frames + t] = or[k].re * SCALE;
                out[(3 * BINS + k) * frames + t] = or[k].im * SCALE;
            }
        }
        let n = out.len() as f64;
        let mean = out.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = out.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = var.sqrt();
        let (m32, s32) = (mean as f32, std as f32);
        let den = 1e-5 + s32;
        for v in out.iter_mut() {
            *v = (*v - m32) / den;
        }
        (m32, s32)
    }

    /// y: out_len, layout [4 stems][4][BINS][frames]. Fills `wave` ([4 stems][2][seg]).
    pub fn output_to_wave<T: Sample>(&mut self, y: &[T], mean: f32, std: f32, wave: &mut [f32]) {
        let (frames, seg) = (self.frames, self.seg);
        let in_len = 4 * BINS * frames;
        let padded = seg + N;
        for s in 0..4 {
            let ys = &y[s * in_len..(s + 1) * in_len];
            // transpose to frame-major while undoing the normalisation
            for c in 0..4 {
                for k in 0..BINS {
                    let row = &ys[(c * BINS + k) * frames..(c * BINS + k + 1) * frames];
                    for t in 0..frames {
                        let v = row[t].to_f32() * std + mean;
                        let slot = &mut self.tmp[((c / 2) * frames + t) * BINS + k];
                        if c % 2 == 0 {
                            slot.re = v;
                        } else {
                            slot.im = v;
                        }
                    }
                }
            }
            // note: tmp holds L at [0..frames), R at [frames..2*frames) as complex bins
            self.ola.iter_mut().for_each(|v| *v = 0.0);
            let (ola_l, ola_r) = self.ola.split_at_mut(padded);
            for t in 0..frames {
                self.xl.copy_from_slice(&self.tmp[t * BINS..(t + 1) * BINS]);
                self.xr.copy_from_slice(&self.tmp[(frames + t) * BINS..(frames + t + 1) * BINS]);
                self.fft.inverse(&self.xl, &self.xr, &mut self.wl, &mut self.wr);
                for i in 0..N {
                    ola_l[t * HOP + i] += self.wl[i] * self.win[i] * SCALE;
                    ola_r[t * HOP + i] += self.wr[i] * self.win[i] * SCALE;
                }
            }
            for i in 0..seg {
                let e = self.env[i + N / 2];
                wave[(s * 2) * seg + i] = ola_l[i + N / 2] / e;
                wave[(s * 2 + 1) * seg + i] = ola_r[i + N / 2] / e;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_from_the_model_signature() {
        let s1 = "program(1.0)\n{\n    func main<ios16>(tensor<fp16, [1, 4, 2049, 128]> mix_spec) {\n";
        assert_eq!(Geometry::from_mil(s1), Some(Geometry::V1));
        let b2 = "{\n    func main<ios16>(tensor<fp32, [2, 4, 2049, 96]> mix_spec) {\n";
        assert_eq!(Geometry::from_mil(b2), Some(Geometry { frames: 96, batch: 2 }));
        assert_eq!(Geometry { frames: 96, batch: 2 }.seg(), 97280);
        assert_eq!(Geometry::V1.seg(), 130048);
        assert_eq!(Geometry::from_mil("func main<ios16>(tensor<fp16, [1, 2, 2049, 128]> mix_spec)"), None);
        assert_eq!(Geometry::from_mil("func main<ios16>(tensor<fp16, [1, 4, 2049, 128]> x)"), None);
    }
}
