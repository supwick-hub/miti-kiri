// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! Stems that add up to the mix (`window_test.consist(..., "tf")`), streaming: STFT of each stem and of
//! the mix (n_fft 4096, hop 1024, Hann, torch defaults: not normalised), the leftover shared by each
//! stem's power share per bin, inverse STFT, then the WAV files. Frames are processed as soon as the
//! overlap-added stems are final, so the work hides behind the network.
use kiri_core::fft::*;
use crate::blocks::BlockSink;
use std::sync::Arc;
use crate::pipeline::ConsistMsg;
use kiri_core::spec::reflect;
use crate::wav::WavWriter;
use anyhow::Result;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    None,
    Tf,
}

pub struct Consist {
    mode: Mode,
    writers: Vec<WavWriter>,
    // tf state
    fft: PairFft,
    win: Vec<f32>,
    w2: Vec<f32>,
    mix: [Vec<f32>; 2],
    mbase: usize,
    est: Vec<Vec<f32>>, // 8
    ebase: usize,
    n: Option<usize>,
    next_t: usize,
    ola: Vec<Vec<f32>>, // 8, padded coordinates
    obase: usize,
    emitted: usize,
    wl: Vec<f32>,
    wr: Vec<f32>,
    spec: Vec<Vec<C32>>, // 5 signals x 2 channels, BINS each
    blocks: Option<Arc<dyn BlockSink>>,
}

impl Consist {
    pub fn new(mode: Mode, paths: &[PathBuf], blocks: Option<Arc<dyn BlockSink>>) -> Result<Self> {
        let mut writers = paths.iter().map(|p| WavWriter::create(p)).collect::<Result<Vec<_>>>()?;
        if blocks.is_some() {
            writers.iter_mut().for_each(|w| w.keep_echo());
        }
        let win = hann();
        let w2 = win.iter().map(|w| w * w).collect();
        Ok(Self {
            mode,
            writers,
            fft: PairFft::new(),
            win,
            w2,
            mix: [Vec::new(), Vec::new()],
            mbase: 0,
            est: vec![Vec::new(); 8],
            ebase: 0,
            n: None,
            next_t: 0,
            ola: vec![Vec::new(); 8],
            obase: 0,
            emitted: 0,
            wl: vec![0.0; N],
            wr: vec![0.0; N],
            spec: vec![vec![C32::new(0.0, 0.0); BINS]; 10],
            blocks,
        })
    }

    fn feed_mix(&self, l: &[f32], r: &[f32]) -> Result<()> {
        match &self.blocks {
            Some(f) => f.push(0, l.to_vec(), r.to_vec()),
            None => Ok(()),
        }
    }

    /// The stem's block to its WAV, and, for the block sink, as the WAV now holds it.
    fn write_stem(&mut self, s: usize, l: &[f32], r: &[f32]) -> Result<()> {
        self.writers[s].write(l, r)?;
        if let (Some(f), Some((el, er))) = (&self.blocks, self.writers[s].take_echo()) {
            f.push(1 + s, el, er)?;
        }
        Ok(())
    }

    /// Returns false when the stream is over.
    pub fn handle(&mut self, m: ConsistMsg) -> Result<bool> {
        match (self.mode, m) {
            (Mode::None, ConsistMsg::Est(chunk, len)) => {
                for s in 0..4 {
                    self.write_stem(s, &chunk[2 * s * len..(2 * s + 1) * len], &chunk[(2 * s + 1) * len..(2 * s + 2) * len])?;
                }
                Ok(true)
            }
            (Mode::None, ConsistMsg::End(_)) => Ok(false),
            (Mode::None, ConsistMsg::Mix(l, r)) => {
                self.feed_mix(&l, &r)?;
                Ok(true)
            }
            (Mode::Tf, ConsistMsg::Mix(l, r)) => {
                self.feed_mix(&l, &r)?;
                self.mix[0].extend_from_slice(&l);
                self.mix[1].extend_from_slice(&r);
                Ok(true)
            }
            (Mode::Tf, ConsistMsg::Est(chunk, len)) => {
                for c in 0..8 {
                    self.est[c].extend_from_slice(&chunk[c * len..(c + 1) * len]);
                }
                self.process()?;
                Ok(true)
            }
            (Mode::Tf, ConsistMsg::End(n)) => {
                self.n = Some(n);
                self.process()?;
                Ok(false)
            }
        }
    }

    pub fn finish(self) -> Result<()> {
        for w in self.writers {
            w.finish()?;
        }
        Ok(())
    }

    fn process(&mut self) -> Result<()> {
        loop {
            let t = self.next_t;
            let f = self.ebase + self.est[0].len();
            let last = match self.n {
                Some(n) => {
                    if t > n / HOP {
                        break;
                    }
                    t == n / HOP
                }
                None => {
                    if HOP * t + N / 2 > f {
                        break;
                    }
                    false
                }
            };
            self.frame(t);
            self.next_t += 1;
            let e = if last { self.n.unwrap() } else { (HOP * (t + 1)).saturating_sub(N / 2) };
            let e = self.n.map_or(e, |n| e.min(n));
            let tmax = t;
            self.emit_to(e, tmax)?;
            let drop_to = (HOP * (t + 1)).saturating_sub(N / 2 + N);
            self.drain(drop_to);
        }
        Ok(())
    }

    fn drain(&mut self, to: usize) {
        if to > self.mbase && to - self.mbase > 1 << 16 {
            let k = to - self.mbase;
            for m in self.mix.iter_mut() {
                m.drain(..k);
            }
            self.mbase = to;
        }
        if to > self.ebase && to - self.ebase > 1 << 16 {
            let k = to - self.ebase;
            for e in self.est.iter_mut() {
                e.drain(..k);
            }
            self.ebase = to;
        }
    }

    fn frame(&mut self, t: usize) {
        let big = isize::MAX / 4;
        let nn = self.n.map_or(big as usize, |n| n);
        let mut idx = [0usize; N];
        for i in 0..N {
            idx[i] = reflect((HOP * t + i) as isize - (N / 2) as isize, nn);
        }
        // 4 stems then the mix
        for s in 0..5 {
            for i in 0..N {
                let j = idx[i];
                let (l, r) = if s < 4 {
                    (self.est[2 * s][j - self.ebase], self.est[2 * s + 1][j - self.ebase])
                } else {
                    (self.mix[0][j - self.mbase], self.mix[1][j - self.mbase])
                };
                self.wl[i] = l * self.win[i];
                self.wr[i] = r * self.win[i];
            }
            let (a, b) = self.spec.split_at_mut(2 * s + 1);
            self.fft.forward(&self.wl, &self.wr, &mut a[2 * s], &mut b[0]);
        }
        // tf split, per channel and bin
        for ch in 0..2 {
            for k in 0..BINS {
                let mut sum = C32::new(0.0, 0.0);
                let mut ps = [0.0f32; 4];
                let mut psum = 0.0f32;
                for s in 0..4 {
                    let v = self.spec[2 * s + ch][k];
                    sum += v;
                    ps[s] = v.norm_sqr();
                    psum += ps[s];
                }
                let r = self.spec[8 + ch][k] - sum;
                let den = psum + 1e-10;
                for s in 0..4 {
                    self.spec[2 * s + ch][k] += r * (ps[s] / den);
                }
            }
        }
        // inverse and overlap-add
        let need = HOP * t + N - self.obase;
        for o in self.ola.iter_mut() {
            if o.len() < need {
                o.resize(need, 0.0);
            }
        }
        let inv = 1.0 / N as f32;
        for s in 0..4 {
            let (a, b) = self.spec.split_at_mut(2 * s + 1);
            self.fft.inverse(&a[2 * s], &b[0], &mut self.wl, &mut self.wr);
            let off = HOP * t - self.obase;
            for i in 0..N {
                self.ola[2 * s][off + i] += self.wl[i] * self.win[i] * inv;
                self.ola[2 * s + 1][off + i] += self.wr[i] * self.win[i] * inv;
            }
        }
    }

    fn env(&self, p: usize, tmax: usize) -> f32 {
        let lo = if p >= N { (p - N) / HOP + 1 } else { 0 };
        let hi = (p / HOP).min(tmax);
        let mut e = 0.0f32;
        for t in lo..=hi {
            e += self.w2[p - HOP * t];
        }
        e
    }

    fn emit_to(&mut self, e: usize, tmax: usize) -> Result<()> {
        if e <= self.emitted {
            return Ok(());
        }
        let len = e - self.emitted;
        let env: Vec<f32> = (0..len).map(|i| self.env(self.emitted + N / 2 + i, tmax)).collect();
        let off = self.emitted + N / 2 - self.obase;
        let mut l = vec![0.0f32; len];
        let mut r = vec![0.0f32; len];
        for s in 0..4 {
            for i in 0..len {
                l[i] = self.ola[2 * s][off + i] / env[i];
                r[i] = self.ola[2 * s + 1][off + i] / env[i];
            }
            self.write_stem(s, &l, &r)?;
        }
        self.emitted = e;
        let to = self.emitted + N / 2;
        if to > self.obase {
            let k = to - self.obase;
            for o in self.ola.iter_mut() {
                let k = k.min(o.len());
                o.drain(..k);
            }
            self.obase = to;
        }
        Ok(())
    }
}

/// `Consist` over a range: the same frames, the same arithmetic, the same order, from an estimate held for
/// [x0, x0 + len) instead of streamed.
pub struct RangeConsist {
    fft: PairFft,
    win: Vec<f32>,
    w2: Vec<f32>,
    wl: Vec<f32>,
    wr: Vec<f32>,
    spec: Vec<Vec<C32>>,
}

impl Default for RangeConsist {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeConsist {
    pub fn new() -> Self {
        let win = hann();
        let w2 = win.iter().map(|w| w * w).collect();
        Self { fft: PairFft::new(), win, w2, wl: vec![0.0; N], wr: vec![0.0; N], spec: vec![vec![C32::new(0.0, 0.0); BINS]; 10] }
    }

    /// Output frames [s, e) of a track of `n` frames; `est` (8 channels) and `mix` (2) hold [x0, x0 + len),
    /// which must include [s - N, e + N) clipped to the track.
    pub fn run(&mut self, n: usize, x0: usize, est: &[Vec<f32>], mix: &[Vec<f32>; 2], s: usize, e: usize) -> Vec<Vec<f32>> {
        self.run_capture(n, x0, est, mix, s, e, None)
    }

    /// `run`, also keeping the final stems' STFT frames this range owns (`owned_rows`), mono, after the
    /// split: `(first_row, has_prev)` and `cap[(row * 4 + stem) * BINS + bin]` from the row before the first
    /// when there is one (for phase).
    #[allow(clippy::too_many_arguments)]
    pub fn run_capture(&mut self, n: usize, x0: usize, est: &[Vec<f32>], mix: &[Vec<f32>; 2], s: usize, e: usize, mut cap: Option<&mut Vec<C32>>) -> Vec<Vec<f32>> {
        let rows = owned_rows(n, s, e);
        let cap_lo = rows.start.saturating_sub(1);
        let t_lo = if s < N / 2 { 0 } else { (s - N / 2) / HOP + 1 };
        let t_hi = ((e - 1 + N / 2) / HOP).min(n / HOP);
        let base = HOP * t_lo; // padded coordinate of ola[0]
        let mut ola = vec![vec![0.0f32; (t_hi - t_lo) * HOP + N]; 8];
        let inv = 1.0 / N as f32;
        let mut idx = [0usize; N];
        for t in t_lo..=t_hi {
            for (i, v) in idx.iter_mut().enumerate() {
                *v = reflect((HOP * t + i) as isize - (N / 2) as isize, n) - x0;
            }
            for sg in 0..5 {
                for i in 0..N {
                    let j = idx[i];
                    let (l, r) = if sg < 4 { (est[2 * sg][j], est[2 * sg + 1][j]) } else { (mix[0][j], mix[1][j]) };
                    self.wl[i] = l * self.win[i];
                    self.wr[i] = r * self.win[i];
                }
                let (a, b) = self.spec.split_at_mut(2 * sg + 1);
                self.fft.forward(&self.wl, &self.wr, &mut a[2 * sg], &mut b[0]);
            }
            for ch in 0..2 {
                for k in 0..BINS {
                    let mut sum = C32::new(0.0, 0.0);
                    let mut ps = [0.0f32; 4];
                    let mut psum = 0.0f32;
                    for sg in 0..4 {
                        let v = self.spec[2 * sg + ch][k];
                        sum += v;
                        ps[sg] = v.norm_sqr();
                        psum += ps[sg];
                    }
                    let r = self.spec[8 + ch][k] - sum;
                    let den = psum + 1e-10;
                    for sg in 0..4 {
                        self.spec[2 * sg + ch][k] += r * (ps[sg] / den);
                    }
                }
            }
            if let Some(c) = cap.as_mut() {
                if t >= cap_lo && t < rows.end {
                    for sg in 0..4 {
                        let (l, r) = (&self.spec[2 * sg], &self.spec[2 * sg + 1]);
                        c.extend(l.iter().zip(r.iter()).map(|(a, b)| (a + b) * 0.5));
                    }
                }
            }
            let off = HOP * t - base;
            for sg in 0..4 {
                let (a, b) = self.spec.split_at_mut(2 * sg + 1);
                self.fft.inverse(&a[2 * sg], &b[0], &mut self.wl, &mut self.wr);
                for i in 0..N {
                    ola[2 * sg][off + i] += self.wl[i] * self.win[i] * inv;
                    ola[2 * sg + 1][off + i] += self.wr[i] * self.win[i] * inv;
                }
            }
        }
        let mut out = vec![vec![0.0f32; e - s]; 8];
        for p in s..e {
            let pp = p + N / 2;
            let lo = if pp >= N { (pp - N) / HOP + 1 } else { 0 };
            let hi = (pp / HOP).min(n / HOP);
            let mut env = 0.0f32;
            for t in lo..=hi {
                env += self.w2[pp - HOP * t];
            }
            for c in 0..8 {
                out[c][p - s] = ola[c][pp - base] / env;
            }
        }
        out
    }
}

/// The STFT rows (frame t centred on sample 1024 t, rows 0..=n/1024) output frames [s, e) own: centres in
/// [s, e), and the last row with the range that ends the track. Ranges that partition the track partition
/// its rows.
pub fn owned_rows(n: usize, s: usize, e: usize) -> std::ops::Range<usize> {
    let r0 = s.div_ceil(HOP);
    let r1 = if e >= n { n / HOP + 1 } else { e.div_ceil(HOP) };
    r0..r1.max(r0)
}
