// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The window plan: step, cross-fade and the mirrored read past a file's end.
use crate::fft::HOP;

pub struct Plan {
    pub seg: usize,
    pub step: usize,
    pub fade: usize,
    pub w: Vec<f32>,
}

impl Plan {
    /// `seg`: samples per window. The step rounds half to even, as Python's `round` does (the reference
    /// implementation): it matters only at a tie (62 hops x 0.75 = 46.5 -> 46); a 127-hop window
    /// (95.25, 127, 63.5 -> 95, 127, 64) is the same either way.
    pub fn new(overlap: f64, seg: usize) -> Self {
        let step = HOP.max(((seg as f64 * (1.0 - overlap) / HOP as f64).round_ties_even() as usize) * HOP);
        let step = step.min(seg);
        let fade = seg - step;
        let mut w = vec![1.0f32; seg];
        if fade > 0 {
            let ramp: Vec<f32> = (0..fade)
                .map(|i| {
                    let x = if fade > 1 { std::f64::consts::PI * i as f64 / (fade - 1) as f64 } else { 0.0 };
                    (0.5 - 0.5 * x.cos()) as f32
                })
                .collect();
            for i in 0..fade {
                w[i] = ramp[i];
                w[seg - 1 - i] = ramp[i];
            }
        }
        Self { seg, step, fade, w }
    }

    /// Weights for window `idx`: the first has no fade-in.
    pub fn weights(&self, idx: usize) -> Vec<f32> {
        let mut w = self.w.clone();
        if idx == 0 {
            for v in w[..self.fade].iter_mut() {
                *v = 1.0;
            }
        }
        w
    }
}

/// Where position `p` of a track of `n` frames reads from once past its end: the audio mirrored there (edge
/// sample repeated, as torch "symmetric" padding / numpy `reflect` of the reversed tail), and for a track shorter
/// than the span, mirrored back and forth (period 2n). Zero padding made a model fade whatever sounded at the
/// file's end.
pub fn mirror_index(p: usize, n: usize) -> usize {
    let m = p % (2 * n);
    if m < n { m } else { 2 * n - 1 - m }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrors_past_the_end() {
        // reflect at the end, edge sample repeated; a track shorter than the span zigzags
        let v: Vec<usize> = (0..12).map(|p| mirror_index(p, 4)).collect();
        assert_eq!(v, [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]);
        assert_eq!(mirror_index(5, 1000), 5);
        assert_eq!(mirror_index(1003, 1000), 996);
    }

    #[test]
    fn steps_as_python_round() {
        // 128 frames: unchanged by the rounding rule
        let w128 = 127 * HOP;
        assert_eq!([0.0, 0.25, 0.5].map(|o| Plan::new(o, w128).step / HOP), [127, 95, 64]);
        // a tie: 46.5 rounds to 46, as Python's round(46.5)
        let tie = 62 * HOP;
        assert_eq!([0.0, 0.25, 0.5].map(|o| Plan::new(o, tie).step / HOP), [62, 46, 31]);
    }
}
