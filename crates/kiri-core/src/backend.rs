// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The network behind one call, so another runtime (ONNX) can be a second backend later.
use crate::spec::Geometry;
use anyhow::Result;

/// How a backend left its output in the buffer it was given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    /// batch x out_len f32 values.
    F32,
    /// batch x out_len fp16 values (raw bits) packed at the start of the same buffer; the spectrum stage
    /// widens them while it reads, off the network's thread.
    F16,
}

pub trait Backend {
    /// One network call on `geometry().batch` windows. `input`: normalised spectra, layout (batch, 4, 2049, frames)
    /// f32. `output`: (batch, 4, 4, 2049, frames) f32, stems in order drums, bass, other, vocals; channels
    /// [L.re, L.im, R.re, R.im].
    fn separate_batch(&mut self, input: &[f32], output: &mut [f32]) -> Result<Format>;
    fn geometry(&self) -> Geometry;
}

/// Identity-ish stand-in for tests of the signal path: every stem gets a quarter of the input.
pub struct Passthrough(pub Geometry);

impl Backend for Passthrough {
    fn separate_batch(&mut self, input: &[f32], output: &mut [f32]) -> Result<Format> {
        let n = self.0.in_len();
        for (b, x) in input.chunks(n).enumerate() {
            let y = &mut output[b * 4 * n..(b + 1) * 4 * n];
            for s in 0..4 {
                for (o, i) in y[s * n..(s + 1) * n].iter_mut().zip(x) {
                    *o = *i * 0.25;
                }
            }
        }
        Ok(Format::F32)
    }
    fn geometry(&self) -> Geometry {
        self.0
    }
}
