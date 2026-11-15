// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The FFT behind a small trait, so a vDSP backend can replace rustfft later.
use rustfft::{num_complex::Complex32, Fft as RFft, FftPlanner};
use std::sync::Arc;

pub type C32 = Complex32;
pub const N: usize = 4096;
pub const HOP: usize = 1024;
pub const BINS: usize = N / 2 + 1;

/// Unnormalised complex FFT of length `N`, in place. One instance per thread.
pub trait FftBackend: Send {
    fn forward(&mut self, buf: &mut [C32]);
    fn inverse(&mut self, buf: &mut [C32]);
}

pub struct RustFft {
    fwd: Arc<dyn RFft<f32>>,
    inv: Arc<dyn RFft<f32>>,
    scratch: Vec<C32>,
}

impl RustFft {
    pub fn new() -> Self {
        let mut p = FftPlanner::<f32>::new();
        let fwd = p.plan_fft_forward(N);
        let inv = p.plan_fft_inverse(N);
        let scratch = vec![C32::new(0.0, 0.0); fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len())];
        Self { fwd, inv, scratch }
    }
}

impl FftBackend for RustFft {
    fn forward(&mut self, buf: &mut [C32]) {
        self.fwd.process_with_scratch(buf, &mut self.scratch);
    }
    fn inverse(&mut self, buf: &mut [C32]) {
        self.inv.process_with_scratch(buf, &mut self.scratch);
    }
}

pub fn new_fft() -> Box<dyn FftBackend> {
    Box::new(RustFft::new())
}

/// Periodic Hann window, as torch.hann_window.
pub fn hann() -> Vec<f32> {
    (0..N).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos()) as f32).collect()
}

/// Two real signals through one complex FFT: forward transform of a windowed frame pair.
/// `l`, `r` are 4096 real samples (already windowed); out_* get bins 0..=2048.
pub struct PairFft {
    fft: Box<dyn FftBackend>,
    z: Vec<C32>,
}

impl PairFft {
    pub fn new() -> Self {
        Self { fft: new_fft(), z: vec![C32::new(0.0, 0.0); N] }
    }

    pub fn forward(&mut self, l: &[f32], r: &[f32], out_l: &mut [C32], out_r: &mut [C32]) {
        for i in 0..N {
            self.z[i] = C32::new(l[i], r[i]);
        }
        self.fft.forward(&mut self.z);
        for k in 0..BINS {
            let a = self.z[k];
            let b = self.z[(N - k) % N].conj();
            out_l[k] = (a + b) * 0.5;
            let d = a - b; // = 2i * XR
            out_r[k] = C32::new(d.im * 0.5, -d.re * 0.5);
        }
    }

    /// Inverse of two Hermitian half spectra; returns unnormalised (x N) real signals in l, r.
    pub fn inverse(&mut self, xl: &[C32], xr: &[C32], l: &mut [f32], r: &mut [f32]) {
        for k in 0..BINS {
            let mut a = xl[k];
            let mut b = xr[k];
            if k == 0 || k == N / 2 {
                a.im = 0.0;
                b.im = 0.0;
            }
            let ib = C32::new(-b.im, b.re); // i * b
            self.z[k] = a + ib;
            if k != 0 && k != N / 2 {
                self.z[N - k] = a.conj() + C32::new(b.im, b.re); // conj(a) + i*conj(b)
            }
        }
        self.fft.inverse(&mut self.z);
        for i in 0..N {
            l[i] = self.z[i].re;
            r[i] = self.z[i].im;
        }
    }
}
