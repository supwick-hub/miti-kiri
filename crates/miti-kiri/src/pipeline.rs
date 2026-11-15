// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The pipelined separation: decode + spectrum in (thread), network (calling thread), spectrum out +
//! overlap-add (thread), stems-sum-to-mix + WAV writing (thread). Plain threads, bounded channels.
//! The window (frames) and the windows per network call (batch) are the model's (`Geometry`), e.g. 128 frames
//! one at a time or fewer frames two at a time. A batch's windows are prepared, run and finished
//! as one message; the last batch is padded with zero windows whose output is dropped.
use kiri_core::backend::{Backend, Format};
use crate::blocks::BlockSink;
use crate::consist::{Consist, Mode};
use kiri_core::decode::Decoder;
use kiri_core::spec::*;
use kiri_core::windowing::{mirror_index, Plan};
use anyhow::{anyhow, bail, Result};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Options {
    /// The model's window and batch; the backend must agree.
    pub geometry: Geometry,
    pub overlap: f64,
    pub consist: Mode,
    /// Machine-readable progress on stderr: `preparing` before the model loads, then
    /// `progress <done> <total>` per window through the network. `total` is this estimate of the
    /// window count (0: unknown) until the last window, which reports the exact count.
    pub progress: Option<usize>,
    /// Where the final mix and stem blocks also go, as the files hold them (a host's waveform overview, say).
    pub blocks: Option<Arc<dyn BlockSink>>,
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub windows: usize,
    pub network: Duration,
    pub load: Duration,
    pub total: Duration,
    pub samples: usize,
}

pub enum ConsistMsg {
    Mix(Vec<f32>, Vec<f32>),
    /// 8 channels (stem*2+ch) of `len` samples each, finalised, contiguous in time.
    Est(Vec<f32>, usize),
    End(usize),
}

/// One window of a batch: its index, first sample and normalisation.
#[derive(Clone, Copy)]
struct Win {
    idx: usize,
    a: usize,
    mean: f32,
    std: f32,
}

enum PrepMsg {
    /// `input`: batch x in_len; slots past `wins.len()` are zero.
    Batch { wins: Vec<Win>, input: Vec<f32> },
    End(usize),
}

enum DoneMsg {
    /// `y`: batch x out_len, as the backend left it (`fmt`).
    Batch { wins: Vec<Win>, y: Vec<f32>, fmt: Format },
    End(usize),
}

#[derive(Clone)]
struct Pool(Arc<Mutex<Vec<Vec<f32>>>>, usize);
impl Pool {
    fn new(len: usize) -> Self {
        Pool(Arc::new(Mutex::new(Vec::new())), len)
    }
    fn get(&self) -> Vec<f32> {
        self.0.lock().unwrap().pop().unwrap_or_else(|| vec![0.0; self.1])
    }
    fn put(&self, v: Vec<f32>) {
        self.0.lock().unwrap().push(v);
    }
}

/// `make_backend` runs on the calling thread after the decode/spectrum thread has started, so the
/// model load overlaps the first windows being prepared.
pub fn separate(
    input: &Path,
    out_dir: &Path,
    opt: &Options,
    make_backend: impl FnOnce() -> Result<Box<dyn Backend>>,
) -> Result<Stats> {
    let t0 = Instant::now();
    std::fs::create_dir_all(out_dir)?;
    let geom = opt.geometry;
    let (seg, in_len, out_len, batch) = (geom.seg(), geom.in_len(), geom.out_len(), geom.batch);
    let plan = Arc::new(Plan::new(opt.overlap, seg));
    let in_pool = Pool::new(batch * in_len);
    let out_pool = Pool::new(batch * out_len);
    let (prep_tx, prep_rx): (SyncSender<PrepMsg>, Receiver<PrepMsg>) = sync_channel(3);
    let (done_tx, done_rx): (SyncSender<DoneMsg>, Receiver<DoneMsg>) = sync_channel(2);
    let (con_tx, con_rx): (SyncSender<ConsistMsg>, Receiver<ConsistMsg>) = sync_channel(64);
    let paths: Vec<PathBuf> = STEMS.iter().map(|s| out_dir.join(format!("{s}.wav"))).collect();
    let mode = opt.consist;
    let con_tx_prep = con_tx.clone();
    let mut stats = Stats::default();
    let feeder = opt.blocks.clone();
    let send_mix = mode == Mode::Tf || feeder.is_some();

    std::thread::scope(|sc| -> Result<()> {
        // sink: consist + writer
        let sink = sc.spawn(move || -> Result<()> {
            let mut c = Consist::new(mode, &paths, feeder)?;
            while let Ok(m) = con_rx.recv() {
                if !c.handle(m)? {
                    break;
                }
            }
            c.finish()
        });

        // prep: decode, window, spectrum in
        let p2 = plan.clone();
        let ip = in_pool.clone();
        let input_path = input.to_path_buf();
        let prep = sc.spawn(move || -> Result<()> {
            let plan = p2;
            let mut dec = Decoder::spawn(&input_path)?;
            let (mut bl, mut br): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
            let mut base = 0usize;
            let mut decoded = 0usize;
            let mut eof = false;
            let mut next_a = 0usize;
            let mut idx = 0usize;
            let mut dsp = WindowDsp::new(geom.frames);
            let mut cl = vec![0.0f32; seg];
            let mut cr = vec![0.0f32; seg];
            let mut wins: Vec<Win> = Vec::with_capacity(batch);
            let mut buf = ip.get();
            loop {
                while !eof && decoded < next_a + seg {
                    match dec.next_chunk(32768)? {
                        Some((l, r)) => {
                            decoded += l.len();
                            if send_mix {
                                con_tx_prep.send(ConsistMsg::Mix(l.clone(), r.clone())).map_err(|_| anyhow!("sink gone"))?;
                            }
                            bl.extend_from_slice(&l);
                            br.extend_from_slice(&r);
                        }
                        None => eof = true,
                    }
                }
                if next_a >= decoded {
                    break;
                }
                let off = next_a - base;
                let have = (decoded - next_a).min(seg);
                cl[..have].copy_from_slice(&bl[off..off + have]);
                cr[..have].copy_from_slice(&br[off..off + have]);
                // past the file's end (only the last window): the audio mirrored, as MixWav::read_window.
                // `eof` is set whenever have < seg, so `decoded` is the track's length.
                for i in have..seg {
                    let k = mirror_index(next_a + i, decoded) - base;
                    cl[i] = bl[k];
                    cr[i] = br[k];
                }
                let slot = wins.len();
                let (mean, std) = dsp.mix_to_input(&cl, &cr, &mut buf[slot * in_len..(slot + 1) * in_len]);
                wins.push(Win { idx, a: next_a, mean, std });
                if wins.len() == batch {
                    let full = std::mem::replace(&mut buf, ip.get());
                    prep_tx.send(PrepMsg::Batch { wins: std::mem::take(&mut wins), input: full }).map_err(|_| anyhow!("network stage gone"))?;
                }
                idx += 1;
                next_a += plan.step;
                // keep the last window's worth of audio: the mirror at the end reads it
                let drop_to = next_a.min(decoded.saturating_sub(seg));
                if drop_to > base {
                    bl.drain(..drop_to - base);
                    br.drain(..drop_to - base);
                    base = drop_to;
                }
            }
            if !wins.is_empty() {
                // the last batch, short: zero windows fill it (their output is dropped)
                buf[wins.len() * in_len..].iter_mut().for_each(|v| *v = 0.0);
                prep_tx.send(PrepMsg::Batch { wins: std::mem::take(&mut wins), input: buf }).map_err(|_| anyhow!("network stage gone"))?;
            }
            dec.finish()?;
            if decoded == 0 {
                bail!("no audio decoded");
            }
            prep_tx.send(PrepMsg::End(decoded)).map_err(|_| anyhow!("network stage gone"))?;
            Ok(())
        });

        // finish: spectrum out + overlap-add
        let p3 = plan.clone();
        let op = out_pool.clone();
        let fin = sc.spawn(move || -> Result<()> {
            let plan = p3;
            let mut dsp = WindowDsp::new(geom.frames);
            let mut wave = vec![0.0f32; 8 * seg];
            let mut acc = vec![0.0f32; 8 * seg];
            let mut norm = vec![0.0f32; seg];
            let mut cur_a = 0usize;
            let mut have = false;
            let emit = |acc: &[f32], norm: &[f32], len: usize| -> Result<()> {
                let mut chunk = vec![0.0f32; 8 * len];
                for c in 0..8 {
                    for i in 0..len {
                        chunk[c * len + i] = acc[c * seg + i] / norm[i].max(1e-6);
                    }
                }
                con_tx.send(ConsistMsg::Est(chunk, len)).map_err(|_| anyhow!("sink gone"))
            };
            while let Ok(m) = done_rx.recv() {
                match m {
                    DoneMsg::Batch { wins, y, fmt } => {
                        for (slot, &Win { idx, a, mean, std }) in wins.iter().enumerate() {
                            match fmt {
                                Format::F32 => dsp.output_to_wave(&y[slot * out_len..(slot + 1) * out_len], mean, std, &mut wave),
                                Format::F16 => {
                                    let h = unsafe { std::slice::from_raw_parts((y.as_ptr() as *const u16).add(slot * out_len), out_len) };
                                    dsp.output_to_wave(h, mean, std, &mut wave)
                                }
                            }
                            if have {
                                let len = a - cur_a;
                                emit(&acc, &norm, len)?;
                                for c in 0..8 {
                                    acc.copy_within(c * seg + len..(c + 1) * seg, c * seg);
                                    acc[c * seg + seg - len..(c + 1) * seg].iter_mut().for_each(|v| *v = 0.0);
                                }
                                norm.copy_within(len.., 0);
                                norm[seg - len..].iter_mut().for_each(|v| *v = 0.0);
                            }
                            cur_a = a;
                            have = true;
                            let w = plan.weights(idx);
                            for c in 0..8 {
                                let (dst, src) = (&mut acc[c * seg..(c + 1) * seg], &wave[c * seg..(c + 1) * seg]);
                                for i in 0..seg {
                                    dst[i] += src[i] * w[i];
                                }
                            }
                            for i in 0..seg {
                                norm[i] += w[i];
                            }
                        }
                        op.put(y);
                    }
                    DoneMsg::End(n) => {
                        if have {
                            emit(&acc, &norm, n - cur_a)?;
                        }
                        con_tx.send(ConsistMsg::End(n)).map_err(|_| anyhow!("sink gone"))?;
                        return Ok(());
                    }
                }
            }
            Ok(())
        });

        // network, on this thread
        let mut net = Duration::ZERO;
        let mut windows = 0usize;
        let mut samples = 0usize;
        let mut res: Result<()> = Ok(());
        let tl = Instant::now();
        if opt.progress.is_some() {
            eprintln!("preparing");
        }
        let mut backend = match make_backend() {
            Ok(b) => Some(b),
            Err(e) => {
                res = Err(e);
                None
            }
        };
        let load = tl.elapsed();
        if let Some(b) = &backend {
            if b.geometry() != geom {
                res = Err(anyhow!("the model's window {:?} is not the planned {:?}", b.geometry(), geom));
                backend = None;
            }
        }
        while backend.is_some() {
            let Ok(m) = prep_rx.recv() else { break };
            match m {
                PrepMsg::Batch { wins, input } => {
                    let mut out = out_pool.get();
                    let t = Instant::now();
                    let fmt = match backend.as_mut().unwrap().separate_batch(&input, &mut out) {
                        Ok(f) => f,
                        Err(e) => {
                            res = Err(e);
                            break;
                        }
                    };
                    net += t.elapsed();
                    for _ in &wins {
                        windows += 1;
                        if let Some(total) = opt.progress {
                            // A short estimate never reads as more than done.
                            eprintln!("progress {} {}", windows, if total == 0 { 0 } else { total.max(windows) });
                        }
                    }
                    in_pool.put(input);
                    if done_tx.send(DoneMsg::Batch { wins, y: out, fmt }).is_err() {
                        break;
                    }
                }
                PrepMsg::End(n) => {
                    samples = n;
                    if opt.progress.is_some() {
                        eprintln!("progress {windows} {windows}");
                    }
                    let _ = done_tx.send(DoneMsg::End(n));
                    break;
                }
            }
        }
        drop(prep_rx);
        drop(done_tx);
        let r1 = prep.join().map_err(|_| anyhow!("prep thread panicked"))?;
        let r2 = fin.join().map_err(|_| anyhow!("finish thread panicked"))?;
        let r3 = sink.join().map_err(|_| anyhow!("sink thread panicked"))?;
        res?;
        r1?;
        r2?;
        r3?;
        stats.windows = windows;
        stats.network = net;
        stats.load = load;
        stats.samples = samples;
        Ok(())
    })?;
    stats.total = t0.elapsed();
    Ok(stats)
}
