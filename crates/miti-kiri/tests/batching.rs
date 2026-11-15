// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The batch is only how windows reach the network: with the pass-through backend, a separation at batch 2 is
//! bit-identical to batch 1 at the same window, for both window lengths and an odd and an even window count
//! (the last batch padded). Needs ffmpeg on PATH (the decoder).
use kiri_core::backend::{Backend, Passthrough};
use miti_kiri::consist::Mode;
use miti_kiri::pipeline::{separate, Options};
use kiri_core::spec::Geometry;
use miti_kiri::wav::WavWriter;

fn run(dir: &std::path::Path, input: &std::path::Path, g: Geometry, consist: Mode) -> Vec<Vec<u8>> {
    let out = dir.join(format!("out-{}-{}-{:?}", g.frames, g.batch, consist));
    let opt = Options { geometry: g, overlap: 0.25, consist, progress: None, blocks: None };
    let st = separate(input, &out, &opt, move || Ok(Box::new(Passthrough(g)) as Box<dyn Backend>)).unwrap();
    assert!(st.windows > 0);
    ["drums", "bass", "other", "vocals"].iter().map(|s| std::fs::read(out.join(format!("{s}.wav"))).unwrap()).collect()
}

#[test]
fn batch_two_equals_batch_one() {
    let dir = std::env::temp_dir().join(format!("kiri-batching-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // 27 s and 31 s of a deterministic signal: an odd and an even number of windows at 25 % overlap
    for secs in [27usize, 31] {
        let n = secs * 44100;
        let l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.3 + ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.1).collect();
        let r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.021).cos() * 0.2).collect();
        let input = dir.join(format!("in-{secs}.wav"));
        let mut w = WavWriter::create(&input).unwrap();
        w.write(&l, &r).unwrap();
        w.finish().unwrap();
        for frames in [128, 96] {
            for consist in [Mode::None, Mode::Tf] {
                let one = run(&dir, &input, Geometry { frames, batch: 1 }, consist);
                let two = run(&dir, &input, Geometry { frames, batch: 2 }, consist);
                assert!(one == two, "{secs} s, {frames} frames, {consist:?}: batch 2 differs from batch 1");
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
