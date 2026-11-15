// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
use clap::Parser;
use miti_kiri::cli::{run, Args};

/// Separate one audio file into drums, bass, other, vocals on the GPU (Core ML)
#[derive(Parser)]
#[command(name = "miti-kiri", version, author, after_help = "miti-kiri 1.0 by Supriya Wickrematillake. https://kiri.terrific.life/")]
struct Cli {
    #[command(flatten)]
    args: Args,
}

fn main() -> anyhow::Result<()> {
    run(&Cli::parse().args, None).map(|_| ())
}
