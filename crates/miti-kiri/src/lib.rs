// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! miti-kiri: stem separation. Windows, the network (through `kiri_core::backend`), overlap-add, the
//! consistency stage and stem writing. What a host wants beside the stems it takes through the
//! `BlockSink` trait.
pub mod blocks;
pub mod cli;
pub mod consist;
pub mod pipeline;
pub mod probe;
pub mod wav;
