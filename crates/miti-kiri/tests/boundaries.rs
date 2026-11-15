// Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
//! The crates depend one way: kiri-core on nothing of ours, miti-kiri on kiri-core alone. Read from
//! `cargo metadata`, so a reverse edge fails the build.
use serde_json::Value;
use std::process::Command;

#[test]
fn dependency_direction_holds() {
    let out = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args(["metadata", "--format-version", "1", "--no-deps", "--offline"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    let ours: Vec<&str> = v["packages"].as_array().unwrap().iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert_eq!(ours, ["kiri-core", "miti-kiri"], "the workspace holds these two crates");
    for p in v["packages"].as_array().unwrap() {
        let name = p["name"].as_str().unwrap();
        let allowed: &[&str] = if name == "kiri-core" { &[] } else { &["kiri-core"] };
        for d in p["dependencies"].as_array().unwrap() {
            let dep = d["name"].as_str().unwrap();
            if ours.contains(&dep) {
                assert!(allowed.contains(&dep), "{name} -> {dep}: reverse edge");
            }
        }
    }
}
