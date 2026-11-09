//! Time batched Ripple decisions.
//!
//! ```text
//! ripple-bench [model_dir] [--cuda|--cpu]
//! ```
//!
//! CPU (ONNX) prints ms/row for N = 1, 64, 256, 1024. Target: under 0.05 ms/row
//! at N >= 256. CUDA (LibTorch, TF32 off) prints N = 64, 256, 1024, 4096,
//! split into forward and CPU post, plus the extra weight bytes. Target:
//! under 0.01 ms/row total at N >= 1024.
//! `RIPPLE_BENCH_CHECK=1` exits non-zero when a run misses its target.
//! `RIPPLE_DEVICE=cpu|cuda` selects the backend (default cuda when the
//! safetensors export and a GPU are both available).

use std::path::PathBuf;
use std::process::ExitCode;

use indococo_bot::ripple::{ripple_ort_threads, RipplePolicy};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut path = PathBuf::from(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
    for arg in args.by_ref() {
        match arg.as_str() {
            // Before any other thread exists. Nightly marks set_var unsafe.
            "--cuda" => unsafe { std::env::set_var("RIPPLE_DEVICE", "cuda") },
            "--cpu" => unsafe { std::env::set_var("RIPPLE_DEVICE", "cpu") },
            other if !other.starts_with('-') => path = PathBuf::from(other),
            other => panic!("unknown argument {other}"),
        }
    }
    if !path.join("ripple-v1.1.onnx").is_file() {
        eprintln!("weights absent at {}", path.display());
        return ExitCode::SUCCESS;
    }
    let policy = RipplePolicy::load(&path).unwrap_or_else(|e| panic!("load ripple: {e}"));
    let cuda = policy.uses_cuda();
    let check = std::env::var("RIPPLE_BENCH_CHECK").ok().as_deref() == Some("1");
    println!("device={}", if cuda { "cuda" } else { "cpu" });
    println!("RIPPLE_ORT_THREADS={}", ripple_ort_threads());
    if let Some((weights, allocator)) = policy.gpu_memory() {
        println!("vram_weights_mb={:.2}", weights as f64 / 1e6);
        if let Some((allocated, reserved)) = allocator {
            println!(
                "vram_allocated_mb={:.2} vram_reserved_mb={:.2}",
                allocated as f64 / 1e6,
                reserved as f64 / 1e6
            );
        }
    }
    let sizes: &[usize] = if cuda {
        &[64, 256, 1024, 4096]
    } else {
        &[1, 64, 256, 1024]
    };
    let (min_n, limit) = if cuda { (1024, 0.01) } else { (256, 0.05) };
    let mut failed = false;
    for &rows in sizes {
        let times = policy.bench_profile(rows, 1, 3);
        println!(
            "N={rows} ms_per_row={:.5} forward_ms_per_row={:.5} post_ms_per_row={:.5}",
            times.total_ms_per_row, times.forward_ms_per_row, times.post_ms_per_row
        );
        if check && rows >= min_n && times.total_ms_per_row >= limit {
            eprintln!("N={rows} {:.5} ms/row is not under {limit}", times.total_ms_per_row);
            failed = true;
        }
    }
    if let Some((weights, allocator)) = policy.gpu_memory() {
        println!("vram_weights_after_mb={:.2}", weights as f64 / 1e6);
        if let Some((allocated, reserved)) = allocator {
            println!(
                "vram_allocated_after_mb={:.2} vram_reserved_after_mb={:.2}",
                allocated as f64 / 1e6,
                reserved as f64 / 1e6
            );
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
