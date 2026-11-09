//! Time batched Ripple ONNX decisions.
//!
//! ```text
//! ripple-bench [model_dir]
//! ```
//!
//! Prints milliseconds per decision and per row for N = 1, 64, 256, 1024.
//! The target is under 0.05 ms/row at N >= 256. Set `RIPPLE_BENCH_CHECK=1`
//! to exit non-zero when a release run misses that target. `RIPPLE_ORT_THREADS`
//! sets the intra-op thread count (default 4).

use std::path::PathBuf;
use std::process::ExitCode;

use indococo_bot::ripple::{RipplePolicy, ripple_ort_threads};

fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("HOME")).join("eval_bots/ripple/bots/v1.1"));
    if !path.join("ripple-v1.1.onnx").is_file() {
        eprintln!("weights absent at {}", path.display());
        return ExitCode::SUCCESS;
    }
    let policy = RipplePolicy::load(&path).unwrap_or_else(|e| panic!("load ripple: {e}"));
    let check = std::env::var("RIPPLE_BENCH_CHECK").ok().as_deref() == Some("1");
    println!("RIPPLE_ORT_THREADS={}", ripple_ort_threads());
    let mut failed = false;
    for &rows in &[1usize, 64, 256, 1024] {
        let (ms, per_row) = policy.bench_infer(rows, 1, 3);
        println!("N={rows} ms_per_decision={ms:.4} ms_per_row={per_row:.5}");
        if check && rows >= 256 && per_row >= 0.05 {
            eprintln!("N={rows} {per_row:.5} ms/row is not under 0.05");
            failed = true;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
