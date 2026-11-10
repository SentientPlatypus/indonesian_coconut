//! Side CUDA streams for Galactus and Megabyte.
//!
//! The PPO update runs on the thread that calls `Learner::learn` and uses
//! that thread's current stream (the default stream, unless something else
//! moved it). Collection pools run on other threads. Each of those threads
//! gets its own stream so its policy kernels are not queued behind the
//! update. Galactus and Megabyte then take a second stream for their own
//! forward, so that forward can overlap the policy kernels already queued
//! on the collector stream.
//!
//! TF32 is process-wide. The guard turns it off only while the league
//! forward is launched, then puts the previous value back. A kernel the
//! update launches in that window sees TF32 off. That is the same window
//! Ripple already opens around its own forward.

use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::Mutex;

unsafe extern "C" {
    fn ripple_allow_tf32_cublas() -> i32;
    fn ripple_set_allow_tf32_cublas(enabled: i32);
    fn ripple_allow_tf32_cudnn() -> i32;
    fn ripple_set_allow_tf32_cudnn(enabled: i32);
    fn ripple_stream_create(device: i32) -> *mut c_void;
    fn ripple_stream_set(stream: *mut c_void);
    fn ripple_stream_restore_default(device: i32);
}

struct Tf32State {
    depth: i32,
    saved_cublas: i32,
    saved_cudnn: i32,
}

static TF32: Mutex<Tf32State> = Mutex::new(Tf32State {
    depth: 0,
    saved_cublas: 1,
    saved_cudnn: 1,
});

struct Tf32Guard;

impl Tf32Guard {
    fn enter() -> Self {
        let mut state = TF32.lock().unwrap();
        if state.depth == 0 {
            state.saved_cublas = unsafe { ripple_allow_tf32_cublas() };
            state.saved_cudnn = unsafe { ripple_allow_tf32_cudnn() };
            unsafe {
                ripple_set_allow_tf32_cublas(0);
                ripple_set_allow_tf32_cudnn(0);
            }
        }
        state.depth += 1;
        Self
    }
}

impl Drop for Tf32Guard {
    fn drop(&mut self) {
        let mut state = TF32.lock().unwrap();
        state.depth -= 1;
        if state.depth == 0 {
            unsafe {
                ripple_set_allow_tf32_cublas(state.saved_cublas);
                ripple_set_allow_tf32_cudnn(state.saved_cudnn);
            }
        }
    }
}

fn stream_of(slot: &RefCell<Option<*mut c_void>>) -> *mut c_void {
    let mut slot = slot.borrow_mut();
    if slot.is_none() {
        *slot = Some(unsafe { ripple_stream_create(0) });
    }
    slot.unwrap_or(std::ptr::null_mut())
}

fn set_stream(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe { ripple_stream_set(ptr) };
    }
}

/// Current-thread collector stream. The learner registers this as the
/// collection-thread hook. A null stream (no CUDA) leaves the thread alone.
pub fn install_collect_stream() {
    std::thread_local! {
        static STREAM: RefCell<Option<*mut c_void>> = const { RefCell::new(None) };
    }
    STREAM.with(|slot| set_stream(stream_of(slot)));
}

/// League forward on its own stream, with TF32 off, then the previous stream
/// restored. On a collection thread that is the collector stream. On the
/// learner thread that is the default stream.
pub fn with_league_forward<R>(f: impl FnOnce() -> R) -> R {
    std::thread_local! {
        static STREAM: RefCell<Option<*mut c_void>> = const { RefCell::new(None) };
    }
    let _tf32 = Tf32Guard::enter();
    STREAM.with(|slot| set_stream(stream_of(slot)));
    let out = f();
    unsafe { ripple_stream_restore_default(0) };
    rlgymppo::reapply_collect_stream();
    out
}
