//! Ripple v1.1 policy and base nets on LibTorch.
//!
//! Same module as Ripple-RLBot `tools/export_network.py` (`ExportNetwork`) at
//! 7ca4273a, which is what `ripple-v1.1.onnx` was exported from. Weights come
//! from `tools/ripple_export_torch.py` (safetensors, PyTorch `[out, in]` layout).
//!
//! Op order for one row batch, fp32, TF32 off:
//!
//! 1. Mask: `others[:, :, 15] == 0`, then three false columns (ball, actions, boosts).
//! 2. Query: MLP on `main[:, :30]` (30 → 128, two hidden 128s, erf-GELU).
//! 3. KV, 8 tokens: other-player MLP (21 → 128) on 5 slots, ball MLP on the
//!    flattened `[6, 9]`, action-queue MLP on the flattened `[6, 8]`, boost MLP
//!    (5 → 32 hidden → 128) reduced with `amax` over the 34 pads.
//! 4. Four times: LayerNorm Q and KV (eps 1e-5), packed QKV linear (384),
//!    reshape to 4 heads of 32, scores `Q K^T / sqrt(32)`, `-inf` on the mask,
//!    softmax, times V, out projection, residual, LayerNorm, MLP 128 → 512 → 128,
//!    residual.
//! 5. `LayerNorm(query) @ action_embeddings.T` → 276 logits.
//! 6. Where `main[:, 30] > 0.5`, gather logits through `invert_map`.
//!
//! KL, kickoff CDF, and the action queue stay on CPU in `ripple.rs`.
//!
//! TF32 is process-wide. The guard turns it off only while these kernels are
//! launched, then puts the previous value back, so the PPO update (which runs
//! after collection) keeps its own precision. A pool that launches its learner
//! forward in the same window can see TF32 off for that launch. Each calling
//! thread uses its own CUDA stream, so two pools do not share the default stream
//! and there is no lock around the forward.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::Mutex;

use tch::{Device, Kind, Tensor};

const DIM: i64 = 128;
const HEADS: i64 = 4;
const HEAD_DIM: i64 = 32;
const LAYERS: usize = 4;
const KV_LEN: i64 = 8;
const N_ACTIONS: i64 = 276;
/// Past this many rows, the forward is split so the boost-pad activation
/// (`[N, 34, 128]`) stays a few hundred MB.
const GPU_CHUNK: usize = 16384;
/// `sqrt(32)` as f32 (bits `0x40b504f3`). The ONNX graph divides by this value.
const ATTN_SCALE: f64 = f32::from_bits(0x40b5_04f3) as f64;

unsafe extern "C" {
    fn ripple_allow_tf32_cublas() -> i32;
    fn ripple_set_allow_tf32_cublas(enabled: i32);
    fn ripple_allow_tf32_cudnn() -> i32;
    fn ripple_set_allow_tf32_cudnn(enabled: i32);
    fn ripple_stream_create(device: i32) -> *mut c_void;
    fn ripple_stream_set(stream: *mut c_void);
    fn ripple_stream_synchronize(stream: *mut c_void);
    fn ripple_copy_nonblocking(dst: *mut c_void, src: *mut c_void);
    fn ripple_cuda_allocated(device: i32) -> usize;
    fn ripple_cuda_reserved(device: i32) -> usize;
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

/// Turns TF32 off for the launches inside the guard. The mutex is not held
/// during the forward, so pools do not queue behind each other's GPU work.
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

fn with_stream<R>(device: Device, f: impl FnOnce(*mut c_void) -> R) -> R {
    let Device::Cuda(index) = device else {
        return f(std::ptr::null_mut());
    };
    std::thread_local! {
        static STREAM: std::cell::RefCell<Option<(usize, *mut c_void)>> = const { std::cell::RefCell::new(None) };
    }
    STREAM.with(|slot| {
        let ptr = {
            let mut slot = slot.borrow_mut();
            if slot.map(|(i, _)| i) != Some(index) {
                let created = unsafe { ripple_stream_create(index as i32) };
                *slot = Some((index, created));
            }
            slot.as_ref().unwrap().1
        };
        if !ptr.is_null() {
            unsafe { ripple_stream_set(ptr) };
        }
        f(ptr)
    })
}

/// Pinned host tensors reused by this thread. Two pools do not share them.
struct HostStage {
    cap: i64,
    device: Device,
    main: Tensor,
    others: Tensor,
    ball: Tensor,
    actions: Tensor,
    boosts: Tensor,
    policy: Tensor,
    base: Tensor,
}

impl HostStage {
    fn alloc(rows: i64, device: Device) -> Self {
        let pin = |shape: &[i64]| Tensor::empty(shape, (Kind::Float, Device::Cpu)).pin_memory(device);
        Self {
            cap: rows,
            device,
            main: pin(&[rows, 32]),
            others: pin(&[rows, 5, 21]),
            ball: pin(&[rows, 6, 9]),
            actions: pin(&[rows, 6, 8]),
            boosts: pin(&[rows, 34, 5]),
            policy: pin(&[rows, N_ACTIONS]),
            base: pin(&[rows, N_ACTIONS]),
        }
    }
}

fn with_stage<R>(rows: usize, device: Device, f: impl FnOnce(&mut HostStage) -> R) -> R {
    std::thread_local! {
        static STAGE: std::cell::RefCell<Option<HostStage>> = const { std::cell::RefCell::new(None) };
    }
    STAGE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let rows = rows as i64;
        let stale = slot
            .as_ref()
            .map(|s| s.cap < rows || s.device != device)
            .unwrap_or(true);
        if stale {
            *slot = Some(HostStage::alloc(rows, device));
        }
        f(slot.as_mut().unwrap())
    })
}

fn upload(pinned: &Tensor, data: &[f32], shape: &[i64], device: Device) -> Tensor {
    let mut view = pinned.narrow(0, 0, shape[0]);
    view.copy_(&Tensor::from_slice(data).reshape(shape));
    view.to_device_(device, Kind::Float, true, true)
}

fn queue_download(pinned: &Tensor, gpu: &Tensor, rows: i64) -> Tensor {
    let view = pinned.narrow(0, 0, rows);
    unsafe {
        ripple_copy_nonblocking(view.as_ptr() as *mut c_void, gpu.as_ptr() as *mut c_void);
    }
    view
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn load(weights: &mut HashMap<String, Tensor>, prefix: &str, device: Device) -> Self {
        Self {
            weight: take(weights, &format!("{prefix}.weight"), device),
            bias: take(weights, &format!("{prefix}.bias"), device),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.linear(&self.weight, Some(&self.bias))
    }
}

struct Mlp {
    layers: [Linear; 4],
}

impl Mlp {
    fn load(weights: &mut HashMap<String, Tensor>, prefix: &str, device: Device) -> Self {
        Self {
            layers: [
                Linear::load(weights, &format!("{prefix}.0"), device),
                Linear::load(weights, &format!("{prefix}.2"), device),
                Linear::load(weights, &format!("{prefix}.4"), device),
                Linear::load(weights, &format!("{prefix}.6"), device),
            ],
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut y = self.layers[0].forward(x);
        for layer in &self.layers[1..] {
            y = y.gelu("none");
            y = layer.forward(&y);
        }
        y
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn load(weights: &mut HashMap<String, Tensor>, prefix: &str, device: Device) -> Self {
        Self {
            weight: take(weights, &format!("{prefix}.weight"), device),
            bias: take(weights, &format!("{prefix}.bias"), device),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let n = self.weight.size()[0];
        x.layer_norm([n], Some(&self.weight), Some(&self.bias), 1e-5, false)
    }
}

struct Block {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_weight: Tensor,
    out_bias: Tensor,
    norm_q: Norm,
    norm_kv: Norm,
    norm_qkv: Norm,
    mlp0: Linear,
    mlp2: Linear,
}

impl Block {
    fn load(weights: &mut HashMap<String, Tensor>, index: usize, device: Device) -> Self {
        let p = format!("layers.{index}");
        Self {
            in_proj_weight: take(weights, &format!("{p}.attn.in_proj_weight"), device),
            in_proj_bias: take(weights, &format!("{p}.attn.in_proj_bias"), device),
            out_weight: take(weights, &format!("{p}.attn.out_proj.weight"), device),
            out_bias: take(weights, &format!("{p}.attn.out_proj.bias"), device),
            norm_q: Norm::load(weights, &format!("{p}.norm_q"), device),
            norm_kv: Norm::load(weights, &format!("{p}.norm_kv"), device),
            norm_qkv: Norm::load(weights, &format!("{p}.norm_qkv"), device),
            mlp0: Linear::load(weights, &format!("{p}.mlp.0"), device),
            mlp2: Linear::load(weights, &format!("{p}.mlp.2"), device),
        }
    }

    fn forward(&self, query: &Tensor, kv: &Tensor, mask: &Tensor) -> Tensor {
        let qn = self.norm_q.forward(query);
        let kvn = self.norm_kv.forward(kv);
        let q_w = self.in_proj_weight.narrow(0, 0, DIM);
        let q_b = self.in_proj_bias.narrow(0, 0, DIM);
        let kv_w = self.in_proj_weight.narrow(0, DIM, DIM * 2);
        let kv_b = self.in_proj_bias.narrow(0, DIM, DIM * 2);
        let q_lin = qn.linear(&q_w, Some(&q_b));
        let kv_lin = kvn.linear(&kv_w, Some(&kv_b));
        let parts = kv_lin.chunk(2, -1);
        let batch = q_lin.size()[0];
        let q = q_lin.reshape(&[batch, 1, HEADS, HEAD_DIM]).transpose(1, 2);
        let k = parts[0]
            .reshape(&[batch, KV_LEN, HEADS, HEAD_DIM])
            .transpose(1, 2);
        let v = parts[1]
            .reshape(&[batch, KV_LEN, HEADS, HEAD_DIM])
            .transpose(1, 2);
        let scores = (&q / ATTN_SCALE).matmul(&k.transpose(-2, -1));
        let scores = scores.masked_fill(&mask, f64::NEG_INFINITY);
        let attn = scores
            .softmax(-1, Kind::Float)
            .matmul(&v)
            .transpose(1, 2)
            .reshape(&[batch, 1, DIM]);
        let mixed = attn.linear(&self.out_weight, Some(&self.out_bias));
        let result = query + mixed;
        let hidden = self.mlp2.forward(
            &self
                .mlp0
                .forward(&self.norm_qkv.forward(&result))
                .gelu("none"),
        );
        result + hidden
    }
}

struct RippleNet {
    main: Mlp,
    others: Mlp,
    ball: Mlp,
    actions: Mlp,
    boosts: Mlp,
    blocks: Vec<Block>,
    state_norm: Norm,
    action_embeddings: Tensor,
    invert_map: Tensor,
}

impl RippleNet {
    fn load(path: &Path, device: Device) -> Result<Self, String> {
        let mut weights: HashMap<String, Tensor> = Tensor::read_safetensors(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .into_iter()
            .collect();
        let net = Self {
            main: Mlp::load(&mut weights, "main_player_processor", device),
            others: Mlp::load(&mut weights, "other_players_processor", device),
            ball: Mlp::load(&mut weights, "ball_processor", device),
            actions: Mlp::load(&mut weights, "action_queue_processor", device),
            boosts: Mlp::load(&mut weights, "boost_processor", device),
            blocks: (0..LAYERS)
                .map(|i| Block::load(&mut weights, i, device))
                .collect(),
            state_norm: Norm::load(&mut weights, "state_norm", device),
            action_embeddings: take(&mut weights, "action_embeddings", device),
            invert_map: take(&mut weights, "invert_map", device),
        };
        if net.blocks.len() != LAYERS {
            return Err(format!("expected {LAYERS} attention layers"));
        }
        Ok(net)
    }

    fn weight_bytes(&self) -> u64 {
        let mut n = 0u64;
        let count = |t: &Tensor| t.numel() as u64 * t.kind().elt_size_in_bytes() as u64;
        for layer in &self.main.layers {
            n += count(&layer.weight) + count(&layer.bias);
        }
        for mlp in [&self.others, &self.ball, &self.actions, &self.boosts] {
            for layer in &mlp.layers {
                n += count(&layer.weight) + count(&layer.bias);
            }
        }
        for block in &self.blocks {
            n += count(&block.in_proj_weight) + count(&block.in_proj_bias);
            n += count(&block.out_weight) + count(&block.out_bias);
            for norm in [&block.norm_q, &block.norm_kv, &block.norm_qkv] {
                n += count(&norm.weight) + count(&norm.bias);
            }
            n += count(&block.mlp0.weight) + count(&block.mlp0.bias);
            n += count(&block.mlp2.weight) + count(&block.mlp2.bias);
        }
        n += count(&self.state_norm.weight) + count(&self.state_norm.bias);
        n += count(&self.action_embeddings) + count(&self.invert_map);
        n
    }

    fn forward(
        &self,
        main: &Tensor,
        others: &Tensor,
        ball: &Tensor,
        actions: &Tensor,
        boosts: &Tensor,
    ) -> Tensor {
        let batch = main.size()[0];
        let pad = others.narrow(2, 15, 1).squeeze_dim(2).eq(0.0);
        let keep = Tensor::zeros(&[batch, 3], (Kind::Bool, main.device()));
        let mask = Tensor::concat(&[pad, keep], 1).unsqueeze(1).unsqueeze(2);
        let query = self.main.forward(&main.narrow(1, 0, 30)).unsqueeze(1);
        let ball = self.ball.forward(&ball.reshape(&[batch, 1, 54]));
        let actions = self.actions.forward(&actions.reshape(&[batch, 1, 48]));
        let boosts = self.boosts.forward(boosts).amax(&[1i64][..], true);
        let kv = Tensor::concat(&[self.others.forward(others), ball, actions, boosts], 1);
        let mut query = query;
        for block in &self.blocks {
            query = block.forward(&query, &kv, &mask);
        }
        let state = self.state_norm.forward(&query.squeeze_dim(1));
        let logits = state.matmul(&self.action_embeddings.transpose(0, 1));
        let side = main.narrow(1, 30, 1).gt(0.5);
        let inverted = logits.index_select(1, &self.invert_map);
        inverted.where_self(&side, &logits)
    }
}

fn take(weights: &mut HashMap<String, Tensor>, name: &str, device: Device) -> Tensor {
    weights
        .remove(name)
        .unwrap_or_else(|| panic!("ripple safetensors missing {name}"))
        .to_device(device)
        .to_kind(if name == "invert_map" {
            Kind::Int64
        } else {
            Kind::Float
        })
}

/// Policy and base nets on one CUDA device. Read-only after load.
pub struct RippleGpu {
    policy: RippleNet,
    base: RippleNet,
    device: Device,
    pub weight_bytes: u64,
}

unsafe impl Send for RippleGpu {}
unsafe impl Sync for RippleGpu {}

impl RippleGpu {
    pub fn load(dir: &Path, device: Device) -> Result<Self, String> {
        let policy = RippleNet::load(&dir.join("ripple-v1.1.safetensors"), device)?;
        let base = RippleNet::load(&dir.join("ripple-v1.1-bcm.safetensors"), device)?;
        let weight_bytes = policy.weight_bytes() + base.weight_bytes();
        Ok(Self {
            policy,
            base,
            device,
            weight_bytes,
        })
    }

    pub fn logits(&self, obs: &[f32], rows: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        if rows == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let width = 32 + 5 * 21 + 6 * 9 + 6 * 8 + 34 * 5;
        if obs.len() < rows * width {
            return Err(format!("obs len {} < {rows} * {width}", obs.len()));
        }
        let mut policy = Vec::with_capacity(rows * N_ACTIONS as usize);
        let mut base = Vec::with_capacity(rows * N_ACTIONS as usize);
        let mut done = 0;
        while done < rows {
            let n = (rows - done).min(GPU_CHUNK);
            let slice = &obs[done * width..(done + n) * width];
            let (p, b) = self.forward_chunk(slice, n)?;
            policy.extend(p);
            base.extend(b);
            done += n;
        }
        Ok((policy, base))
    }

    fn forward_chunk(&self, obs: &[f32], rows: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let width = obs.len() / rows;
        let packed = super::pack_obs(obs, rows, width);
        let device = self.device;
        let _tf32 = Tf32Guard::enter();
        with_stage(rows, device, |stage| {
            with_stream(device, |stream| {
                tch::no_grad(|| {
                    let rows_i = rows as i64;
                    let main = upload(&stage.main, &packed[0], &[rows_i, 32], device);
                    let others = upload(&stage.others, &packed[1], &[rows_i, 5, 21], device);
                    let ball = upload(&stage.ball, &packed[2], &[rows_i, 6, 9], device);
                    let actions = upload(&stage.actions, &packed[3], &[rows_i, 6, 8], device);
                    let boosts = upload(&stage.boosts, &packed[4], &[rows_i, 34, 5], device);
                    let policy_gpu = self.policy.forward(&main, &others, &ball, &actions, &boosts);
                    // One stream, so this D2H finishes before the base forward starts.
                    // Pinned memory skips the pageable bounce buffer, and the CPU
                    // waits once at the end instead of on every copy.
                    let policy_host = queue_download(&stage.policy, &policy_gpu, rows_i);
                    let base_gpu = self.base.forward(&main, &others, &ball, &actions, &boosts);
                    let base_host = queue_download(&stage.base, &base_gpu, rows_i);
                    if stream.is_null() {
                        if let Device::Cuda(index) = device {
                            tch::Cuda::synchronize(index as i64);
                        }
                    } else {
                        unsafe { ripple_stream_synchronize(stream) };
                    }
                    drop(policy_gpu);
                    drop(base_gpu);
                    let policy = copy_logits(&policy_host, rows)?;
                    let base = copy_logits(&base_host, rows)?;
                    Ok((policy, base))
                })
            })
        })
    }

    pub fn allocator_bytes(&self) -> Option<(u64, u64)> {
        let Device::Cuda(index) = self.device else {
            return None;
        };
        let allocated = unsafe { ripple_cuda_allocated(index as i32) } as u64;
        let reserved = unsafe { ripple_cuda_reserved(index as i32) } as u64;
        Some((allocated, reserved))
    }
}

fn copy_logits(logits: &Tensor, rows: usize) -> Result<Vec<f32>, String> {
    let owned;
    let cpu = if logits.device() == Device::Cpu {
        logits
    } else {
        owned = logits.to(Device::Cpu).contiguous();
        &owned
    };
    if cpu.size() != [rows as i64, N_ACTIONS] {
        return Err(format!(
            "logits shape {:?}, expected [{rows}, {N_ACTIONS}]",
            cpu.size()
        ));
    }
    let n = rows * N_ACTIONS as usize;
    let mut out = vec![0f32; n];
    cpu.f_copy_data(&mut out, n).map_err(|e| e.to_string())?;
    Ok(out)
}

pub fn cuda_device() -> Option<Device> {
    if tch::Cuda::is_available() {
        Some(Device::Cuda(0))
    } else {
        None
    }
}
