//! indococo 1v1 bot on the indococo-ppo framework.
//!
//! * [`obs`] / [`action`]: the exact input and action space of `1v1.pt`.
//! * [`state`]: a Python-shaped per-step view used by every reward.
//! * [`rewards`]: the E8B2 reward set, one module per Python class.
//! * [`env`]: the training environment.

pub mod action;
pub mod consts;
pub mod env;
pub mod math;
pub mod obs;
pub mod rewards;
pub mod state;

/// Hidden layers of the policy and critic (`1v1.pt` DiscreteFF).
pub const LAYERS: [usize; 4] = [2048, 2048, 1024, 1024];
