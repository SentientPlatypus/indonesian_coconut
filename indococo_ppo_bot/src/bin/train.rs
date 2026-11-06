#![recursion_limit = "256"]
//! indococo 1v1 trainer.
//!
//! ```text
//! indococo-train train  <train.toml>                  # train (resumes from checkpoints)
//! indococo-train import <1v1.pt> <checkpoints_dir>    # warm start from the Python policy
//! indococo-train export <checkpoint_dir> <out.safetensors>
//! indococo-train record <reward.toml> <out.jsonl> <steps> [checkpoint_dir]
//! indococo-train rates  <setters.toml> <candidate_ckpt> <opponent_ckpt|self> <steps> [threads]
//! indococo-train curr <level> <episodes> <policy> [threads] [data_prefix] [noise]   # FR5 BC tools
//! indococo-train snapshots <setters.toml> <ckpt> <steps> <threads> <out.f32>  # FR7 replay states
//! indococo-train normal-obs <ckpt> <setters.toml> <steps> <threads> <out.f32>
//! indococo-train galactus-h2h <ckpt> <galactus_dir> <games> [threads] [dump.jsonl]  # Rust Galactus parity
//! indococo-train megabyte-obs <megabyte_dir> <states.jsonl>    # Dump Megabyte obs for validation
//! indococo-train megabyte-h2h <ckpt> <megabyte_dir> <games> [threads]  # Rust Megabyte h2h
//! indococo-train megabyte-logits <megabyte_dir> <obs.jsonl>    # Rust Megabyte logits for parity check
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use burn::prelude::*;
use burn::tensor::Device;
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, PytorchStore, SafetensorsStore};
use indococo_bot::LAYERS;
use indococo_bot::action::N_ACTIONS;
use indococo_bot::env::{BotEnv, create_env};
use indococo_bot::events::StepEvents;
use indococo_bot::obs::{base_obs_size, obs_size};
use indococo_bot::resets;
use indococo_bot::rewards::RewardConfig;
use indococo_bot::rewards::StepReward;
use indococo_bot::rewards::flip_reset_use::{FlipResetUse, FlipResetUseParams};
use indococo_bot::setters::{WeightedSetter, default_mix};
use indococo_bot::state::StateView;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::init_from_default;
use rlgymppo::{
    Actic, ComputePrecision, DecayShape, ExternalOpponent, FixedOpponent, LearnerConfig, LrSchedule, Net,
    PpoLearnerConfig, SelfPlayConfig, SkillTrackerConfig, TrainPrecision, default_adamw_optimizer,
};
use serde::{Deserialize, Serialize};

#[path = "../bc_cli.rs"]
mod bc_cli;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainConfig {
    reward_config: PathBuf,
    checkpoints: PathBuf,
    backend: String,
    threads: usize,
    pools: usize,
    games: usize,
    timesteps_per_iteration: usize,
    mini_batch_size: usize,
    epochs: usize,
    policy_lr: f32,
    critic_lr: f32,
    entropy: f32,
    gamma: f32,
    lambda: f32,
    policy_bf16: bool,
    critic_bf16: bool,
    save_every: u64,
    keep_checkpoints: usize,
    iterations: u64,
    self_play_old_versions: bool,
    skill_tracker: bool,
    /// Critic-only warm-up: until the run has this many cumulative steps, the
    /// policy learning rate is 0, so AdamW leaves the actor unchanged and only
    /// the critic learns. Use after `import`, whose critic is untrained.
    #[serde(default)]
    critic_warmup_steps: u64,
    /// Learning-rate schedule for policy and critic (default: constant).
    #[serde(default)]
    lr_schedule: LrScheduleConfig,
    /// Frozen opponents mixed into self-play.
    #[serde(default)]
    opponents: OpponentsConfig,
    /// Weighted reset mix, see `indococo_bot::setters`.
    #[serde(default = "default_mix")]
    state_setters: Vec<WeightedSetter>,
}

/// `[lr_schedule]`: constant, or flat until `hold_steps` cumulative steps and
/// then a linear or cosine decay to `min_scale` x the base rates over
/// `decay_steps`. Steps include the critic warm-up.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum LrScheduleConfig {
    #[default]
    Constant,
    Linear {
        hold_steps: u64,
        decay_steps: u64,
        min_scale: f32,
    },
    Cosine {
        hold_steps: u64,
        decay_steps: u64,
        min_scale: f32,
    },
}

impl LrScheduleConfig {
    /// The framework schedule counts PPO updates: one per iteration.
    fn schedule(self, timesteps_per_iteration: u64) -> LrSchedule {
        let (hold_steps, decay_steps, min_scale, shape) = match self {
            Self::Constant => return LrSchedule::Constant,
            Self::Linear {
                hold_steps,
                decay_steps,
                min_scale,
            } => (hold_steps, decay_steps, min_scale, DecayShape::Linear),
            Self::Cosine {
                hold_steps,
                decay_steps,
                min_scale,
            } => (hold_steps, decay_steps, min_scale, DecayShape::Cosine),
        };
        LrSchedule::HoldThenDecay {
            hold_iterations: hold_steps / timesteps_per_iteration,
            decay_iterations: decay_steps.div_ceil(timesteps_per_iteration),
            min_scale,
            shape,
        }
    }
}

/// `[opponents]`: with probability `chance` an iteration is played against
/// a frozen opponent instead of the current policy (on a random side). The
/// opponent is drawn by weight from the saved past versions (as one group,
/// only with `self_play_old_versions`) and the `[[opponents.fixed]]`
/// checkpoints, e.g. the imported `1v1.pt` and kept-best checkpoints.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OpponentsConfig {
    chance: f32,
    past_versions_weight: f32,
    fixed: Vec<FixedOpponentConfig>,
    /// `[[opponents.external]]`: frozen foreign bots ported to Rust
    /// (`kind = "galactus"`, `path` = dir with galactus_weights.{json,bin}).
    external: Vec<ExternalOpponentConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalOpponentConfig {
    name: String,
    kind: String,
    path: PathBuf,
    weight: f32,
}

/// Loads the external opponents (inference on `dev`).
fn load_external_opponents(cfg: &OpponentsConfig, dev: &Device) -> Vec<ExternalOpponent> {
    cfg.external
        .iter()
        .map(|o| {
            let policy: std::sync::Arc<dyn rlgymppo::ExternalPolicy> = match o.kind.as_str() {
                "galactus" => std::sync::Arc::new(
                    indococo_bot::galactus::GalactusPolicy::load(&o.path, dev)
                        .unwrap_or_else(|e| panic!("opponent {}: {e}", o.name)),
                ),
                "megabyte" => {
                    // Megabyte requires tch backend; map from Burn device
                    let tch_dev = tch::Device::Cuda(0);  // TODO: extract index from dev when it's libtorch_cuda
                    std::sync::Arc::new(
                        indococo_bot::megabyte::MegabytePolicy::load(&o.path, tch_dev)
                            .unwrap_or_else(|e| panic!("opponent {}: {e}", o.name)),
                    )
                }
                other => panic!("opponent {}: unknown external kind {other:?}", o.name),
            };
            ExternalOpponent { name: o.name.clone(), policy, weight: o.weight }
        })
        .collect()
}

impl Default for OpponentsConfig {
    fn default() -> Self {
        Self {
            chance: 0.15,
            past_versions_weight: 1.0,
            fixed: Vec::new(),
            external: Vec::new(),
        }
    }
}

/// A frozen checkpoint directory (`actor.bpk`) in the training pool.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedOpponentConfig {
    name: String,
    checkpoint: PathBuf,
    weight: f32,
}

/// Loads the fixed opponents' actors onto `dev` (the critic is unused).
fn load_fixed_opponents(cfg: &OpponentsConfig, dev: &Device) -> Vec<FixedOpponent> {
    cfg.fixed
        .iter()
        .map(|o| {
            let mut model = new_model(dev);
            let path = o.checkpoint.join("actor.bpk");
            model.actor = load_net_any(&path, NetKind::Actor, dev)
                .unwrap_or_else(|e| panic!("opponent {}: {}: {e}", o.name, path.display()));
            FixedOpponent {
                name: o.name.clone(),
                model,
                weight: o.weight,
            }
        })
        .collect()
}

/// One learner run: the policy learning rate and how many iterations to run (`None` = until stopped).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Phase {
    policy_lr: f32,
    iterations: Option<u64>,
}

/// Splits training into an optional critic-only warm-up and the main PPO phase,
/// given the cumulative steps already in the checkpoint being resumed.
fn plan_phases(cfg: &TrainConfig, done_steps: u64) -> Vec<Phase> {
    let main = Phase {
        policy_lr: cfg.policy_lr,
        iterations: (cfg.iterations > 0).then_some(cfg.iterations),
    };
    if done_steps >= cfg.critic_warmup_steps {
        return vec![main];
    }
    let ts = cfg.timesteps_per_iteration as u64;
    let warmup = Phase {
        policy_lr: 0.0,
        iterations: Some((cfg.critic_warmup_steps - done_steps).div_ceil(ts)),
    };
    vec![warmup, main]
}

/// `cumulative_timesteps` of the newest (highest-numbered) checkpoint, or 0.
fn checkpoint_steps(dir: &Path) -> u64 {
    let newest = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            e.file_name()
                .to_str()?
                .parse::<u64>()
                .ok()
                .map(|n| (n, e.path()))
        })
        .max_by_key(|(n, _)| *n);
    newest
        .and_then(|(_, p)| std::fs::read_to_string(p.join("stats.toml")).ok())
        .and_then(|t| t.parse::<toml::Table>().ok())
        .and_then(|t| t.get("cumulative_timesteps")?.as_integer())
        .map_or(0, |n| n as u64)
}

fn precision(bf16: bool) -> ComputePrecision {
    if bf16 {
        ComputePrecision::Bf16
    } else {
        ComputePrecision::F32
    }
}

fn device(backend: &str) -> Device {
    match backend {
        #[cfg(feature = "torch")]
        #[allow(deprecated)]
        "torch" => Device::libtorch_cuda(0),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        other => panic!("backend {other:?} is not compiled in"),
    }
}

fn read_rewards(path: &Path) -> RewardConfig {
    RewardConfig::from_file(path).unwrap_or_else(|e| panic!("reward config: {e}"))
}

fn learner_config(
    cfg: &TrainConfig,
    phase: Phase,
    device: Device,
    fixed_opponents: Vec<FixedOpponent>,
    external_opponents: Vec<ExternalOpponent>,
) -> LearnerConfig {
    let ts = cfg.timesteps_per_iteration;
    LearnerConfig {
        render: false,
        num_pools: cfg.pools,
        num_threads_per_pool: cfg.threads / cfg.pools,
        num_games_per_pool: cfg.games / cfg.pools,
        timesteps_per_save: cfg.save_every,
        checkpoints_limit: Some(cfg.keep_checkpoints),
        checkpoints_folder: cfg.checkpoints.clone(),
        num_additional_iterations: phase.iterations,
        metrics_jsonl: std::env::var("INDOCOCO_METRICS").ok().map(PathBuf::from),
        ppo: PpoLearnerConfig {
            timesteps_per_iteration: ts,
            batch_size: ts,
            mini_batch_size: cfg.mini_batch_size,
            gpu_timestep_buffer_size: ts,
            truncation_value_batch_size: ts,
            epochs: cfg.epochs,
            learning_rate: phase.policy_lr,
            critic_learning_rate: Some(cfg.critic_lr),
            lr_schedule: cfg.lr_schedule.schedule(ts as u64),
            entropy_scale: cfg.entropy,
            normalize_entropy: false,
            standardize_returns: true,
            gamma: cfg.gamma,
            lambda: cfg.lambda,
            precision: TrainPrecision {
                policy: precision(cfg.policy_bf16),
                critic: precision(cfg.critic_bf16),
            },
            ..Default::default()
        },
        self_play: SelfPlayConfig {
            save_policy_versions: true,
            ts_per_version: 250_000_000,
            max_old_versions: 10,
            max_saved_versions: Some(20),
            train_against_old_versions: cfg.self_play_old_versions,
            train_against_old_chance: cfg.opponents.chance,
            past_versions_weight: cfg.opponents.past_versions_weight,
            fixed_opponents,
            external_opponents,
        },
        skill_tracker: SkillTrackerConfig {
            enabled: cfg.skill_tracker,
            num_arenas: 12,
            update_interval: 20,
            async_eval: false,
            ..Default::default()
        },
        // The Python DiscreteFF has no input normalization or shared head.
        norm: rlgymppo::NormSelection::None,
        shared_head_layer_sizes: vec![],
        policy_layer_sizes: LAYERS.to_vec(),
        critic_layer_sizes: LAYERS.to_vec(),
        render_device: device.clone(),
        device: device.autodiff(),
        skill_tracker_device: None,
        ..Default::default()
    }
}

fn train(config: &Path) {
    let text = std::fs::read_to_string(config).expect("read train config");
    let cfg: TrainConfig = toml::from_str(&text).expect("parse train config");
    let rewards = read_rewards(&cfg.reward_config);
    init_from_default(true).expect("load collision meshes");
    let dev = device(&cfg.backend);
    let fixed = load_fixed_opponents(&cfg.opponents, &dev);
    let external = load_external_opponents(&cfg.opponents, &dev);
    for phase in plan_phases(&cfg, checkpoint_steps(&cfg.checkpoints)) {
        println!(
            "PHASE policy_lr={} iterations={:?}",
            phase.policy_lr, phase.iterations
        );
        let (rewards, setters) = (rewards.clone(), cfg.state_setters.clone());
        let mut learner = learner_config(&cfg, phase, dev.clone(), fixed.clone(), external.clone()).init(
            move |_| create_env(&rewards, &setters),
            default_adamw_optimizer(),
        );
        learner.load();
        learner.learn();
    }
}

/// Fixed probe input used to compare logits across PyTorch and Burn.
fn probe_input() -> Vec<f32> {
    (0..obs_size())
        .map(|i| ((i as f32) * 0.37).sin() * 0.8)
        .collect()
}

fn new_model(dev: &Device) -> Actic {
    model_with_input(obs_size(), dev)
}

fn model_with_input(input: usize, dev: &Device) -> Actic {
    Actic::new(
        input,
        N_ACTIONS,
        LAYERS.to_vec(),
        LAYERS.to_vec(),
        &[],
        dev,
        None,
    )
}

/// Python `DiscreteFF` keys `model.{0,2,4,6,8}` <-> Burn `linear_layers.{0..4}`.
const KEY_MAP: [(&str, &str); 5] = [
    ("model.0.", "linear_layers.0."),
    ("model.2.", "linear_layers.1."),
    ("model.4.", "linear_layers.2."),
    ("model.6.", "linear_layers.3."),
    ("model.8.", "linear_layers.4."),
];

fn print_probe_logits(actor: &rlgymppo::Net, dev: &Device) {
    let x = Tensor::<2>::from_data(TensorData::new(probe_input(), [1, obs_size()]), dev);
    let logits = actor
        .forward(x)
        .into_data()
        .try_to_vec::<f32>()
        .expect("logits");
    println!(
        "PROBE_LOGITS {}",
        serde_json::to_string(&logits).expect("json")
    );
}

/// Writes checkpoint `<out>/1` with the Python actor and a fresh critic.
/// The `.pt` file is only read.
fn import(pt: &Path, out: &Path) {
    let dev = Device::flex();
    let out = out.join("1");
    std::fs::create_dir_all(&out).expect("create checkpoint dir");
    let mut model = new_model(&dev);
    let mut store = PytorchStore::from_file(pt);
    for (from, to) in KEY_MAP {
        store = store.with_key_remapping(format!("^{}", regex_escape(from)), to);
    }
    let result = model.actor.load_from(&mut store).expect("load .pt");
    println!("import: {result:?}");
    print_probe_logits(&model.actor, &dev);
    model
        .actor
        .clone()
        .save_file(out.join("actor.bpk"))
        .expect("save actor");
    model
        .critic
        .clone()
        .save_file(out.join("critic.bpk"))
        .expect("save critic");
    std::fs::write(
        out.join("stats.toml"),
        "cumulative_timesteps = 0\ncumulative_model_updates = 0\ncumulative_epochs = 0\ntransfer_rounds = 0\n\n[return_stat]\nmean = 0.0\nmean_sqr = 0.0\ncount = 0\n",
    )
    .expect("write stats");
    println!("wrote {}", out.display());
}

fn regex_escape(s: &str) -> String {
    s.replace('.', r"\.")
}

/// Exports an actor to PyTorch-layout safetensors (`model.{0,2,4,6,8}`).
fn export(ckpt: &Path, out: &Path) {
    let dev = Device::flex();
    let actor = new_model(&dev)
        .actor
        .try_load_file(ckpt.join("actor.bpk"))
        .expect("load actor");
    let mut store = SafetensorsStore::from_file(out)
        .overwrite(true)
        .with_to_adapter(BurnToPyTorchAdapter);
    for (to, from) in KEY_MAP {
        store = store.with_key_remapping(format!("^{}", regex_escape(from)), to);
    }
    actor.save_into(&mut store).expect("save safetensors");
    print_probe_logits(&actor, &dev);
}

/// Records `steps` environment steps for the parity checks: one JSON line
/// per step with the [`StateView`] and both observations. `reset` marks the
/// first state of an episode (rewards are reset on it, not scored); every
/// other line is scored. Actions come from the checkpoint's policy (argmax,
/// 20% random) when given, otherwise uniformly random.
fn record(reward_toml: &Path, out: &Path, steps: usize, ckpt: Option<&Path>) {
    init_from_default(true).expect("load collision meshes");
    let cfg = read_rewards(reward_toml);
    let dev = Device::flex();
    let actor = ckpt.map(|c| {
        new_model(&dev)
            .actor
            .try_load_file(c.join("actor.bpk"))
            .expect("load actor")
    });
    let mut env = create_env(&cfg, &default_mix());
    let mut rng = SmallRng::seed_from_u64(7);
    let mut lines = Vec::with_capacity(steps);

    let (mut state, mut obs, _) = env.reset();
    let mut reset = true;
    for _ in 0..steps {
        lines.push(record_line(&state, &obs, reset));
        let actions: Vec<usize> = obs
            .iter()
            .map(|o| match &actor {
                Some(a) if rng.random_range(0.0..1.0) > 0.2 => {
                    let x = Tensor::<2>::from_data(TensorData::new(o.clone(), [1, obs_size()]), &dev);
                    let logits = a
                        .forward(x)
                        .into_data()
                        .try_to_vec::<f32>()
                        .expect("logits");
                    (0..N_ACTIONS)
                        .max_by(|&i, &j| logits[i].total_cmp(&logits[j]))
                        .expect("actions")
                }
                _ => rng.random_range(0..N_ACTIONS),
            })
            .collect();
        let res = env.step(&state, &actions);
        reset = false;
        (state, obs) = (res.state, res.obs);
        if res.is_terminal || res.truncated {
            // Record the terminal state (it is scored), then start over.
            lines.push(record_line(&state, &obs, false));
            let (s, o, _) = env.reset();
            (state, obs) = (s, o);
            reset = true;
        }
    }
    std::fs::write(out, lines.join("\n")).expect("write recording");
    println!("recorded {} lines to {}", lines.len(), out.display());
}

/// One recorded step: the reward view plus what the Python obs builder needs
/// beyond it (pad timers in `BOOST_LOCATIONS` order, a few car flags).
fn record_line(state: &rlgymppo::rlgym::GameState, obs: &[Vec<f32>], reset: bool) -> String {
    let order = indococo_bot::obs::pad_order(state);
    let pads: Vec<f32> = order
        .iter()
        .map(|&k| state.boost_pads[k].1.cooldown)
        .collect();
    let extra: Vec<_> = state
        .cars
        .iter()
        .map(|(_, c)| {
            serde_json::json!({
                "handbrake": c.handbrake_val,
                "jump": c.controls.jump,
                "controls": indococo_bot::obs::prev_action(c),
                "is_jumping": c.is_jumping,
                "is_supersonic": c.is_supersonic,
            })
        })
        .collect();
    serde_json::json!({
        "reset": reset,
        "state": StateView::from_game_state(state),
        "pads": pads,
        "extra": extra,
        "obs": obs,
    })
    .to_string()
}

/// Flip reset counts of the candidate car in `rates`, for one setter label.
#[derive(Clone, Debug, Default, Serialize)]
struct ResetCounts {
    car_steps: u64,
    /// Episodes started by this setter.
    episodes: u64,
    /// Of those, episodes that start with the candidate in the air (the
    /// setter's flier), and how many of them reset within `EARLY_STEPS`.
    episodes_as_flier: u64,
    flier_episodes_with_early_reset: u64,
    /// Flier episodes whose first reset came after `EARNED_FROM_STEP` steps
    /// of control and within `EARLY_STEPS`: earned, as in the curriculum.
    flier_episodes_with_earned_reset: u64,
    episodes_with_reset: u64,
    /// Episodes with a reset in their first `EARLY_STEPS` steps (3 s): the
    /// setter's own situation, before play moves on.
    episodes_with_early_reset: u64,
    resets: u64,
    resets_old_detector: u64,
    /// Resets followed by a dodge and a goalward touch (`flip_reset_use`'s use).
    uses: u64,
    /// Team goals within 3 s of a reset.
    goals_after_reset: u64,
}

impl ResetCounts {
    fn add(&mut self, o: &Self) {
        self.car_steps += o.car_steps;
        self.episodes += o.episodes;
        self.episodes_as_flier += o.episodes_as_flier;
        self.flier_episodes_with_early_reset += o.flier_episodes_with_early_reset;
        self.flier_episodes_with_earned_reset += o.flier_episodes_with_earned_reset;
        self.episodes_with_reset += o.episodes_with_reset;
        self.episodes_with_early_reset += o.episodes_with_early_reset;
        self.resets += o.resets;
        self.resets_old_detector += o.resets_old_detector;
        self.uses += o.uses;
        self.goals_after_reset += o.goals_after_reset;
    }

    fn per_min(&self, n: u64) -> f64 {
        let minutes = self.car_steps as f64 * f64::from(indococo_bot::consts::TICK_SKIP)
            / indococo_bot::consts::TICKS_PER_SECOND
            / 60.0;
        if minutes > 0.0 {
            n as f64 / minutes
        } else {
            0.0
        }
    }

    fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "counts": self,
            "resets_per_min": self.per_min(self.resets),
            "resets_old_detector_per_min": self.per_min(self.resets_old_detector),
            "uses_per_min": self.per_min(self.uses),
            "episodes_with_reset_share": self.episodes_with_reset as f64 / self.episodes.max(1) as f64,
            "episodes_with_early_reset_share":
                self.episodes_with_early_reset as f64 / self.episodes.max(1) as f64,
            "flier_early_reset_share":
                self.flier_episodes_with_early_reset as f64 / self.episodes_as_flier.max(1) as f64,
            "flier_earned_reset_share":
                self.flier_episodes_with_earned_reset as f64 / self.episodes_as_flier.max(1) as f64,
        })
    }
}

/// Steps (3 s) during which a reset counts as early in its episode.
const EARLY_STEPS: u64 = 45;
/// First step whose reset counts as earned (after 0.33 s of control; see
/// `curriculum::Attempt`).
const EARNED_FROM_STEP: u64 = 6;

/// One environment of the `rates` run, the candidate on car 0 (blue).
struct RateSlot {
    env: BotEnv,
    state: GameState,
    obs: Vec<Vec<f32>>,
    prev: StateView,
    label: &'static str,
    steps_into_episode: u64,
    flier: bool,
    reset_this_episode: bool,
    uses: FlipResetUse,
    goals: FlipResetUse,
}

impl RateSlot {
    fn new(mix: &[WeightedSetter]) -> Self {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs, _) = env.reset();
        let only = |use_weight, goal_bonus| {
            FlipResetUse::new(FlipResetUseParams {
                obtain: 0.0,
                shot_bonus: 0.0,
                use_weight,
                goal_bonus,
                ..FlipResetUseParams::default()
            })
        };
        let mut slot = Self {
            prev: StateView::from_game_state(&state),
            label: env.get_shared_info().setter(),
            env,
            state,
            obs,
            steps_into_episode: 0,
            flier: false,
            reset_this_episode: false,
            uses: only(1.0, 0.0),
            goals: only(0.0, 1.0),
        };
        slot.start_episode();
        slot
    }

    fn start_episode(&mut self) {
        self.prev = StateView::from_game_state(&self.state);
        self.label = self.env.get_shared_info().setter();
        self.steps_into_episode = 0;
        self.flier = self.prev.cars[0].pos.z > indococo_bot::state::AIRBORNE_MIN_Z;
        self.reset_this_episode = false;
        self.uses.reset(&self.prev);
        self.goals.reset(&self.prev);
    }

    /// Steps with `actions` and counts the candidate's resets into `counts`.
    fn step(&mut self, actions: &[usize], counts: &mut BTreeMap<&'static str, ResetCounts>) {
        let res = self.env.step(&self.state, actions);
        let view = self.env.get_shared_info().view(&res.state);
        let c = counts.entry(self.label).or_default();
        c.car_steps += 1;
        if self.steps_into_episode == 0 {
            c.episodes += 1;
            c.episodes_as_flier += u64::from(self.flier);
        }
        self.steps_into_episode += 1;
        if resets::is_flip_reset(&self.prev.cars[0], &view.cars[0], &view.ball) {
            c.resets += 1;
            if !self.reset_this_episode {
                c.episodes_with_reset += 1;
                let early = self.steps_into_episode <= EARLY_STEPS;
                c.episodes_with_early_reset += u64::from(early);
                c.flier_episodes_with_early_reset += u64::from(early && self.flier);
                let earned = early && self.steps_into_episode >= EARNED_FROM_STEP;
                c.flier_episodes_with_earned_reset += u64::from(earned && self.flier);
            }
            self.reset_this_episode = true;
        }
        c.resets_old_detector +=
            u64::from(StepEvents::between(&self.prev, &view)[0].flip_reset_legacy);
        c.uses += u64::from(self.uses.get_rewards(&view)[0] > 0.0);
        c.goals_after_reset += u64::from(self.goals.get_rewards(&view)[0] > 0.0);
        (self.state, self.obs, self.prev) = (res.state, res.obs, view);
        if res.is_terminal || res.truncated {
            let (state, obs, _) = self.env.reset();
            (self.state, self.obs) = (state, obs);
            self.start_episode();
        }
    }
}

fn load_actor(ckpt: &Path, dev: &Device) -> Net {
    load_net_any(&ckpt.join("actor.bpk"), NetKind::Actor, dev)
        .unwrap_or_else(|e| panic!("load actor {}: {e}", ckpt.display()))
}

#[derive(Clone, Copy, PartialEq)]
enum NetKind {
    Actor,
    Critic,
}

/// Input width of a saved net (`linear_layers.0.weight` is `[in, out]`).
fn bpk_input_width(path: &Path) -> Result<usize, String> {
    use burn_store::{BurnpackStore, ModuleStore};
    let mut store = BurnpackStore::from_file(path);
    let t = store
        .get_tensor("linear_layers.0.weight")
        .map_err(|e| format!("{}: {e:?}", path.display()))?
        .ok_or_else(|| format!("{}: no linear_layers.0.weight", path.display()))?;
    Ok(t.shape[0])
}

/// Loads an actor or critic for this process's obs width. A pre-FR8 net
/// (base width) loaded in extended mode gets zero weights for the 14 new
/// inputs, so it computes exactly what it computed on the base obs.
fn load_net_any(path: &Path, kind: NetKind, dev: &Device) -> Result<Net, String> {
    let pick = |m: Actic| if kind == NetKind::Actor { m.actor } else { m.critic };
    let width = bpk_input_width(path)?;
    if width == obs_size() {
        return pick(new_model(dev)).try_load_file(path).map_err(|e| format!("{e:?}"));
    }
    if width != base_obs_size() {
        return Err(format!("{}: input width {width}, this process uses {}", path.display(), obs_size()));
    }
    let legacy = pick(model_with_input(base_obs_size(), dev))
        .try_load_file(path)
        .map_err(|e| format!("{e:?}"))?;
    let mut st = SafetensorsStore::from_bytes(None);
    legacy.save_into(&mut st).map_err(|e| format!("{e:?}"))?;
    let bytes = widen_safetensors(&st.get_bytes().map_err(|e| format!("{e:?}"))?, base_obs_size(), obs_size());
    let mut st = SafetensorsStore::from_bytes(Some(bytes));
    let mut net = pick(new_model(dev));
    net.load_from(&mut st).map_err(|e| format!("{e:?}"))?;
    Ok(net)
}

/// Rewrites `linear_layers.0.weight` (`[from, out]`, f32 row-major) as
/// `[to, out]` with zero rows for the new inputs.
fn widen_safetensors(bytes: &[u8], from: usize, to: usize) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("header len")) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).expect("safetensors header");
    let data = &bytes[8 + n..];
    let mut keys: Vec<(String, serde_json::Value)> = header.into_iter().collect();
    keys.sort_by_key(|(k, v)| if k == "__metadata__" { 0 } else { v["data_offsets"][0].as_u64().unwrap_or(0) + 1 });
    let mut out_header = serde_json::Map::new();
    let mut out_data = Vec::with_capacity(data.len() + (to - from) * 4 * 2048);
    for (k, mut v) in keys {
        if k == "__metadata__" {
            out_header.insert(k, v);
            continue;
        }
        let (a, b) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
        let start = out_data.len();
        out_data.extend_from_slice(&data[a..b]);
        if k == "linear_layers.0.weight" {
            let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            assert_eq!(shape[0], from, "{k} shape {shape:?}");
            assert_eq!(v["dtype"], "F32");
            out_data.resize(out_data.len() + (to - from) * shape[1] * 4, 0);
            v["shape"] = serde_json::json!([to, shape[1]]);
        }
        v["data_offsets"] = serde_json::json!([start, out_data.len()]);
        out_header.insert(k, v);
    }
    let mut h = serde_json::to_vec(&out_header).expect("header");
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(out_data);
    out
}

/// `widen <src_ck> <dst_ck>`: an FR8 (extended obs) checkpoint from a
/// pre-FR8 one, actor and critic first layers padded with zero weights;
/// stats.toml copied. Prints the max |logit diff| and |value diff| between
/// the source on base obs and the widened nets on extended obs.
fn widen(src: &Path, dst: &Path) {
    assert!(indococo_bot::obs::obs_ext(), "widen needs INDOCOCO_OBS_EXT=1");
    init_from_default(true).expect("load collision meshes");
    let dev = Device::flex();
    std::fs::create_dir_all(dst).expect("create dst");
    let actor = load_net_any(&src.join("actor.bpk"), NetKind::Actor, &dev).expect("actor");
    let critic = load_net_any(&src.join("critic.bpk"), NetKind::Critic, &dev).expect("critic");
    actor.clone().save_file(dst.join("actor.bpk")).expect("save actor");
    critic.clone().save_file(dst.join("critic.bpk")).expect("save critic");
    std::fs::copy(src.join("stats.toml"), dst.join("stats.toml")).expect("copy stats");
    // Verify on real env obs (random + kickoff states) against the source.
    let base_actor = model_with_input(base_obs_size(), &dev).actor.try_load_file(src.join("actor.bpk")).expect("src actor");
    let base_critic = model_with_input(base_obs_size(), &dev).critic.try_load_file(src.join("critic.bpk")).expect("src critic");
    let re_actor = load_net_any(&dst.join("actor.bpk"), NetKind::Actor, &dev).expect("reload actor");
    let mut env = create_env(&RewardConfig::default(), &default_mix());
    let mut rng = SmallRng::seed_from_u64(11);
    let (mut state, mut obs, _) = env.reset();
    let (mut ext_rows, mut base_rows) = (Vec::new(), Vec::new());
    for _ in 0..512 {
        for o in &obs {
            ext_rows.extend_from_slice(o);
            base_rows.extend_from_slice(&o[..base_obs_size()]);
        }
        let actions: Vec<usize> = obs.iter().map(|_| rng.random_range(0..N_ACTIONS)).collect();
        let res = env.step(&state, &actions);
        (state, obs) = (res.state, res.obs);
        if res.is_terminal || res.truncated {
            let (s, o, _) = env.reset();
            (state, obs) = (s, o);
        }
    }
    let n = ext_rows.len() / obs_size();
    let xe = Tensor::<2>::from_data(TensorData::new(ext_rows, [n, obs_size()]), &dev);
    let xb = Tensor::<2>::from_data(TensorData::new(base_rows, [n, base_obs_size()]), &dev);
    let maxdiff = |a: Tensor<2>, b: Tensor<2>| -> f32 {
        (a - b).abs().max().into_data().try_to_vec::<f32>().expect("diff")[0]
    };
    let ext_nonzero = {
        let e = xe.clone().slice([0..n, base_obs_size()..obs_size()]).abs().max();
        e.into_data().try_to_vec::<f32>().expect("ext")[0]
    };
    let out = serde_json::json!({
        "rows": n,
        "ext_max_abs_input": ext_nonzero,
        "actor_max_abs_logit_diff": maxdiff(actor.forward(xe.clone()), base_actor.forward(xb.clone())),
        "reloaded_actor_max_abs_logit_diff": maxdiff(re_actor.forward(xe.clone()), base_actor.forward(xb.clone())),
        "critic_max_abs_value_diff": maxdiff(critic.forward(xe), base_critic.forward(xb)),
        "dst": dst,
    });
    println!("WIDEN {out}");
}

/// Share of `rates` decisions replaced by a random action. Greedy policies
/// from the same kickoff otherwise replay the same game in every env.
const RATES_RANDOM_ACTION: f64 = 0.03;

/// Greedy actions of `actor` for a batch of observations.
fn argmax_actions(actor: &Net, obs: &[&Vec<f32>], dev: &Device) -> Vec<usize> {
    let flat: Vec<f32> = obs.iter().flat_map(|o| o.iter().copied()).collect();
    let x = Tensor::<2>::from_data(TensorData::new(flat, [obs.len(), obs_size()]), dev);
    let logits = actor
        .forward(x)
        .into_data()
        .try_to_vec::<f32>()
        .expect("logits");
    logits
        .chunks(N_ACTIONS)
        .map(|row| {
            (0..N_ACTIONS)
                .max_by(|&i, &j| row[i].total_cmp(&row[j]))
                .expect("actions")
        })
        .collect()
}

fn rates_worker(
    mix: &[WeightedSetter],
    candidate: &Path,
    opponent: Option<&Path>,
    steps: usize,
) -> BTreeMap<&'static str, ResetCounts> {
    const ENVS: usize = 16;
    let dev = Device::flex();
    let cand = load_actor(candidate, &dev);
    let opp = opponent.map(|p| load_actor(p, &dev));
    let mut slots: Vec<RateSlot> = (0..ENVS).map(|_| RateSlot::new(mix)).collect();
    let mut counts = BTreeMap::new();
    let mut rng = SmallRng::seed_from_u64(rand::rng().next_u64());
    let mut explore = |a: usize| {
        if rng.random_bool(RATES_RANDOM_ACTION) {
            rng.random_range(0..N_ACTIONS)
        } else {
            a
        }
    };
    for _ in 0..steps.div_ceil(ENVS) {
        let blue: Vec<&Vec<f32>> = slots.iter().map(|s| &s.obs[0]).collect();
        let orange: Vec<&Vec<f32>> = slots.iter().map(|s| &s.obs[1]).collect();
        let a0 = argmax_actions(&cand, &blue, &dev);
        let a1 = argmax_actions(opp.as_ref().unwrap_or(&cand), &orange, &dev);
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.step(&[explore(a0[i]), explore(a1[i])], &mut counts);
        }
    }
    counts
}

/// Plays `steps` steps (greedy policies with 3% random actions, candidate on
/// blue) over the setter
/// mix in `setters_toml` and prints the candidate's flip reset rates per
/// setter and overall as one JSON line.
fn rates(setters_toml: &Path, candidate: &Path, opponent: &str, steps: usize, threads: usize) {
    #[derive(Deserialize)]
    struct Mix {
        state_setters: Vec<WeightedSetter>,
    }
    init_from_default(true).expect("load collision meshes");
    let text = std::fs::read_to_string(setters_toml).expect("read setters");
    let mix: Mix = toml::from_str(&text).expect("parse setters");
    let opponent = (opponent != "self").then(|| PathBuf::from(opponent));
    let per_thread = steps.div_ceil(threads);
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (mix, opp) = (&mix.state_setters, opponent.as_deref());
                s.spawn(move || rates_worker(mix, candidate, opp, per_thread))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    let mut by_label: BTreeMap<&str, ResetCounts> = BTreeMap::new();
    let mut all = ResetCounts::default();
    for counts in &results {
        for (label, c) in counts {
            by_label.entry(label).or_default().add(c);
            all.add(c);
        }
    }
    let per_setter: serde_json::Map<String, serde_json::Value> = by_label
        .iter()
        .map(|(k, c)| ((*k).to_string(), c.summary()))
        .collect();
    let out = serde_json::json!({
        "candidate": candidate,
        "opponent": opponent.as_deref().map_or("self".into(), |p| p.display().to_string()),
        "setters": setters_toml,
        "steps": steps,
        "all": all.summary(),
        "per_setter": per_setter,
    });
    println!("{out}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| -> &str {
        args.get(i)
            .map_or_else(|| panic!("missing argument {i}"), String::as_str)
    };
    match arg(0) {
        "train" => train(Path::new(arg(1))),
        "curr" => bc_cli::curr(&args[1..]),
        "normal-obs" => bc_cli::normal_obs(&args[1..]),
        "import" => import(Path::new(arg(1)), Path::new(arg(2))),
        "widen" => widen(Path::new(arg(1)), Path::new(arg(2))),
        "export" => export(Path::new(arg(1)), Path::new(arg(2))),
        "record" => record(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("steps"),
            args.get(4).map(Path::new),
        ),
        "snapshots" => snapshots(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("steps"),
            arg(4).parse().expect("threads"),
            Path::new(arg(5)),
        ),
        "rates" => rates(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3),
            arg(4).parse().expect("steps"),
            args.get(5).map_or(8, |t| t.parse().expect("threads")),
        ),
        "h2h" => h2h(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("games"),
            args.get(4).map_or(8, |t| t.parse().expect("threads")),
        ),
        "bench-infer" => bench_infer(
            Path::new(arg(1)),
            arg(2).parse().expect("rows"),
            arg(3).parse().expect("iters"),
        ),
        "galactus-h2h" => galactus_h2h(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("games"),
            args.get(4).map_or(8, |t| t.parse().expect("threads")),
            args.get(5).map(Path::new),
        ),
        "megabyte-obs" => megabyte_obs(Path::new(arg(1)), Path::new(arg(2))),
        "megabyte-h2h" => megabyte_h2h(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("games"),
            args.get(4).map_or(8, |t| t.parse().expect("threads")),
        ),
        "megabyte-logits" => megabyte_logits(Path::new(arg(1)), Path::new(arg(2))),
        other => panic!("unknown mode {other:?}"),
    }
}

/// `galactus-h2h`: the candidate checkpoint (greedy) against the Rust
/// Galactus port, first goal wins, kickoff starts, both sides. Galactus holds
/// its previous controls for 7 ticks and applies the new action on the 8th,
/// as in the collector. With `dump`, worker 0 writes its first 4000 Galactus
/// decisions (raw state + Rust obs without slot shuffle + logits + action)
/// for `tools/galactus/parity.py`.
fn galactus_h2h(ckpt: &Path, gdir: &Path, games: usize, threads: usize, dump: Option<&Path>) {
    init_from_default(true).expect("load collision meshes");
    let mix = vec![WeightedSetter {
        weight: 1.0,
        kind: indococo_bot::setters::SetterKind::Kickoff,
    }];
    let per = games.div_ceil(threads);
    let t0 = std::time::Instant::now();
    let res: Vec<[u64; 4]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let mix = &mix;
                let dump = if t == 0 { dump } else { None };
                s.spawn(move || galactus_worker(mix, ckpt, gdir, per, dump))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = [0u64; 4];
    for r in &res {
        for i in 0..4 {
            tot[i] += r[i];
        }
    }
    let [w, l, d, steps] = tot;
    let out = serde_json::json!({
        "a": ckpt, "b": "galactus(rust)", "games": w + l + d, "wins": w, "losses": l, "draws": d,
        "win_rate_decisive": if w + l > 0 { w as f64 / (w + l) as f64 } else { 0.5 },
        "steps": steps, "seconds": t0.elapsed().as_secs_f64(),
    });
    println!("{out}");
}

fn v3(v: rlgymppo::rlgym::rocketsim::Vec3A) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn galactus_worker(
    mix: &[WeightedSetter],
    ckpt: &Path,
    gdir: &Path,
    games: usize,
    dump: Option<&Path>,
) -> [u64; 4] {
    use indococo_bot::galactus as gx;
    use rlgymppo::ExternalPolicy;
    use std::io::Write;
    const ENVS: usize = 8;
    let dev = Device::flex();
    let na = load_actor(ckpt, &dev);
    let gal = gx::GalactusPolicy::load(gdir, &dev).expect("galactus weights");
    let mut dump = dump.map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("dump")));
    let mut dumped = 0usize;
    let mut out = [0u64; 4];
    let mut started = 0usize;
    struct G {
        env: BotEnv,
        state: GameState,
        obs: Vec<Vec<f32>>,
        a_team: u8,
        live: bool,
    }
    let mut new_game = |k: usize, started: &mut usize| -> G {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs, _) = env.reset();
        let live = *started < games;
        *started += 1;
        G { env, state, obs, a_team: (k % 2) as u8, live }
    };
    let mut gs: Vec<G> = (0..ENVS).map(|k| new_game(k, &mut started)).collect();
    let mut gobs = vec![0.0f32; gx::OBS_SIZE];
    let mut gmask = vec![false; gx::N_ACTIONS];
    while gs.iter().any(|g| g.live) {
        let mut oa = Vec::new();
        for g in gs.iter().filter(|g| g.live) {
            for (i, (info, _)) in g.state.cars.iter().enumerate() {
                if info.team as u8 == g.a_team {
                    oa.push(&g.obs[i]);
                }
            }
        }
        let aa = argmax_actions(&na, &oa, &dev);
        let mut ia = 0;
        for k in 0..ENVS {
            if !gs[k].live {
                continue;
            }
            let g = &mut gs[k];
            let mut acts = Vec::new();
            let mut ext = Vec::new();
            for (i, (info, car)) in g.state.cars.iter().enumerate() {
                if info.team as u8 == g.a_team {
                    acts.push(aa[ia]);
                    ia += 1;
                } else {
                    acts.push(0);
                    gal.build(&g.state, i, &mut gobs, &mut gmask);
                    let logits = gal.logits(&gobs, 1);
                    let a = gx::masked_argmax(&logits, &gmask, 1)[0];
                    if let Some(f) = dump.as_mut().filter(|_| dumped < 4000) {
                        let pred = gx::ball_prediction(&g.state.ball);
                        let plain = gx::build_obs_with_pred(&g.state, i, &pred, false);
                        let order = indococo_bot::obs::pad_order(&g.state);
                        let cars: Vec<_> = g.state.cars.iter().map(|(ci, c)| serde_json::json!({
                            "team": ci.team as u8, "pos": v3(c.phys.pos), "vel": v3(c.phys.vel),
                            "ang_vel": v3(c.phys.ang_vel),
                            "rot": [v3(c.phys.rot_mat.x_axis), v3(c.phys.rot_mat.y_axis), v3(c.phys.rot_mat.z_axis)],
                            "boost": c.boost, "on_ground": indococo_bot::obs::flags(c).on_ground,
                            "has_flipped": c.has_flipped, "has_double_jumped": c.has_double_jumped,
                            "air_time_since_jump": c.air_time_since_jump, "is_demoed": c.is_demoed,
                            "has_jumped": c.has_jumped,
                            "controls": [c.controls.throttle, c.controls.steer, c.controls.pitch, c.controls.yaw,
                                c.controls.roll, f32::from(c.controls.jump), f32::from(c.controls.boost), f32::from(c.controls.handbrake)],
                        })).collect();
                        let lp = gal.logits(&plain, 1);
                        let ap = gx::masked_argmax(&lp, &gmask, 1)[0];
                        let pads: Vec<f32> = order.iter().map(|&p| g.state.boost_pads[p].1.cooldown).collect();
                        let line = serde_json::json!({
                            "car": i, "cars": cars, "pads": pads,
                            "ball": {"pos": v3(g.state.ball.phys.pos), "vel": v3(g.state.ball.phys.vel), "ang_vel": v3(g.state.ball.phys.ang_vel)},
                            "obs": plain, "mask": gmask, "logits": lp, "action": ap, "action_played": a,
                        });
                        writeln!(f, "{line}").expect("dump write");
                        dumped += 1;
                    }
                    ext.push((i, car.controls, gx::controls_of(a)));
                }
            }
            g.env.pre_step(&g.state, &acts);
            for &(i, held, _) in &ext {
                g.env.arena.set_car_controls(i, held);
            }
            g.env.step_physics(gx::ACTION_DELAY);
            for &(i, _, new) in &ext {
                g.env.arena.set_car_controls(i, new);
            }
            g.env.step_physics(gx::TICK_SKIP - gx::ACTION_DELAY);
            let res = g.env.post_step();
            out[3] += 1;
            if res.is_terminal || res.truncated {
                let view = g.env.get_shared_info().view(&res.state);
                match view.scoring_team() {
                    Some(t) if t == g.a_team => out[0] += 1,
                    Some(_) => out[1] += 1,
                    None => out[2] += 1,
                }
                *g = new_game(k, &mut started);
            } else {
                (g.state, g.obs) = (res.state, res.obs);
            }
        }
    }
    out
}

/// `megabyte-obs`: dump Megabyte obs for deterministic kickoff states.
/// Generates 20 obs (10 seeds × 2 cars) for validation. Optionally runs inference.
fn megabyte_obs(mdir: &Path, _states_path: &Path) {
    use indococo_bot::megabyte as mx;
    use rlgymppo::ExternalPolicy;
    
    init_from_default(true).expect("load collision meshes");
    let dev = tch::Device::Cuda(0);
    let mb = mx::MegabytePolicy::load(mdir, dev).expect("megabyte weights");
    
    // Generate deterministic kickoff states
    let mix = vec![WeightedSetter { weight: 1.0, kind: indococo_bot::setters::SetterKind::Kickoff }];
    let mut env = create_env(&RewardConfig::default(), &mix);
    
    eprintln!("Generating deterministic kickoff obs (10 seeds × 2 cars)...");
    for seed in 0..10 {
        let (state, _, _) = env.reset();
        for car_idx in 0..2 {
            let mut obs = vec![0.0f32; mx::OBS_SIZE];
            let mut mask = vec![false; mx::N_ACTIONS];
            mb.build(&state, car_idx, &mut obs, &mut mask);
            
            // Optionally run inference
            let actions = mb.infer(&obs, &mask, 1);
            
            let out = serde_json::json!({
                "seed": seed,
                "car": car_idx,
                "obs": obs,
                "action": actions[0],
            });
            println!("{out}");
        }
    }
    eprintln!("✓ Generated 20 observations (233 floats each)");
}

/// `megabyte-logits`: read obs vectors from JSON and output logits + argmax.
/// Each line should be {"obs": [233 floats]}. Outputs {"logits": [90 floats], "action": int}.
fn megabyte_logits(mdir: &Path, obs_file: &Path) {
    use indococo_bot::megabyte as mx;
    use std::io::{BufRead, BufReader};
    
    init_from_default(true).expect("load collision meshes");
    let dev = tch::Device::Cuda(0);
    let mb = mx::MegabytePolicy::load(mdir, dev).expect("megabyte weights");
    
    let f = std::fs::File::open(obs_file).expect("open obs file");
    let reader = BufReader::new(f);
    
    for line in reader.lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        
        let json: serde_json::Value = serde_json::from_str(&line).expect("parse JSON");
        let obs_arr = json["obs"].as_array().expect("obs array");
        let obs: Vec<f32> = obs_arr.iter()
            .map(|v| v.as_f64().expect("obs float") as f32)
            .collect();
        
        if obs.len() != mx::OBS_SIZE {
            eprintln!("Warning: expected {} obs, got {}", mx::OBS_SIZE, obs.len());
            continue;
        }
        
        let mask = vec![false; mx::N_ACTIONS];
        let logits = mb.logits(&obs, 1);
        
        // Find argmax
        let mut max_idx = 0;
        let mut max_val = logits[0];
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        
        let out = serde_json::json!({
            "logits": logits,
            "action": max_idx,
        });
        println!("{out}");
    }
}

fn megabyte_h2h(ckpt: &Path, mdir: &Path, games: usize, threads: usize) {
    // Ensure obs extension enabled for 106-wide actor (as galactus-h2h does)
    unsafe { std::env::set_var("INDOCOCO_OBS_EXT", "1"); }
    
    init_from_default(true).expect("load collision meshes");
    let mix = vec![WeightedSetter { weight: 1.0, kind: indococo_bot::setters::SetterKind::Kickoff }];
    let per = games.div_ceil(threads);
    let t0 = std::time::Instant::now();
    let res: Vec<[u64; 4]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let mix = &mix;
            s.spawn(move || megabyte_worker(mix, ckpt, mdir, per))
        }).collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = [0u64; 4];
    for r in &res { for i in 0..4 { tot[i] += r[i]; } }
    let [w, l, d, steps] = tot;
    let out = serde_json::json!({
        "a": ckpt, "b": "megabyte(rust)", "games": w + l + d, "wins": w, "losses": l, "draws": d,
        "win_rate_decisive": if w + l > 0 { w as f64 / (w + l) as f64 } else { 0.5 },
        "steps": steps, "seconds": t0.elapsed().as_secs_f64(),
    });
    println!("{out}");
}

fn megabyte_worker(mix: &[WeightedSetter], ckpt: &Path, mdir: &Path, games: usize) -> [u64; 4] {
    use indococo_bot::megabyte as mx;
    use rlgymppo::ExternalPolicy;
    const ENVS: usize = 8;
    let dev = Device::flex();
    let na = load_actor(ckpt, &dev);
    let mb = mx::MegabytePolicy::load(mdir, tch::Device::Cuda(0)).expect("megabyte weights");
    let mut out = [0u64; 4];
    let mut started = 0usize;
    struct G { env: BotEnv, state: GameState, obs: Vec<Vec<f32>>, a_team: u8, live: bool }
    let mut new_game = |k: usize, started: &mut usize| -> G {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs, _) = env.reset();
        let live = *started < games;
        *started += 1;
        G { env, state, obs, a_team: (k % 2) as u8, live }
    };
    let mut g: Vec<G> = (0..ENVS).map(|k| new_game(k, &mut started)).collect();
    let mut mb_obs: Vec<f32> = vec![0.0; ENVS * mx::OBS_SIZE];
    let mut mb_mask: Vec<bool> = vec![false; ENVS * mx::N_ACTIONS];
    loop {
        let live = g.iter().filter(|e| e.live).count();
        if live == 0 { break; }
        // Get candidate actions (greedy)
        let mut oa = Vec::new();
        for e in g.iter().filter(|e| e.live) {
            for (i, (info, _)) in e.state.cars.iter().enumerate() {
                if info.team as u8 == e.a_team {
                    oa.push(&e.obs[i]);
                }
            }
        }
        let aa = argmax_actions(&na, &oa, &dev);
        let mut ia = 0;
        // Build Megabyte obs and get actions
        for k in 0..ENVS {
            if !g[k].live { continue; }
            let i = if g[k].a_team == 0 { 0 } else { 1 };
            let off = k * mx::OBS_SIZE;
            let moff = k * mx::N_ACTIONS;
            mb.build(&g[k].state, i, &mut mb_obs[off..off + mx::OBS_SIZE], &mut mb_mask[moff..moff + mx::N_ACTIONS]);
        }
        let mb_acts = mb.infer(&mb_obs, &mb_mask, ENVS);
        // Step environments
        for k in 0..ENVS {
            if !g[k].live { continue; }
            let a_act = aa[ia];
            ia += 1;
            let mb_act = mb_acts[k];
            let acts = [a_act, mb_act];
            let acts_order = if g[k].a_team == 0 { &acts[..] } else { &[acts[1], acts[0]][..] };
            let e = &mut g[k];
            e.env.pre_step(&e.state, acts_order);
            e.env.step_physics(mx::TICK_SKIP);
            let res = e.env.post_step();
            out[3] += 1;
            if res.is_terminal || res.truncated {
                let view = g[k].env.get_shared_info().view(&res.state);
                match view.scoring_team() {
                    Some(t) if t == g[k].a_team => out[0] += 1,
                    Some(_) => out[1] += 1,
                    None => out[2] += 1,
                }
                g[k] = new_game(k, &mut started);
            } else {
                (g[k].state, g[k].obs) = (res.state, res.obs);
            }
        }
    }
    out
}

/// `bench-infer <ckpt> <rows> <iters>`: per-decision inference cost of the
/// collector paths (one full-batch submit vs the two indexed submits of an
/// opponent iteration), on the training device.
fn bench_infer(ckpt: &Path, rows: usize, iters: usize) {
    let dev = device("torch");
    let mut a = new_model(&dev);
    a.actor = load_actor(ckpt, &dev);
    let b = a.clone();
    let mut fixed = new_model(&dev);
    fixed.actor = load_actor(ckpt, &dev);
    let w = obs_size();
    let obs: Vec<f32> = (0..rows * w).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let half: Vec<usize> = (0..rows).step_by(2).collect();
    let other: Vec<usize> = (1..rows).step_by(2).collect();
    let time = |name: &str, f: &mut dyn FnMut()| {
        for _ in 0..50 { f(); }
        let t = std::time::Instant::now();
        for _ in 0..iters { f(); }
        println!("{name}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / iters as f64);
    };
    time("full flat", &mut || { a.submit_react_flat(&obs, rows, w, &[], 0, &dev).wait(); });
    time("indexed half (current)", &mut || { a.submit_react_indexed_flat(&obs, w, &[], 0, &half, &dev).wait(); });
    time("two indexed, same model clone", &mut || {
        let p = a.submit_react_indexed_flat(&obs, w, &[], 0, &half, &dev);
        let q = b.submit_react_indexed_flat(&obs, w, &[], 0, &other, &dev);
        p.wait(); q.wait();
    });
    time("two indexed, separately loaded model", &mut || {
        let p = a.submit_react_indexed_flat(&obs, w, &[], 0, &half, &dev);
        let q = fixed.submit_react_indexed_flat(&obs, w, &[], 0, &other, &dev);
        p.wait(); q.wait();
    });
    time("indexed half (fixed only)", &mut || { fixed.submit_react_indexed_flat(&obs, w, &[], 0, &half, &dev).wait(); });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(warmup: u64, iterations: u64) -> TrainConfig {
        let text = include_str!("../../config/train.toml");
        let mut c: TrainConfig = toml::from_str(text).expect("train.toml");
        c.critic_warmup_steps = warmup;
        c.iterations = iterations;
        c
    }

    #[test]
    fn warmup_freezes_policy_then_trains() {
        let c = cfg(30_000_000, 0);
        let p = plan_phases(&c, 0);
        assert_eq!(
            p,
            vec![
                Phase {
                    policy_lr: 0.0,
                    iterations: Some(300)
                },
                Phase {
                    policy_lr: c.policy_lr,
                    iterations: None
                },
            ]
        );
    }

    #[test]
    fn warmup_resumes_partially_and_is_skipped_when_done() {
        let c = cfg(30_000_000, 50);
        assert_eq!(plan_phases(&c, 29_950_000)[0].iterations, Some(1));
        assert_eq!(
            plan_phases(&c, 30_000_000),
            vec![Phase {
                policy_lr: c.policy_lr,
                iterations: Some(50)
            }]
        );
        assert_eq!(plan_phases(&cfg(0, 0), 0).len(), 1);
    }

    fn parse(extra: &str) -> TrainConfig {
        let text = format!("{}\n{extra}", include_str!("../../config/train.toml"));
        toml::from_str(&text).expect("train config")
    }

    #[test]
    fn defaults_keep_the_previous_behavior() {
        let c = parse("");
        assert_eq!(c.lr_schedule.schedule(100_000), LrSchedule::Constant);
        assert_eq!(c.opponents.chance, 0.15);
        assert!(c.opponents.fixed.is_empty());
        assert_eq!(c.state_setters, default_mix());
    }

    #[test]
    fn lr_schedule_converts_steps_to_iterations() {
        let c = parse(
            "[lr_schedule]\nkind = \"cosine\"\nhold_steps = 300_000_000\ndecay_steps = 250_000_000\nmin_scale = 0.1",
        );
        let s = c.lr_schedule.schedule(100_000);
        assert_eq!(
            s,
            LrSchedule::HoldThenDecay {
                hold_iterations: 3000,
                decay_iterations: 2500,
                min_scale: 0.1,
                shape: DecayShape::Cosine
            }
        );
        assert_eq!(s.scale(3000), 1.0);
        assert!((s.scale(5500) - 0.1).abs() < 1e-6);
        assert!(toml::from_str::<LrScheduleConfig>("kind = \"step\"").is_err());
    }

    #[test]
    fn opponents_and_state_setters_parse() {
        let c = parse(
            r#"
            [opponents]
            chance = 0.3
            past_versions_weight = 2.0
            [[opponents.fixed]]
            name = "1v1.pt"
            checkpoint = "opponents/1v1"
            weight = 1.0

            [[state_setters]]
            kind = "kickoff"
            weight = 1.0
            [[state_setters]]
            kind = "air_near_ball"
            weight = 1.0
            "#,
        );
        assert_eq!(c.opponents.chance, 0.3);
        assert_eq!(c.opponents.fixed[0].checkpoint, Path::new("opponents/1v1"));
        assert_eq!(c.state_setters.len(), 2);
    }
}

/// Steps of history kept per env for `snapshots` (a recorded state is taken
/// 0..=SNAP_LAG steps before the trigger, so the policy still has to set up).
const SNAP_LAG: usize = 15;

/// One car in a reset-shaped aerial spot: off the ground above 300 uu, the
/// ball above it (higher than 400 uu) within 1000 uu.
fn aerial_spot(state: &GameState, i: usize) -> bool {
    let car = &state.cars[i].1;
    let (c, b) = (car.phys.pos, state.ball.phys.pos);
    let wheels = car.wheels_with_contact().iter().any(Option::is_some);
    !wheels && c.z > 300.0 && b.z > 400.0 && b.z > c.z + 50.0 && (b - c).length() < 1000.0
}

fn snapshots_worker(mix: &[WeightedSetter], ckpt: &Path, steps: usize, seed: u64) -> Vec<f32> {
    const ENVS: usize = 16;
    let dev = Device::flex();
    let actor = load_actor(ckpt, &dev);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut envs: Vec<_> = (0..ENVS)
        .map(|_| {
            let mut env = create_env(&RewardConfig::default(), mix);
            let (state, obs, _) = env.reset();
            (env, state, obs, std::collections::VecDeque::<Vec<f32>>::new(), [0u64; 2], 0u64)
        })
        .collect();
    let mut out = Vec::new();
    for _ in 0..steps.div_ceil(ENVS) {
        let obs: Vec<&Vec<f32>> = envs.iter().flat_map(|e| e.2.iter()).collect();
        let acts = argmax_actions(&actor, &obs, &dev);
        for (k, (env, state, o, hist, cool, t)) in envs.iter_mut().enumerate() {
            let label = env.get_shared_info().setter();
            let cars: Vec<_> = state.cars.iter().map(|(_, c)| c).collect();
            hist.push_back(indococo_bot::setters::encode_snapshot(&state.ball, &cars));
            if hist.len() > SNAP_LAG + 1 {
                hist.pop_front();
            }
            *t += 1;
            if matches!(label, "kickoff" | "random") {
                for i in 0..2 {
                    if aerial_spot(state, i) && *t >= cool[i] {
                        let lag = rng.random_range(0..hist.len());
                        out.extend_from_slice(&hist[hist.len() - 1 - lag]);
                        cool[i] = *t + 30;
                    }
                }
            }
            let a: Vec<usize> = (0..2)
                .map(|i| {
                    if rng.random_bool(0.05) { rng.random_range(0..N_ACTIONS) } else { acts[k * 2 + i] }
                })
                .collect();
            let res = env.step(state, &a);
            (*state, *o) = (res.state, res.obs);
            if res.is_terminal || res.truncated {
                let (s2, o2, _) = env.reset();
                (*state, *o) = (s2, o2);
                hist.clear();
                *cool = [0; 2];
                *t = 0;
            }
        }
    }
    out
}

/// Self-play `ckpt` over the setter mix and record aerial states near the
/// ball from kickoff/random episodes (see [`aerial_spot`]) for the `replay`
/// setter. Writes raw little-endian f32 records of `SNAPSHOT_LEN`.
fn snapshots(setters_toml: &Path, ckpt: &Path, steps: usize, threads: usize, out: &Path) {
    #[derive(Deserialize)]
    struct Mix {
        state_setters: Vec<WeightedSetter>,
    }
    init_from_default(true).expect("load collision meshes");
    let mix: Mix = toml::from_str(&std::fs::read_to_string(setters_toml).expect("read setters"))
        .expect("parse setters");
    let per = steps.div_ceil(threads);
    let all: Vec<f32> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let m = &mix.state_setters;
                s.spawn(move || snapshots_worker(m, ckpt, per, 1000 + t as u64))
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().expect("worker")).collect()
    });
    let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(out, bytes).expect("write snapshots");
    println!(
        "{{\"snapshots\":{},\"steps\":{steps},\"out\":\"{}\"}}",
        all.len() / indococo_bot::setters::SNAPSHOT_LEN,
        out.display()
    );
}

/// Kickoff games of `a` against `b` in this process's team mode
/// (`INDOCOCO_TEAM_SIZE`), greedy with 3% random actions; `a` plays blue in
/// even envs and orange in odd ones. A game ends on a goal or after 300 s
/// (draw). Also counts flip resets and their goalward uses per side (all cars).
/// Prints one JSON line: wins, losses, draws, win_rate_decisive, resets.
fn h2h(a: &Path, b: &Path, games: usize, threads: usize) {
    init_from_default(true).expect("load collision meshes");
    let mix = vec![WeightedSetter {
        weight: 1.0,
        kind: indococo_bot::setters::SetterKind::Kickoff,
    }];
    let per = games.div_ceil(threads);
    let res: Vec<[u64; 7]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let mix = &mix;
                s.spawn(move || h2h_worker(mix, a, b, per, 7000 + t as u64))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = [0u64; 7];
    for r in &res {
        for i in 0..7 {
            tot[i] += r[i];
        }
    }
    let [w, l, d, ra, rb, ua, ub] = tot;
    let out = serde_json::json!({
        "a": a, "b": b, "team_size": indococo_bot::obs::team_size(),
        "games": w + l + d, "wins": w, "losses": l, "draws": d,
        "win_rate_decisive": if w + l > 0 { w as f64 / (w + l) as f64 } else { 0.5 },
        "resets_a": ra, "resets_b": rb, "uses_a": ua, "uses_b": ub,
    });
    println!("{out}");
}

fn h2h_worker(mix: &[WeightedSetter], a: &Path, b: &Path, games: usize, seed: u64) -> [u64; 7] {
    const ENVS: usize = 8;
    let dev = Device::flex();
    let (na, nb) = (load_actor(a, &dev), load_actor(b, &dev));
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = [0u64; 7];
    let mut started = 0usize;
    struct G {
        env: BotEnv,
        state: GameState,
        obs: Vec<Vec<f32>>,
        prev: StateView,
        a_team: u8,
        uses: FlipResetUse,
        live: bool,
    }
    let mut new_game = |k: usize, started: &mut usize| -> G {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs, _) = env.reset();
        let prev = StateView::from_game_state(&state);
        let mut uses = FlipResetUse::new(FlipResetUseParams {
            obtain: 0.0,
            shot_bonus: 0.0,
            goal_bonus: 0.0,
            ..FlipResetUseParams::default()
        });
        uses.reset(&prev);
        let live = *started < games;
        *started += 1;
        G { env, state, obs, prev, a_team: (k % 2) as u8, uses, live }
    };
    let mut gs: Vec<G> = (0..ENVS).map(|k| new_game(k, &mut started)).collect();
    while gs.iter().any(|g| g.live) {
        let mut oa = Vec::new();
        let mut ob = Vec::new();
        for g in gs.iter().filter(|g| g.live) {
            for (i, c) in g.prev.cars.iter().enumerate() {
                if c.team == g.a_team { oa.push(&g.obs[i]); } else { ob.push(&g.obs[i]); }
            }
        }
        let (aa, ab) = (argmax_actions(&na, &oa, &dev), argmax_actions(&nb, &ob, &dev));
        let (mut ia, mut ib) = (0, 0);
        for k in 0..ENVS {
            if !gs[k].live {
                continue;
            }
            let g = &mut gs[k];
            let acts: Vec<usize> = g.prev.cars.iter().map(|c| {
                let x = if c.team == g.a_team { ia += 1; aa[ia - 1] } else { ib += 1; ab[ib - 1] };
                if rng.random_bool(RATES_RANDOM_ACTION) { rng.random_range(0..N_ACTIONS) } else { x }
            }).collect();
            let res = g.env.step(&g.state, &acts);
            let view = g.env.get_shared_info().view(&res.state);
            let used = g.uses.get_rewards(&view);
            for (i, c) in view.cars.iter().enumerate() {
                let side = usize::from(c.team != g.a_team);
                if resets::is_flip_reset(&g.prev.cars[i], c, &view.ball) {
                    out[3 + side] += 1;
                }
                if used[i] > 0.0 {
                    out[5 + side] += 1;
                }
            }
            let done = res.is_terminal || res.truncated;
            if done {
                match view.scoring_team() {
                    Some(t) if t == g.a_team => out[0] += 1,
                    Some(_) => out[1] += 1,
                    None => out[2] += 1,
                }
                *g = new_game(k, &mut started);
            } else {
                (g.state, g.obs, g.prev) = (res.state, res.obs, view);
            }
        }
    }
    out
}
