#![recursion_limit = "256"]
//! indococo 1v1 trainer.
//!
//! ```text
//! indococo-train train  <train.toml>                  # train (resumes from checkpoints)
//! indococo-train import <1v1.pt> <checkpoints_dir>    # warm start from the Python policy
//! indococo-train export <checkpoint_dir> <out.safetensors>
//! indococo-train record <reward.toml> <out.jsonl> <steps> [checkpoint_dir]
//! indococo-train rates  <setters.toml> <candidate_ckpt> <opponent_ckpt|self> <steps> [threads]
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
use indococo_bot::obs::OBS_SIZE;
use indococo_bot::resets;
use indococo_bot::rewards::RewardConfig;
use indococo_bot::rewards::StepReward;
use indococo_bot::rewards::flip_reset_use::{FlipResetUse, FlipResetUseParams};
use indococo_bot::setters::{WeightedSetter, default_mix};
use indococo_bot::state::StateView;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::init_from_default;
use rlgymppo::{
    Actic, ComputePrecision, DecayShape, FixedOpponent, LearnerConfig, LrSchedule, Net,
    PpoLearnerConfig, SelfPlayConfig, SkillTrackerConfig, TrainPrecision, default_adamw_optimizer,
};
use serde::{Deserialize, Serialize};

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
}

impl Default for OpponentsConfig {
    fn default() -> Self {
        Self {
            chance: 0.15,
            past_versions_weight: 1.0,
            fixed: Vec::new(),
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
            model.actor = model
                .actor
                .try_load_file(&path)
                .unwrap_or_else(|e| panic!("opponent {}: {}: {e:?}", o.name, path.display()));
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
    for phase in plan_phases(&cfg, checkpoint_steps(&cfg.checkpoints)) {
        println!(
            "PHASE policy_lr={} iterations={:?}",
            phase.policy_lr, phase.iterations
        );
        let (rewards, setters) = (rewards.clone(), cfg.state_setters.clone());
        let mut learner = learner_config(&cfg, phase, dev.clone(), fixed.clone()).init(
            move |_| create_env(&rewards, &setters),
            default_adamw_optimizer(),
        );
        learner.load();
        learner.learn();
    }
}

/// Fixed probe input used to compare logits across PyTorch and Burn.
fn probe_input() -> Vec<f32> {
    (0..OBS_SIZE)
        .map(|i| ((i as f32) * 0.37).sin() * 0.8)
        .collect()
}

fn new_model(dev: &Device) -> Actic {
    Actic::new(
        OBS_SIZE,
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
    let x = Tensor::<2>::from_data(TensorData::new(probe_input(), [1, OBS_SIZE]), dev);
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
    let mut store = SafetensorsStore::from_file(out).with_to_adapter(BurnToPyTorchAdapter);
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
                    let x = Tensor::<2>::from_data(TensorData::new(o.clone(), [1, OBS_SIZE]), &dev);
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
        })
    }
}

/// Steps (3 s) during which a reset counts as early in its episode.
const EARLY_STEPS: u64 = 45;

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
    new_model(dev)
        .actor
        .try_load_file(ckpt.join("actor.bpk"))
        .unwrap_or_else(|e| panic!("load actor {}: {e:?}", ckpt.display()))
}

/// Greedy actions of `actor` for a batch of observations.
fn argmax_actions(actor: &Net, obs: &[&Vec<f32>], dev: &Device) -> Vec<usize> {
    let flat: Vec<f32> = obs.iter().flat_map(|o| o.iter().copied()).collect();
    let x = Tensor::<2>::from_data(TensorData::new(flat, [obs.len(), OBS_SIZE]), dev);
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
    for _ in 0..steps.div_ceil(ENVS) {
        let blue: Vec<&Vec<f32>> = slots.iter().map(|s| &s.obs[0]).collect();
        let orange: Vec<&Vec<f32>> = slots.iter().map(|s| &s.obs[1]).collect();
        let a0 = argmax_actions(&cand, &blue, &dev);
        let a1 = argmax_actions(opp.as_ref().unwrap_or(&cand), &orange, &dev);
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.step(&[a0[i], a1[i]], &mut counts);
        }
    }
    counts
}

/// Plays `steps` steps (greedy policies, candidate on blue) over the setter
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
        "import" => import(Path::new(arg(1)), Path::new(arg(2))),
        "export" => export(Path::new(arg(1)), Path::new(arg(2))),
        "record" => record(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("steps"),
            args.get(4).map(Path::new),
        ),
        "rates" => rates(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3),
            arg(4).parse().expect("steps"),
            args.get(5).map_or(8, |t| t.parse().expect("threads")),
        ),
        other => panic!("unknown mode {other:?}"),
    }
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
