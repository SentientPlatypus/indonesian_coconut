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
//! indococo-train ripple-h2h <ckpt> <ripple_dir> <games> [threads]  # Rust Ripple h2h (CPU)
//! indococo-train ripple-dump <ripple_dir> <steps> <out.jsonl>  # Dump Ripple tensors for Python parity
//! indococo-train nexto-h2h <ckpt> <nexto_dir> <games> [threads] [dump.jsonl]  # Nexto (TorchScript), team size from INDOCOCO_TEAM_SIZE
//! indococo-train necto-h2h <ckpt> <necto_dir> <games> [threads] [dump.jsonl]  # Necto (TorchScript), same
//! indococo-train learn-replay <train.toml> <ckpt_dir> <memory_dump> <out_dir> [seed]
//! indococo-train nf-bench <nexto_dir> <team_size 2|3> <rows> <iters> <replicas> <intra_threads> <chunk_rows> <parallel_callers>
//! H2H_REWARD_CONFIG=<reward toml> selects the role params.
//! INDOCOCO_OBS_TEAM=1 indococo-train widen-team <1v1_ckpt> <dst_ckpt>  # TM1 warm start (actor+critic)
//! INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1 indococo-train widen-pred <235_ckpt> <dst_ckpt>  # TM3 warm start
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
    /// Gather rollout N+1 with the pre-update policy while rollout N is
    /// trained. Off keeps the sequential loop.
    #[serde(default)]
    async_collect: bool,
    /// Where Galactus and Megabyte run. `default` is the historical
    /// placement (both on CUDA, the calling thread's stream). `cuda` is
    /// that placement plus a side stream and TF32 off for their forward.
    /// `cpu` forces both onto the CPU.
    #[serde(default)]
    league_bots_device: LeagueBotsDevice,
    /// High-priority CUDA stream for collection when `async_collect` is on.
    /// Default false keeps today's low-priority pool stream.
    #[serde(default)]
    collect_stream_priority: bool,
    /// Process-global LibTorch intra-op threads. 0 = do not call set_num_threads.
    #[serde(default)]
    nf_intra_threads: usize,
    /// ST35: prioritized fictitious self-play over the league (replaces the
    /// static `[opponents]` weights for every rollout pick). See
    /// `indococo_bot::pfsp`.
    #[serde(default)]
    pfsp: Option<indococo_bot::pfsp::PfspConfig>,
    /// TM1 team training (2v2 / 3v3): mode mix and obs slot shuffle.
    #[serde(default)]
    team: Option<TeamTrainConfig>,
    /// TM12 critic-only extra inputs (default empty = off).
    #[serde(default)]
    critic: CriticTrainConfig,
}

/// `[critic]` of the train config (TM12). Default empty = off.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CriticTrainConfig {
    extra_inputs: Vec<String>,
}

/// `[team]` of the train config.
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TeamTrainConfig {
    /// Share of collected EXPERIENCE per cars-per-team, e.g.
    /// `{ "1" = 0.25, "2" = 0.40, "3" = 0.35 }` (arenas are split by
    /// share / cars, see `env::set_mode_mix`). Empty = INDOCOCO_TEAM_SIZE for
    /// every arena. Needs the team-ext obs (INDOCOCO_OBS_TEAM=1) when it
    /// mixes sizes.
    mode_mix: BTreeMap<String, f64>,
    /// Per-episode random slot order within allies and opponents.
    obs_shuffle: bool,
}

impl Default for TeamTrainConfig {
    fn default() -> Self {
        Self {
            mode_mix: BTreeMap::new(),
            obs_shuffle: true,
        }
    }
}

/// Device for the Galactus and Megabyte forwards.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum LeagueBotsDevice {
    #[default]
    Default,
    Cpu,
    Cuda,
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

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalOpponentConfig {
    name: String,
    kind: String,
    path: PathBuf,
    weight: f32,
    /// Nexto / Necto only: share of episodes in which the bot samples
    /// (beta 0.5) instead of playing greedily. Default 0.
    #[serde(default)]
    stochastic: f32,
    /// Nexto / Necto only: play the scripted speedflip kickoff (same as
    /// the eval harness). Default false so existing rollouts stay identical.
    #[serde(default)]
    kickoff_script: bool,
    /// Nexto / Necto: copies of the TorchScript module. Default 1 (today).
    #[serde(default = "default_nf_replicas")]
    replicas: usize,
    /// Nexto / Necto: infer chunk size. 0 = off (today).
    #[serde(default)]
    chunk_rows: usize,
}

fn default_nf_replicas() -> usize {
    1
}

/// `RIPPLE_WEIGHT` overrides a toml `kind = "ripple"` entry. A positive value
/// with no entry adds one at `RIPPLE_DIR` (or the v1.1 default). Zero removes it.
fn ripple_external_specs(cfg: &OpponentsConfig) -> Vec<ExternalOpponentConfig> {
    let env_weight = indococo_bot::ripple::ripple_weight_override();
    let mut found = false;
    let mut specs = Vec::with_capacity(cfg.external.len());
    for o in &cfg.external {
        if o.kind != "ripple" {
            specs.push(o.clone());
            continue;
        }
        found = true;
        if let Some(weight) = indococo_bot::ripple::merged_ripple_weight(Some(o.weight), env_weight)
        {
            let mut spec = o.clone();
            spec.weight = weight;
            specs.push(spec);
        }
    }
    if !found {
        if let Some(weight) = indococo_bot::ripple::merged_ripple_weight(None, env_weight) {
            let path = std::env::var("RIPPLE_DIR")
                .unwrap_or_else(|_| "/home/ubuntu/eval_bots/ripple/bots/v1.1".to_string());
            specs.push(ExternalOpponentConfig {
                name: "ripple".into(),
                kind: "ripple".into(),
                path: PathBuf::from(path),
                weight,
                stochastic: 0.0,
                kickoff_script: false,
                replicas: 1,
                chunk_rows: 0,
            });
        }
    }
    specs
}

/// Loads the external opponents. `league` selects the Galactus and
/// Megabyte device. Ripple keeps its own `RIPPLE_DEVICE` selection.
fn load_external_opponents(
    cfg: &OpponentsConfig,
    dev: &Device,
    league: LeagueBotsDevice,
) -> Vec<ExternalOpponent> {
    ripple_external_specs(cfg)
        .iter()
        .map(|o| {
            let policy: std::sync::Arc<dyn rlgymppo::ExternalPolicy> = match o.kind.as_str() {
                "galactus" => {
                    let gdev = league_galactus_device(league, dev);
                    let mut policy = indococo_bot::galactus::GalactusPolicy::load(&o.path, &gdev)
                        .unwrap_or_else(|e| panic!("opponent {}: {e}", o.name));
                    policy.side_stream = league == LeagueBotsDevice::Cuda;
                    std::sync::Arc::new(policy)
                }
                "megabyte" => {
                    let tch_dev = match league {
                        LeagueBotsDevice::Cpu => tch::Device::Cpu,
                        LeagueBotsDevice::Default | LeagueBotsDevice::Cuda => tch::Device::Cuda(0),
                    };
                    let mut policy = indococo_bot::megabyte::MegabytePolicy::load(&o.path, tch_dev)
                        .unwrap_or_else(|e| panic!("opponent {}: {e}", o.name));
                    policy.side_stream = league == LeagueBotsDevice::Cuda;
                    std::sync::Arc::new(policy)
                }
                "ripple" => std::sync::Arc::new(
                    indococo_bot::ripple::RipplePolicy::load(&o.path)
                        .unwrap_or_else(|e| panic!("opponent {}: {e}", o.name)),
                ),
                // TM4: Nexto / Necto (TorchScript on LibTorch, path = bot dir).
                k @ ("nexto" | "necto") => {
                    let kind = indococo_bot::nectofam::Kind::parse(k).expect("kind");
                    let tch_dev = match league {
                        LeagueBotsDevice::Cpu => tch::Device::Cpu,
                        LeagueBotsDevice::Default | LeagueBotsDevice::Cuda => indococo_bot::nectofam::NfModel::default_device(),
                    };
                    let policy = indococo_bot::nectofam::NfPolicy::load_with(
                        kind,
                        &o.path,
                        tch_dev,
                        o.stochastic,
                        o.replicas.max(1),
                        o.chunk_rows,
                    )
                        .unwrap_or_else(|e| panic!("opponent {}: {e}", o.name))
                        .with_kickoff_script(o.kickoff_script);
                    println!(
                        "LEAGUE {} ({k}): delay {}, stochastic {}, kickoff_script {}, device {:?}, replicas {}, chunk_rows {}",
                        o.name,
                        policy.model.kind.delay(),
                        policy.stochastic,
                        policy.kickoff_script,
                        tch_dev,
                        o.replicas.max(1),
                        o.chunk_rows
                    );
                    std::sync::Arc::new(policy)
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

#[allow(deprecated)]
fn league_galactus_device(league: LeagueBotsDevice, learner: &Device) -> Device {
    match league {
        LeagueBotsDevice::Cpu => Device::libtorch(),
        LeagueBotsDevice::Default | LeagueBotsDevice::Cuda => learner.clone(),
    }
}

fn device(backend: &str) -> Device {
    match backend {
        #[cfg(feature = "torch")]
        #[allow(deprecated)]
        "torch" => Device::libtorch_cuda(0),
        // CPU LibTorch: used by the frozen-copy smoke so it can run while TM10
        // holds the GPU. Production train templates stay on backend = "torch".
        #[cfg(feature = "torch")]
        #[allow(deprecated)]
        "torch_cpu" => Device::libtorch(),
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
        async_collect: cfg.async_collect,
        critic_extra_width: indococo_bot::critic_extras::width(),
        ..Default::default()
    }
}

/// ST35: validates `[pfsp]` against the loaded league and registers the
/// per-rollout opponent-mix hook. Without `[pfsp]` (or `enabled = false`)
/// PFSP stays off and the static weights apply, as before.
fn install_pfsp(cfg: &TrainConfig, fixed: &[FixedOpponent], external: &[ExternalOpponent]) {
    let Some(p) = cfg.pfsp.clone() else { return };
    if p.enabled {
        let names: Vec<&str> = fixed
            .iter()
            .map(|o| o.name.as_str())
            .chain(external.iter().map(|o| o.name.as_str()))
            .collect();
        p.validate(&names).unwrap_or_else(|e| panic!("[pfsp]: {e}"));
        if !cfg.self_play_old_versions && p.past_share > 0.0 {
            println!("PFSP: self_play_old_versions is off, past_share goes to self-play");
        }
        println!(
            "PFSP on: self {:.2}, past {:.2}, league {:.2} over {names:?}; floor {:.3} of league, clamp {:?} of all, pinned {:?}, ema {} games, smooth {}",
            p.self_share, p.past_share, p.league_share(), p.floor, p.clamp, p.pinned, p.ema_games, p.smooth
        );
    }
    indococo_bot::pfsp::install(p);
    rlgymppo::set_opponent_mix_hook(pfsp_mix_hook);
}

fn pfsp_mix_hook(names: &[&str], num_versions: usize) -> Option<rlgymppo::OpponentMix> {
    indococo_bot::pfsp::pick_mix(names, num_versions).map(|m| rlgymppo::OpponentMix {
        chance: m.chance as f32,
        past_weight: m.past as f32,
        weights: m.league.iter().map(|&w| w as f32).collect(),
    })
}

fn train(config: &Path) {
    let text = std::fs::read_to_string(config).expect("read train config");
    let cfg: TrainConfig = toml::from_str(&text).expect("parse train config");
    train_cfg(cfg);
}

fn print_tf32_status() {
    #[cfg(feature = "torch")]
    {
        let cublas = indococo_bot::tf32_cublas();
        let cudnn = indococo_bot::tf32_cudnn();
        let override_flag = std::env::var("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE").ok();
        println!(
            "TF32 cublas={cublas} cudnn={cudnn} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={override_flag:?}"
        );
        if cublas == 0 && override_flag.as_deref() == Some("1") {
            eprintln!(
                "WARNING: TF32 cublas is OFF despite TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1; not changing it"
            );
        }
    }
}

fn train_cfg(cfg: TrainConfig) {
    print_tf32_status();
    if cfg.nf_intra_threads > 0 {
        tch::set_num_threads(cfg.nf_intra_threads as i32);
        println!(
            "nf_intra_threads={} tch::get_num_threads()={}",
            cfg.nf_intra_threads,
            tch::get_num_threads()
        );
    }
    if cfg.async_collect && cfg.collect_stream_priority {
        rlgymppo::set_collect_stream_hook(indococo_bot::install_league_collect_stream_high);
    } else if cfg.async_collect {
        rlgymppo::set_collect_stream_hook(indococo_bot::install_league_collect_stream);
    } else {
        rlgymppo::clear_collect_stream_hook();
    }
    // ST33: setters may condition on the rollout's opponent (opp_weights).
    rlgymppo::set_opponent_pick_hook(indococo_bot::setters::set_current_opponent);
    let rewards = read_rewards(&cfg.reward_config);
    let role_params = rewards.team.roles.clone().unwrap_or_default();
    indococo_bot::critic_extras::install(&cfg.critic.extra_inputs, role_params);
    init_from_default(true).expect("load collision meshes");
    let dev = device(&cfg.backend);
    let fixed = load_fixed_opponents(&cfg.opponents, &dev);
    let external = load_external_opponents(&cfg.opponents, &dev, cfg.league_bots_device);
    install_pfsp(&cfg, &fixed, &external);
    // TM1: rewards read the learner step (team-spirit schedule).
    indococo_bot::team::set_step_source(rlgymppo::cumulative_steps);
    if let Some(t) = &cfg.team {
        indococo_bot::obs::set_obs_shuffle(t.obs_shuffle && indococo_bot::obs::obs_team());
        if !t.mode_mix.is_empty() {
            let mix: Vec<(usize, f64)> = t
                .mode_mix
                .iter()
                .map(|(k, &p)| (k.parse().unwrap_or_else(|_| panic!("[team] mode_mix key {k:?}")), p))
                .filter(|&(_, p)| p > 0.0)
                .collect();
            let sizes: std::collections::BTreeSet<usize> = mix.iter().map(|m| m.0).collect();
            assert!(
                sizes.len() <= 1 || indococo_bot::obs::obs_team(),
                "[team] mode_mix mixes team sizes: needs INDOCOCO_OBS_TEAM=1 (fixed-width obs)"
            );
            indococo_bot::env::set_mode_mix(&mix);
            println!("TEAM mode_mix (experience share) {mix:?}, obs_team {}, obs_pred {}, obs width {}, shuffle {}", indococo_bot::obs::obs_team(), indococo_bot::obs::obs_pred(), obs_size(), indococo_bot::obs::obs_shuffle());
        } else {
            println!("TEAM team_size {}, obs_team {}, shuffle {}", indococo_bot::obs::team_size(), indococo_bot::obs::obs_team(), indococo_bot::obs::obs_shuffle());
        }
    }
    for phase in plan_phases(&cfg, checkpoint_steps(&cfg.checkpoints)) {
        println!(
            "PHASE policy_lr={} iterations={:?}",
            phase.policy_lr, phase.iterations
        );
        let (rewards, setters) = (rewards.clone(), cfg.state_setters.clone());
        let mut learner = learner_config(&cfg, phase, dev.clone(), fixed.clone(), external.clone()).init(
            move |id| indococo_bot::env::create_env_for_game(&rewards, &setters, id),
            default_adamw_optimizer(),
        );
        learner.load();
        learner.learn();
    }
    if indococo_bot::obs::obs_pred() {
        let (n, r, ns) = indococo_bot::obs::pred_stats();
        println!(
            "PRED_STATS updates {n} rebuilds {r} ({:.2}%) mean {:.1} us/update",
            100.0 * r as f64 / n.max(1) as f64,
            ns as f64 / 1e3 / n.max(1) as f64
        );
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
    /// Demos performed by the candidate (demos-for).
    demos: u64,
    /// Times the candidate got demoed (demos-against).
    demoed: u64,
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
        self.demos += o.demos;
        self.demoed += o.demoed;
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
            "demos_per_min": self.per_min(self.demos),
            "demos_against_per_min": self.per_min(self.demoed),
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
        let events = StepEvents::between(&self.prev, &view);
        c.resets_old_detector += u64::from(events[0].flip_reset_legacy);
        c.uses += u64::from(self.uses.get_rewards(&view)[0] > 0.0);
        c.goals_after_reset += u64::from(self.goals.get_rewards(&view)[0] > 0.0);
        c.demos += u64::from(events[0].demo);
        c.demoed += u64::from(events[0].demoed);
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
    // TM3: a 235-wide team-ext net in a prediction process (TM1 / TM2
    // finals, BEST-team, warm starts): zero rows for the 35 prediction
    // inputs appended after its columns.
    if indococo_bot::obs::obs_pred() && width == indococo_bot::obs::TEAM_OBS_SIZE {
        let legacy = pick(model_with_input(width, dev)).try_load_file(path).map_err(|e| format!("{e:?}"))?;
        let mut st = SafetensorsStore::from_bytes(None);
        legacy.save_into(&mut st).map_err(|e| format!("{e:?}"))?;
        let bytes = widen_safetensors(&st.get_bytes().map_err(|e| format!("{e:?}"))?, width, obs_size());
        let mut st = SafetensorsStore::from_bytes(Some(bytes));
        let mut net = pick(new_model(dev));
        net.load_from(&mut st).map_err(|e| format!("{e:?}"))?;
        return Ok(net);
    }
    // TM1: a 106-wide 1v1 net in a team-ext process (league opponents,
    // warm starts): columns mapped onto the team layout, zero elsewhere.
    let one = indococo_bot::obs::ext_obs_size_for(1);
    if indococo_bot::obs::obs_team() && width == one {
        let legacy = pick(model_with_input(one, dev)).try_load_file(path).map_err(|e| format!("{e:?}"))?;
        let mut st = SafetensorsStore::from_bytes(None);
        legacy.save_into(&mut st).map_err(|e| format!("{e:?}"))?;
        let bytes = remap_safetensors(
            &st.get_bytes().map_err(|e| format!("{e:?}"))?,
            one,
            obs_size(),
            indococo_bot::obs::team_col_of_1v1,
        );
        let mut st = SafetensorsStore::from_bytes(Some(bytes));
        let mut net = pick(new_model(dev));
        net.load_from(&mut st).map_err(|e| format!("{e:?}"))?;
        return Ok(net);
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

/// Rewrites `linear_layers.0.weight` (`[from, out]`, f32 row-major) as
/// `[to, out]`: input row `k` moves to row `map(k)`, all other rows zero.
fn remap_safetensors(bytes: &[u8], from: usize, to: usize, map: fn(usize) -> usize) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("header len")) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).expect("safetensors header");
    let data = &bytes[8 + n..];
    let mut keys: Vec<(String, serde_json::Value)> = header.into_iter().collect();
    keys.sort_by_key(|(k, v)| if k == "__metadata__" { 0 } else { v["data_offsets"][0].as_u64().unwrap_or(0) + 1 });
    let mut out_header = serde_json::Map::new();
    let mut out_data = Vec::with_capacity(data.len() + to * 4 * 2048);
    for (k, mut v) in keys {
        if k == "__metadata__" {
            out_header.insert(k, v);
            continue;
        }
        let (a, b) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
        let start = out_data.len();
        if k == "linear_layers.0.weight" {
            let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            assert_eq!(shape[0], from, "{k} shape {shape:?}");
            assert_eq!(v["dtype"], "F32");
            let row = shape[1] * 4;
            let src = &data[a..b];
            let mut w = vec![0u8; to * row];
            for r in 0..from {
                let d = map(r);
                assert!(d < to, "remap row {r} -> {d} out of {to}");
                w[d * row..(d + 1) * row].copy_from_slice(&src[r * row..(r + 1) * row]);
            }
            out_data.extend_from_slice(&w);
            v["shape"] = serde_json::json!([to, shape[1]]);
        } else {
            out_data.extend_from_slice(&data[a..b]);
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

/// `widen-team <src_ck> <dst_ck>` (INDOCOCO_OBS_TEAM=1): a team-ext
/// checkpoint from a 106-wide 1v1 one. Actor AND critic first layers are
/// remapped ([`indococo_bot::obs::team_col_of_1v1`]), new columns zero, so
/// the critic keeps its 1v1 values (still run the critic warm-up: team
/// returns differ). stats.toml copied. Verifies on 1v1 states that the
/// widened nets on team-ext obs equal the source nets on the 1v1 obs, and
/// prints the logit change on 2v2/3v3 states (teammates are unseen at step 0).
fn widen_team(src: &Path, dst: &Path) {
    assert!(indococo_bot::obs::obs_team(), "widen-team needs INDOCOCO_OBS_TEAM=1");
    init_from_default(true).expect("load collision meshes");
    let dev = Device::flex();
    std::fs::create_dir_all(dst).expect("create dst");
    let one = indococo_bot::obs::ext_obs_size_for(1);
    let actor = load_net_any(&src.join("actor.bpk"), NetKind::Actor, &dev).expect("actor");
    let critic = load_net_any(&src.join("critic.bpk"), NetKind::Critic, &dev).expect("critic");
    actor.clone().save_file(dst.join("actor.bpk")).expect("save actor");
    critic.clone().save_file(dst.join("critic.bpk")).expect("save critic");
    std::fs::copy(src.join("stats.toml"), dst.join("stats.toml")).expect("copy stats");
    let base_actor = model_with_input(one, &dev).actor.try_load_file(src.join("actor.bpk")).expect("src actor");
    let base_critic = model_with_input(one, &dev).critic.try_load_file(src.join("critic.bpk")).expect("src critic");
    let re_actor = load_net_any(&dst.join("actor.bpk"), NetKind::Actor, &dev).expect("reload actor");
    let mut report = serde_json::Map::new();
    for per_team in 1..=3usize {
        let mut env = indococo_bot::env::create_env_sized(&RewardConfig::default(), &default_mix(), per_team);
        let mut rng = SmallRng::seed_from_u64(11 + per_team as u64);
        let (mut state, mut obs, _) = env.reset();
        let (mut team_rows, mut one_rows) = (Vec::new(), Vec::new());
        for _ in 0..256 {
            let order = indococo_bot::obs::pad_order(&state);
            for (i, o) in obs.iter().enumerate() {
                team_rows.extend_from_slice(o);
                // The 1v1 obs of the same car: own car + FIRST opponent only.
                let mut b = indococo_bot::obs::build_head(&state, i, &order);
                let t = state.cars[i].0.team;
                let opp = (0..state.cars.len()).find(|&j| state.cars[j].0.team != t).expect("opponent");
                let mut two = state.clone();
                two.cars = vec![state.cars[i].clone(), state.cars[opp].clone()];
                let full = indococo_bot::obs::build_one(&two, 0, &order);
                b.extend_from_slice(&full[72..92]);
                let e = indococo_bot::obs::ext_features(&b, indococo_bot::obs::prev_action(&state.cars[i].1));
                b.extend(e);
                one_rows.extend(b);
            }
            let actions: Vec<usize> = obs.iter().map(|_| rng.random_range(0..N_ACTIONS)).collect();
            let res = env.step(&state, &actions);
            (state, obs) = (res.state, res.obs);
            if res.is_terminal || res.truncated {
                let (s, o, _) = env.reset();
                (state, obs) = (s, o);
            }
        }
        let n = team_rows.len() / obs_size();
        let xt = Tensor::<2>::from_data(TensorData::new(team_rows, [n, obs_size()]), &dev);
        let xo = Tensor::<2>::from_data(TensorData::new(one_rows, [n, one]), &dev);
        let maxdiff = |a: Tensor<2>, b: Tensor<2>| -> f32 {
            (a - b).abs().max().into_data().try_to_vec::<f32>().expect("diff")[0]
        };
        report.insert(
            format!("{per_team}v{per_team}"),
            serde_json::json!({
                "rows": n,
                "actor_max_abs_logit_diff": maxdiff(actor.forward(xt.clone()), base_actor.forward(xo.clone())),
                "reloaded_actor_max_abs_logit_diff": maxdiff(re_actor.forward(xt.clone()), base_actor.forward(xo.clone())),
                "critic_max_abs_value_diff": maxdiff(critic.forward(xt), base_critic.forward(xo)),
            }),
        );
    }
    report.insert("dst".into(), serde_json::json!(dst));
    println!("WIDEN_TEAM {}", serde_json::Value::Object(report));
}

/// `widen-pred <src_ck> <dst_ck>` (INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1):
/// a TM3 checkpoint (270 inputs) from a 235-wide team-ext one. Actor AND
/// critic first layers get 35 zero rows at the end (the prediction columns),
/// every other tensor is copied, stats.toml copied. A source that is already
/// 270-wide is copied as is. Verifies on 1v1 / 2v2 / 3v3 env obs that the
/// first-layer weights are the source's bit for bit with exact zeros after
/// them, and reports the max |logit| / |value| difference between the widened
/// nets on the 270 obs and the source nets on its first 235 columns (zero
/// rows only change the float summation order).
fn widen_pred(src: &Path, dst: &Path) {
    use indococo_bot::obs::{PRED_OBS_SIZE, TEAM_OBS_SIZE, obs_pred, obs_team};
    assert!(obs_team() && obs_pred(), "widen-pred needs INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1");
    assert_eq!(obs_size(), PRED_OBS_SIZE);
    init_from_default(true).expect("load collision meshes");
    let dev = Device::flex();
    std::fs::create_dir_all(dst).expect("create dst");
    let width = bpk_input_width(&src.join("actor.bpk")).expect("actor width");
    let cwidth = bpk_input_width(&src.join("critic.bpk")).expect("critic width");
    assert_eq!(width, cwidth, "actor / critic input widths differ");
    if width == PRED_OBS_SIZE {
        for f in ["actor.bpk", "critic.bpk", "stats.toml"] {
            std::fs::copy(src.join(f), dst.join(f)).expect("copy");
        }
        println!("WIDEN_PRED {}", serde_json::json!({"already": PRED_OBS_SIZE, "dst": dst}));
        return;
    }
    assert_eq!(width, TEAM_OBS_SIZE, "widen-pred needs a {TEAM_OBS_SIZE}-wide source, got {width}");
    let actor = load_net_any(&src.join("actor.bpk"), NetKind::Actor, &dev).expect("actor");
    let critic = load_net_any(&src.join("critic.bpk"), NetKind::Critic, &dev).expect("critic");
    actor.clone().save_file(dst.join("actor.bpk")).expect("save actor");
    critic.clone().save_file(dst.join("critic.bpk")).expect("save critic");
    std::fs::copy(src.join("stats.toml"), dst.join("stats.toml")).expect("copy stats");
    let base_actor = model_with_input(width, &dev).actor.try_load_file(src.join("actor.bpk")).expect("src actor");
    let base_critic = model_with_input(width, &dev).critic.try_load_file(src.join("critic.bpk")).expect("src critic");
    let re_actor = load_net_any(&dst.join("actor.bpk"), NetKind::Actor, &dev).expect("reload actor");
    let re_critic = load_net_any(&dst.join("critic.bpk"), NetKind::Critic, &dev).expect("reload critic");
    let mut report = serde_json::Map::new();
    report.insert("actor_first_layer_exact".into(), serde_json::json!(first_layer_extends(&base_actor, &re_actor, width)));
    report.insert("critic_first_layer_exact".into(), serde_json::json!(first_layer_extends(&base_critic, &re_critic, width)));
    for per_team in 1..=3usize {
        let mut env = indococo_bot::env::create_env_sized(&RewardConfig::default(), &default_mix(), per_team);
        let mut rng = SmallRng::seed_from_u64(31 + per_team as u64);
        let (mut state, mut obs, _) = env.reset();
        let (mut rows, mut base_rows) = (Vec::new(), Vec::new());
        for _ in 0..256 {
            for o in &obs {
                rows.extend_from_slice(o);
                base_rows.extend_from_slice(&o[..width]);
            }
            let actions: Vec<usize> = obs.iter().map(|_| rng.random_range(0..N_ACTIONS)).collect();
            let res = env.step(&state, &actions);
            (state, obs) = (res.state, res.obs);
            if res.is_terminal || res.truncated {
                let (s, o, _) = env.reset();
                (state, obs) = (s, o);
            }
        }
        let n = rows.len() / obs_size();
        let x = Tensor::<2>::from_data(TensorData::new(rows, [n, obs_size()]), &dev);
        let xb = Tensor::<2>::from_data(TensorData::new(base_rows, [n, width]), &dev);
        let maxdiff = |a: Tensor<2>, b: Tensor<2>| -> f32 {
            (a - b).abs().max().into_data().try_to_vec::<f32>().expect("diff")[0]
        };
        report.insert(
            format!("{per_team}v{per_team}"),
            serde_json::json!({
                "rows": n,
                "actor_max_abs_logit_diff": maxdiff(actor.forward(x.clone()), base_actor.forward(xb.clone())),
                "reloaded_actor_max_abs_logit_diff": maxdiff(re_actor.forward(x.clone()), base_actor.forward(xb.clone())),
                "critic_max_abs_value_diff": maxdiff(critic.forward(x), base_critic.forward(xb)),
            }),
        );
    }
    report.insert("dst".into(), serde_json::json!(dst));
    println!("WIDEN_PRED {}", serde_json::Value::Object(report));
}

/// True if `wide`'s first-layer weight (`[in, out]`) is `narrow`'s rows bit
/// for bit followed by exact zero rows, and every other tensor is identical.
fn first_layer_extends(narrow: &Net, wide: &Net, from: usize) -> bool {
    let tensors = |m: &Net| -> BTreeMap<String, (Vec<usize>, Vec<u32>)> {
        let mut st = SafetensorsStore::from_bytes(None);
        m.clone().save_into(&mut st).expect("save");
        let bytes = st.get_bytes().expect("bytes");
        let n = u64::from_le_bytes(bytes[..8].try_into().expect("len")) as usize;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + n]).expect("header");
        let data = &bytes[8 + n..];
        header
            .into_iter()
            .filter(|(k, _)| k != "__metadata__")
            .map(|(k, v)| {
                let (a, b) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
                let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                let bits = data[a..b].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                (k, (shape, bits))
            })
            .collect()
    };
    let (a, b) = (tensors(narrow), tensors(wide));
    a.len() == b.len()
        && a.iter().all(|(k, (sa, va))| {
            let Some((sb, vb)) = b.get(k) else { return false };
            if k == "linear_layers.0.weight" {
                let keep = from * sa[1];
                sa[0] == from && sb[0] > from && sa[1] == sb[1] && vb[..keep] == va[..] && vb[keep..].iter().all(|&x| x == 0)
            } else {
                sa == sb && va == vb
            }
        })
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

/// One PPO update on a dumped rollout, using the same LearnerConfig as
/// `train` (PPO hyperparams, layer sizes, device, checkpoint stats and
/// optimizers). Collection is slimmed so this does not start a 384-game
/// simulator; that does not change the update math.
fn learn_replay(config: &Path, ckpt_dir: &Path, memory_dump: &Path, out_dir: &Path, seed: u64) {
    let text = std::fs::read_to_string(config).expect("read train config");
    let mut cfg: TrainConfig = toml::from_str(&text).expect("parse train config");
    assert!(
        ckpt_dir.join("actor.bpk").is_file(),
        "learn-replay ckpt_dir must contain actor.bpk: {}",
        ckpt_dir.display()
    );
    cfg.checkpoints = ckpt_dir.to_path_buf();
    cfg.opponents = OpponentsConfig::default();
    cfg.pfsp = None;
    cfg.async_collect = false;
    cfg.skill_tracker = false;
    cfg.self_play_old_versions = false;
    cfg.iterations = 0;
    cfg.save_every = u64::MAX / 4;
    // One game per pool is enough to probe obs/action sizes.
    cfg.games = cfg.pools.max(1);
    cfg.threads = cfg.pools.max(1);

    print_tf32_status();
    rlgymppo::clear_collect_stream_hook();
    rlgymppo::set_opponent_pick_hook(indococo_bot::setters::set_current_opponent);
    let rewards = read_rewards(&cfg.reward_config);
    init_from_default(true).expect("load collision meshes");
    let dev = device(&cfg.backend);
    indococo_bot::team::set_step_source(rlgymppo::cumulative_steps);
    if let Some(t) = &cfg.team {
        indococo_bot::obs::set_obs_shuffle(t.obs_shuffle && indococo_bot::obs::obs_team());
        if !t.mode_mix.is_empty() {
            let mix: Vec<(usize, f64)> = t
                .mode_mix
                .iter()
                .map(|(k, &p)| (k.parse().unwrap_or_else(|_| panic!("[team] mode_mix key {k:?}")), p))
                .filter(|&(_, p)| p > 0.0)
                .collect();
            indococo_bot::env::set_mode_mix(&mix);
        }
    }
    let memory = rlgymppo::Memory::load_from_dir(memory_dump)
        .unwrap_or_else(|e| panic!("load memory dump {}: {e}", memory_dump.display()));
    let phase = Phase {
        policy_lr: cfg.policy_lr,
        iterations: Some(1),
    };
    let (rewards, setters) = (rewards, cfg.state_setters.clone());
    let mut learner = learner_config(&cfg, phase, dev, Vec::new(), Vec::new()).init(
        move |id| indococo_bot::env::create_env_for_game(&rewards, &setters, id),
        default_adamw_optimizer(),
    );
    learner.load();
    learner.learn_replay(&memory, seed, out_dir);
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

/// Newest numeric checkpoint directory, if the folder exists and holds one.
fn newest_checkpoint(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().filter_map(|e| {
        let name = e.file_name();
        let text = name.to_str()?;
        text.parse::<u64>().ok().map(|n| (n, e.path()))
    }).max_by_key(|(n, _)| *n).map(|(_, p)| p)
}

/// Four training shapes, plus the CPU league placement, each for `iters`
/// PPO updates. Checkpoints are written only under a fresh temp directory.
/// The newest checkpoint in the toml's `checkpoints` path is copied in
/// (read-only) so the step count and network match that run; a missing
/// folder starts from a fresh init. Prints one `SPEED` line per iteration.
/// Iteration 1 has overlap 0. Later iterations report how much of that
/// rollout's collection ran during the previous update.
/// CPU microbench of Nexto `infer`. `intra_threads = 0` leaves LibTorch's
/// default; the process-wide thread count is always printed. `parallel_callers`
/// threads call `infer` at once (2 mimics the two collector pools).
fn nf_bench(
    nexto_dir: &Path,
    team_size: usize,
    rows: usize,
    iters: usize,
    replicas: usize,
    intra_threads: usize,
    chunk_rows: usize,
    parallel_callers: usize,
) {
    use indococo_bot::env::create_env_sized;
    use indococo_bot::nectofam::{Kind, NfPolicy, ROW};
    use rlgymppo::ExternalPolicy;
    use std::time::Instant;

    assert!(
        team_size == 2 || team_size == 3,
        "team_size must be 2 or 3, got {team_size}"
    );
    assert!(rows > 0 && iters > 0, "rows and iters must be positive");
    init_from_default(true).expect("load collision meshes");
    if intra_threads > 0 {
        tch::set_num_threads(intra_threads as i32);
    }
    println!(
        "nf-bench dir={} team_size={team_size} rows={rows} iters={iters} replicas={} intra_threads={intra_threads} chunk_rows={chunk_rows} callers={} tch::get_num_threads()={}",
        nexto_dir.display(),
        replicas.max(1),
        parallel_callers.max(1),
        tch::get_num_threads()
    );
    let policy = NfPolicy::load_with(
        Kind::Nexto,
        nexto_dir,
        tch::Device::Cpu,
        0.0,
        replicas.max(1),
        chunk_rows,
    )
    .unwrap_or_else(|e| panic!("load nexto: {e}"));

    let mut obs = vec![0.0f32; rows * ROW];
    let mut mask = vec![true; 90];
    let mut env = create_env_sized(&RewardConfig::default(), &default_mix(), team_size);
    let (mut state, _, _) = env.reset();
    let mut rng = SmallRng::seed_from_u64(0x6e58_0b00);
    let mut filled = 0usize;
    while filled < rows {
        for car in 0..state.cars.len() {
            if filled >= rows {
                break;
            }
            let start = filled * ROW;
            policy.build(&state, car, &mut obs[start..start + ROW], &mut mask);
            filled += 1;
        }
        let actions: Vec<usize> = (0..state.cars.len())
            .map(|_| rng.random_range(0..90))
            .collect();
        let res = env.step(&state, &actions);
        state = res.state;
        if res.is_terminal || res.truncated {
            let (s, _, _) = env.reset();
            state = s;
        }
    }
    let dummy_masks = vec![true; rows];
    // Warmup.
    let _ = policy.infer(&obs, &dummy_masks, rows);

    let callers = parallel_callers.max(1);
    let mut times_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..callers {
                scope.spawn(|| {
                    let _ = policy.infer(&obs, &dummy_masks, rows);
                });
            }
        });
        times_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let mean = times_ms.iter().sum::<f64>() / n as f64;
    println!(
        "nf-bench ms_per_round n={n} min={:.3} median={:.3} mean={:.3} max={:.3} callers={callers}",
        sorted[0],
        median,
        mean,
        sorted[n - 1]
    );
    for (i, ms) in times_ms.iter().enumerate() {
        println!("  iter {i} {ms:.3} ms");
    }
}

fn speed_variants() -> [(&'static str, bool, bool, LeagueBotsDevice); 7] {
    [
        ("off", false, false, LeagueBotsDevice::Default),
        ("async", true, false, LeagueBotsDevice::Default),
        ("async_prio", true, true, LeagueBotsDevice::Default),
        ("league_cuda", false, false, LeagueBotsDevice::Cuda),
        ("league_cpu", false, false, LeagueBotsDevice::Cpu),
        ("both", true, false, LeagueBotsDevice::Cuda),
        ("both_cpu", true, false, LeagueBotsDevice::Cpu),
    ]
}

fn parse_speed_variants(spec: Option<&str>) -> Vec<(&'static str, bool, bool, LeagueBotsDevice)> {
    let all = speed_variants();
    let Some(spec) = spec.filter(|s| !s.is_empty()) else {
        return all.to_vec();
    };
    spec.split(',')
        .map(|name| {
            let name = name.trim();
            all.iter()
                .copied()
                .find(|(n, _, _, _)| *n == name)
                .unwrap_or_else(|| panic!("unknown SPEED_VARIANTS entry {name:?}"))
        })
        .collect()
}

fn speed_bench_keep() -> bool {
    matches!(std::env::var("SPEED_BENCH_KEEP"), Ok(ref v) if v == "1")
}


/// Parse a train.toml and exit. Used to dry-run rendered launcher configs
/// without starting a trainer (no GPU, no env, no writes).
fn parse_config(path: &Path) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("parse-config: cannot read {}: {e}", path.display());
        std::process::exit(2);
    });
    let cfg: TrainConfig = toml::from_str(&text).unwrap_or_else(|e| {
        eprintln!("parse-config FAIL {}: {e}", path.display());
        std::process::exit(1);
    });
    println!(
        "parse-config ok path={} nf_intra_threads={} async_collect={} collect_stream_priority={} epochs={} policy_bf16={} critic_bf16={}",
        path.display(),
        cfg.nf_intra_threads,
        cfg.async_collect,
        cfg.collect_stream_priority,
        cfg.epochs,
        cfg.policy_bf16,
        cfg.critic_bf16
    );
}

fn speed_bench(config: &Path, iters: u64) {
    assert!(iters > 0, "speed-bench needs at least one iteration");
    let text = std::fs::read_to_string(config).expect("read train config");
    unsafe { std::env::set_var("INDOCOCO_SPEED", "1"); }
    let variants = parse_speed_variants(std::env::var("SPEED_VARIANTS").ok().as_deref());
    println!(
        "speed-bench iters={iters} variants={} (iteration 1 overlap is 0; later lines are the overlapped collect)",
        variants.iter().map(|(n, _, _, _)| *n).collect::<Vec<_>>().join(",")
    );
    for (name, async_collect, collect_stream_priority, league) in variants {
        let mut cfg: TrainConfig = toml::from_str(&text).expect("parse train config");
        // temp_dir honors TMPDIR, which the scratch layout sets to speedup_work/tmp.
        let dir = std::env::temp_dir().join(format!(
            "indococo-speed-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("temp checkpoint dir");
        let live_checkpoints = cfg.checkpoints.clone();
        cfg.async_collect = async_collect;
        cfg.collect_stream_priority = collect_stream_priority;
        cfg.league_bots_device = league;
        cfg.iterations = iters;
        cfg.critic_warmup_steps = 0;
        cfg.save_every = u64::MAX / 4;
        cfg.checkpoints = dir.clone();
        if let Some(src) = newest_checkpoint(&live_checkpoints) {
            let dst = dir.join(src.file_name().unwrap_or_default());
            match copy_dir(&src, &dst) {
                Ok(()) => println!(
                    "VARIANT {name} resumed a read-only copy of {}",
                    src.display()
                ),
                Err(e) => println!(
                    "VARIANT {name} fresh init (copy {} failed: {e})",
                    src.display()
                ),
            }
        } else {
            println!(
                "VARIANT {name} fresh init (no checkpoint under {})",
                live_checkpoints.display()
            );
        }
        unsafe { std::env::set_var("INDOCOCO_METRICS", dir.join("metrics.jsonl")); }
        println!(
            "VARIANT {name} async_collect={async_collect} collect_stream_priority={collect_stream_priority} league_bots_device={league:?} dir={}",
            dir.display()
        );
        train_cfg(cfg);
        if speed_bench_keep() {
            println!("VARIANT {name} kept {}", dir.display());
        } else {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// `widen-critic-extras <src_ck> <dst_ck>`: copy actor and stats, widen the
/// critic's first layer by the process-wide critic extra width (zero-init
/// columns). If the critic is already wide, copy it. Prints WIDEN_CRITIC
/// and checks that V is identical at step 0 (zero extras vs the source).
fn widen_critic_extras(src: &Path, dst: &Path) {
    let extra = indococo_bot::critic_extras::width();
    // The CLI is usually run before `train` installs extra_inputs. Default
    // to the TM12 set (width 10) when the process-wide config is still empty,
    // unless the caller set CRITIC_EXTRAS=0.
    let extra = if extra == 0 && std::env::var("CRITIC_EXTRAS").ok().as_deref() != Some("0") {
        10
    } else {
        extra
    };
    init_from_default(true).expect("load collision meshes");
    let dev = Device::flex();
    std::fs::create_dir_all(dst).expect("create dst");
    std::fs::copy(src.join("actor.bpk"), dst.join("actor.bpk")).expect("copy actor");
    if src.join("stats.toml") != dst.join("stats.toml") {
        let _ = std::fs::copy(src.join("stats.toml"), dst.join("stats.toml"));
    }
    let from = bpk_input_width(&src.join("critic.bpk")).expect("critic width");
    let to = obs_size() + extra;
    if extra == 0 || from == to {
        if src.join("critic.bpk") != dst.join("critic.bpk") {
            std::fs::copy(src.join("critic.bpk"), dst.join("critic.bpk")).expect("copy critic");
        }
        println!("WIDEN_CRITIC extra={extra} from={from} to={to} copied (no change)");
        return;
    }
    assert_eq!(from, obs_size(), "widen-critic-extras needs a {0}-wide critic, got {from}", obs_size());
    let bytes = std::fs::read(src.join("critic.bpk")).expect("read critic");
    let wide = widen_safetensors(&bytes, from, to);
    std::fs::write(dst.join("critic.bpk"), &wide).expect("write critic");
    // Step-0 identity: source critic on obs equals widened critic on obs||zeros.
    let src_c = model_with_input(from, &dev).critic.try_load_file(src.join("critic.bpk")).expect("src critic");
    let dst_c = model_with_input(to, &dev).critic.try_load_file(dst.join("critic.bpk")).expect("dst critic");
    let n = 8;
    let mut rows = Vec::with_capacity(n * from);
    for i in 0..n * from {
        rows.push(((i as f32) * 0.37).sin() * 0.8);
    }
    let x = Tensor::<2>::from_data(TensorData::new(rows.clone(), [n, from]), &dev);
    let mut wide_rows = Vec::with_capacity(n * to);
    for r in 0..n {
        wide_rows.extend_from_slice(&rows[r * from..(r + 1) * from]);
        wide_rows.extend(std::iter::repeat(0.0).take(extra));
    }
    let xw = Tensor::<2>::from_data(TensorData::new(wide_rows, [n, to]), &dev);
    let a = src_c.forward(x).into_data().try_to_vec::<f32>().expect("src v");
    let b = dst_c.forward(xw).into_data().try_to_vec::<f32>().expect("dst v");
    let maxdiff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
    println!("WIDEN_CRITIC extra={extra} from={from} to={to} step0_max_abs_value_diff={maxdiff}");
    assert!(maxdiff < 1e-5, "widened critic is not identical at step 0: {maxdiff}");
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
        "widen-team" => widen_team(Path::new(arg(1)), Path::new(arg(2))),
        "widen-pred" => widen_pred(Path::new(arg(1)), Path::new(arg(2))),
        "widen-critic-extras" => widen_critic_extras(Path::new(arg(1)), Path::new(arg(2))),
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
        "ripple-h2h" => ripple_h2h(
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("games"),
            args.get(4).map_or(8, |t| t.parse().expect("threads")),
            args.get(5).map(|s| s == "--idle-opponent").unwrap_or(false),
        ),
        "ripple-dump" => ripple_dump(
            Path::new(arg(1)),
            arg(2).parse().expect("steps"),
            Path::new(arg(3)),
        ),
        k @ ("nexto-h2h" | "necto-h2h") => nf_h2h(
            indococo_bot::nectofam::Kind::parse(k.trim_end_matches("-h2h")).expect("kind"),
            Path::new(arg(1)),
            Path::new(arg(2)),
            arg(3).parse().expect("games"),
            args.get(4).map_or(8, |t| t.parse().expect("threads")),
            args.get(5).map(Path::new),
        ),
        "parse-config" => parse_config(Path::new(arg(1))),
        "speed-bench" => speed_bench(Path::new(arg(1)), arg(2).parse().expect("iters")),
        "nf-bench" => nf_bench(
            Path::new(arg(1)),
            arg(2).parse().expect("team_size"),
            arg(3).parse().expect("rows"),
            arg(4).parse().expect("iters"),
            arg(5).parse().expect("replicas"),
            arg(6).parse().expect("intra_threads"),
            arg(7).parse().expect("chunk_rows"),
            arg(8).parse().expect("parallel_callers"),
        ),
        "learn-replay" => learn_replay(
            Path::new(arg(1)),
            Path::new(arg(2)),
            Path::new(arg(3)),
            Path::new(arg(4)),
            args.get(5).map_or(0, |s| s.parse().expect("seed")),
        ),
        "reward-audit" => reward_audit(
            Path::new(arg(1)),
            Path::new(arg(2)),
            Path::new(arg(3)),
            arg(4).parse().expect("steps"),
            args.get(5).map_or(6, |t| t.parse().expect("threads")),
        ),
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
    let mix = vec![WeightedSetter { opp_weights: Default::default(), mode_weights: Default::default(),
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
        "steps": steps, "jitter": h2h_jitter(false), "seconds": t0.elapsed().as_secs_f64(),
    });
    println!("{out}");
}

fn v3(v: rlgymppo::rlgym::rocketsim::Vec3A) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Kickoff spawn jitter for the scripted-opponent harnesses, matching
/// `tools/pool_eval/poolkit/match.py`: every car moves by up to 60 uu in x and
/// y and turns by up to 4 degrees, the ball stays at the center. Without it a
/// greedy learner against a deterministic bot only ever plays one game per
/// kickoff spawn (10 distinct 1v1 games), so a 400-game score is really a
/// score over those few lines. On by default for `nexto-h2h`/`necto-h2h`;
/// `H2H_JITTER=1` turns it on for `galactus-h2h`/`megabyte-h2h` (off there by
/// default so old numbers stay comparable), `H2H_JITTER=0` turns it off.
fn h2h_jitter(default: bool) -> bool {
    std::env::var("H2H_JITTER").map(|v| v != "0").unwrap_or(default)
}

fn jittered_reset(env: &mut BotEnv, rng: Option<&mut SmallRng>) -> (GameState, Vec<Vec<f32>>) {
    use rlgymppo::rlgym::{Obs as _, SharedInfoProvider as _};
    let (state, obs, _) = env.reset();
    // H2H_KICKOFF_FIX=1 (diagnostic, off by default; EXPERIMENTS.md "Eval
    // fidelity"): undo two Rust-port reset artifacts the C++ engine does not
    // have: wheel contacts unknown on the first tick after a state set, and
    // flip_time = 0, which pitch-locks air control for 0.3 s (PITCHLOCK_EXTRA_TIME)
    // although the car has not flipped.
    let kickoff_fix = std::env::var("H2H_KICKOFF_FIX").map(|v| v == "1").unwrap_or(false);
    if rng.is_none() && !kickoff_fix {
        return (state, obs);
    }
    if let Some(rng) = rng {
        const UU: f32 = 60.0;
        let yaw = 4.0f32.to_radians();
        for i in 0..state.cars.len() {
            let mut car = env.arena.get_car_state(i).clone();
            car.phys.pos.x += rng.random_range(-UU..UU);
            car.phys.pos.y += rng.random_range(-UU..UU);
            let a = rng.random_range(-yaw..yaw);
            car.phys.rot_mat = rlgymppo::rlgym::rocketsim::Mat3A::from_rotation_z(a) * car.phys.rot_mat;
            env.arena.set_car_state(i, car.core());
        }
    }
    if kickoff_fix {
        for i in 0..state.cars.len() {
            env.arena.edit_car_state_full(i, |s| s.flip_time = 1.0);
        }
        for i in 0..state.cars.len() {
            env.arena.refresh_car_contacts(i);
        }
    }
    let state = env.current_state();
    env.shared_info.reset(&state);
    env.observations.reset(&state, &mut env.shared_info);
    let obs = env.observations.build_obs(&state, &mut env.shared_info);
    (state, obs)
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
    let mut jrng = h2h_jitter(false).then(|| SmallRng::seed_from_u64(rand::rng().next_u64()));
    let mut new_game = |k: usize, started: &mut usize| -> G {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs) = jittered_reset(&mut env, jrng.as_mut());
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

/// `nexto-h2h` / `necto-h2h`: the candidate checkpoint (greedy, CPU) against
/// Nexto or Necto (their TorchScript nets on LibTorch, CPU, one intra-op
/// thread per worker) in this process's team mode (`INDOCOCO_TEAM_SIZE`, both
/// sides, kickoff starts, first goal wins, 300 s / 30 s no-touch = draw).
/// The bots are replayed tick by tick as their RLBot `bot.py` runs them
/// ([`indococo_bot::nectofam::NfBot`]): own obs builder, decision 1 tick
/// after each 8-tick boundary, controls 6 (Nexto) / 7 (Necto) ticks later,
/// the scripted speedflip for the kickoff taker (`NF_KICKOFF=0` turns it
/// off), Nexto sampling during kickoffs. `NF_STOCHASTIC=<p>`: share of games
/// in which the bot samples (beta 0.5) every decision; default 0 (greedy, as
/// in RLBot). Works with any checkpoint width the process can load
/// (106 / 235 / 270 with the matching INDOCOCO_OBS_* flags). With `dump`,
/// worker 0 writes its first 3000 bot decisions (state, Rust obs, logits)
/// for `tools/nectofam/parity.py`. Kickoff spawns are jittered as in the
/// Python pool eval (see [`h2h_jitter`]; `H2H_JITTER=0` turns it off).
fn nf_h2h(kind: indococo_bot::nectofam::Kind, ckpt: &Path, dir: &Path, games: usize, threads: usize, dump: Option<&Path>) {
    if std::env::var("INDOCOCO_OBS_EXT").is_err() {
        // SAFETY: before any thread is spawned.
        unsafe { std::env::set_var("INDOCOCO_OBS_EXT", "1") };
    }
    tch::set_num_threads(1);
    init_from_default(true).expect("load collision meshes");
    let mix = vec![WeightedSetter { opp_weights: Default::default(), mode_weights: Default::default(), weight: 1.0, kind: indococo_bot::setters::SetterKind::Kickoff }];
    let stochastic: f32 = std::env::var("NF_STOCHASTIC").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let script = std::env::var("NF_KICKOFF").map_or(true, |v| v != "0");
    let per = games.div_ceil(threads);
    let t0 = std::time::Instant::now();
    let res: Vec<[u64; 6]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let mix = &mix;
                let dump = if t == 0 { dump } else { None };
                s.spawn(move || nf_worker(kind, mix, ckpt, dir, per, 9100 + t as u64, stochastic, script, dump))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = [0u64; 6];
    for r in &res {
        for i in 0..6 {
            tot[i] += r[i];
        }
    }
    let [w, l, d, steps, decisions, scripted] = tot;
    let out = serde_json::json!({
        "a": ckpt, "b": format!("{}(torchscript)", kind.name()), "team_size": indococo_bot::obs::team_size(),
        "games": w + l + d, "wins": w, "losses": l, "draws": d,
        "win_rate_decisive": if w + l > 0 { w as f64 / (w + l) as f64 } else { 0.5 },
        "steps": steps, "bot_decisions": decisions, "scripted_kickoffs": scripted,
        "delay": kind.delay(), "stochastic": stochastic, "kickoff_script": script,
        "jitter": h2h_jitter(true),
        "device": format!("{:?}", indococo_bot::nectofam::NfModel::default_device()),
        "seconds": t0.elapsed().as_secs_f64(),
    });
    println!("{out}");
}

#[allow(clippy::too_many_arguments)]
fn nf_worker(
    kind: indococo_bot::nectofam::Kind,
    mix: &[WeightedSetter],
    ckpt: &Path,
    dir: &Path,
    games: usize,
    seed: u64,
    stochastic: f32,
    script: bool,
    dump: Option<&Path>,
) -> [u64; 6] {
    use indococo_bot::nectofam as nf;
    use std::collections::HashMap;
    use std::io::Write;
    const ENVS: usize = 8;
    const DUMP_MAX: usize = 3000;
    let dev = Device::flex();
    let jitter = h2h_jitter(true);
    let na = load_actor(ckpt, &dev);
    let model = nf::NfModel::load(kind, dir, nf::NfModel::default_device()).unwrap_or_else(|e| panic!("{e}"));
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut dump = dump.map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("dump")));
    let mut dumped = 0usize;
    let mut out = [0u64; 6];
    let mut traj = traj_writer(kind.name(), seed);
    let mut started = 0usize;
    struct G {
        env: BotEnv,
        state: GameState,
        obs: Vec<Vec<f32>>,
        a_team: u8,
        bots: Vec<Option<nf::NfBot>>,
        seq: Vec<u32>,
        uid: usize,
        live: bool,
    }
    let mut new_game = |k: usize, started: &mut usize, rng: &mut SmallRng| -> G {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs) = jittered_reset(&mut env, jitter.then_some(&mut *rng));
        let a_team = (k % 2) as u8;
        let stoch = stochastic > 0.0 && rng.random::<f32>() < stochastic;
        let bots = state
            .cars
            .iter()
            .map(|(info, _)| (info.team as u8 != a_team).then(|| nf::NfBot::new(kind, stoch, script)))
            .collect();
        let n = state.cars.len();
        let live = *started < games;
        let uid = *started;
        *started += 1;
        G { env, state, obs, a_team, bots, seq: vec![0; n], uid, live }
    };
    let mut gs: Vec<G> = (0..ENVS).map(|k| new_game(k, &mut started, &mut rng)).collect();
    let lpr = match kind {
        nf::Kind::Nexto => 90,
        nf::Kind::Necto => 12,
    };
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
        for g in gs.iter_mut().filter(|g| g.live) {
            let acts: Vec<usize> = g
                .state
                .cars
                .iter()
                .map(|(info, _)| {
                    if info.team as u8 == g.a_team {
                        ia += 1;
                        aa[ia - 1]
                    } else {
                        0
                    }
                })
                .collect();
            g.env.pre_step(&g.state, &acts);
        }
        for tick in 0..TICK_SKIP_NF {
            let cur: Vec<Option<GameState>> = gs
                .iter()
                .map(|g| g.live.then(|| if tick == 0 { g.state.clone() } else { g.env.current_state() }))
                .collect();
            // Decisions due on this tick, batched per entity count.
            let mut reqs: Vec<(usize, usize, nf::NfObs, bool)> = Vec::new();
            for (k, g) in gs.iter_mut().enumerate() {
                let Some(st) = &cur[k] else { continue };
                for (i, b) in g.bots.iter_mut().enumerate() {
                    if let Some((o, s)) = b.as_mut().and_then(|bot| bot.begin_tick(st, i)) {
                        reqs.push((k, i, o, s));
                    }
                }
            }
            let mut decided: HashMap<(usize, usize), usize> = HashMap::new();
            let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (r, q) in reqs.iter().enumerate() {
                groups.entry(q.2.ents).or_default().push(r);
            }
            for (ents, idx) in groups {
                let (mut q, mut kv, mut mk) = (Vec::new(), Vec::new(), Vec::new());
                for &r in &idx {
                    q.extend_from_slice(&reqs[r].2.q);
                    kv.extend_from_slice(&reqs[r].2.kv);
                    mk.extend_from_slice(&reqs[r].2.mask);
                }
                let logits = model.logits(ents, &q, &kv, &mk, idx.len());
                for (j, &r) in idx.iter().enumerate() {
                    let (k, i, ref o, s) = reqs[r];
                    let l = &logits[j * lpr..(j + 1) * lpr];
                    let a = nf::pick(kind, l, s, &mut rng);
                    decided.insert((k, i), a);
                    out[4] += 1;
                    if let Some(f) = dump.as_mut().filter(|_| dumped < DUMP_MAX) {
                        let st = cur[k].as_ref().expect("live");
                        let cars: Vec<_> = st.cars.iter().map(|(ci, c)| {
                            let fl = indococo_bot::obs::flags(c);
                            serde_json::json!({
                                "team": ci.team as u8, "pos": v3(c.phys.pos), "vel": v3(c.phys.vel), "ang_vel": v3(c.phys.ang_vel),
                                "rot": [v3(c.phys.rot_mat.x_axis), v3(c.phys.rot_mat.y_axis), v3(c.phys.rot_mat.z_axis)],
                                "boost": c.boost, "on_ground": fl.on_ground, "has_flipped": c.has_flipped,
                                "has_double_jumped": c.has_double_jumped, "air_time_since_jump": c.air_time_since_jump,
                                "has_jumped": c.has_jumped, "is_demoed": c.is_demoed,
                            })
                        }).collect();
                        let line = serde_json::json!({
                            "kind": kind.name(), "uid": gs[k].uid, "car": i, "seq": gs[k].seq[i], "tick": st.tick_count,
                            "cars": cars,
                            "ball": {"pos": v3(st.ball.phys.pos), "vel": v3(st.ball.phys.vel), "ang_vel": v3(st.ball.phys.ang_vel)},
                            "pads_active": nf::pads_active(st).to_vec(), "kickoff": nf::kickoff_phase(st), "stochastic": s,
                            "ents": o.ents, "q": o.q.to_vec(), "kv": o.kv, "mask": o.mask, "logits": l, "action": a,
                        });
                        writeln!(f, "{line}").expect("dump write");
                        dumped += 1;
                    }
                    gs[k].seq[i] += 1;
                }
            }
            for (k, g) in gs.iter_mut().enumerate() {
                let Some(st) = &cur[k] else { continue };
                for i in 0..g.bots.len() {
                    if let Some(bot) = g.bots[i].as_mut() {
                        let ko = bot.script && nf::kickoff_phase(st) && tick == 0;
                        let c = bot.end_tick(decided.get(&(k, i)).copied(), st, i);
                        if ko && c == nf::kickoff_rows()[0] && g.seq[i] <= 1 {
                            out[5] += 1;
                        }
                        g.env.arena.set_car_controls(i, nf::to_controls(&c));
                    }
                }
                g.env.step_physics(1);
            }
        }
        for k in 0..ENVS {
            if !gs[k].live {
                continue;
            }
            let res = gs[k].env.post_step();
            out[3] += 1;
            {
                let tv = gs[k].env.get_shared_info().view(&res.state);
                traj_line(&mut traj, gs[k].uid, gs[k].a_team, &tv, res.is_terminal || res.truncated);
            }
            if res.is_terminal || res.truncated {
                let view = gs[k].env.get_shared_info().view(&res.state);
                match view.scoring_team() {
                    Some(t) if t == gs[k].a_team => out[0] += 1,
                    Some(_) => out[1] += 1,
                    None => out[2] += 1,
                }
                gs[k] = new_game(k, &mut started, &mut rng);
            } else {
                (gs[k].state, gs[k].obs) = (res.state, res.obs);
            }
        }
    }
    if let Some(f) = traj.as_mut() {
        use std::io::Write;
        f.flush().expect("traj flush");
    }
    if let Some(mut f) = dump {
        f.flush().expect("dump flush");
    }
    out
}

const TICK_SKIP_NF: u32 = indococo_bot::nectofam::TICK_SKIP;

/// Spacing analysis: with `TRAJ_DUMP=<prefix>`, `h2h` and `nexto-h2h` /
/// `necto-h2h` workers write one JSON line per policy step (ball, every car:
/// team, pos, vel, boost, demoed, cumulative ball touches) to
/// `<prefix>.<tag>.<seed>.jsonl`, read by `tools/team_spacing.py`.
type TrajW = Option<std::io::BufWriter<std::fs::File>>;
fn traj_writer(tag: &str, seed: u64) -> TrajW {
    let p = std::env::var("TRAJ_DUMP").ok()?;
    Some(std::io::BufWriter::new(std::fs::File::create(format!("{p}.{tag}.{seed}.jsonl")).expect("traj dump")))
}
fn traj_line(w: &mut TrajW, gid: usize, a_team: u8, v: &StateView, done: bool) {
    use std::io::Write;
    let Some(f) = w.as_mut() else { return };
    let r = |x: f64| x.round();
    let cars: Vec<_> = v.cars.iter().map(|c| serde_json::json!([
        c.team, r(c.pos.x), r(c.pos.y), r(c.pos.z), r(c.vel.x), r(c.vel.y), r(c.vel.z),
        (c.boost * 100.0).round() / 100.0, u8::from(c.is_demoed()), c.ball_touches,
    ])).collect();
    let b = &v.ball;
    let s = if done { v.scoring_team().map_or(-1, i64::from) } else { -1 };
    let line = serde_json::json!({"g": gid, "a": a_team, "t": v.tick_count,
        "b": [r(b.pos.x), r(b.pos.y), r(b.pos.z), r(b.vel.x), r(b.vel.y), r(b.vel.z)],
        "c": cars, "d": done, "s": s});
    writeln!(f, "{line}").expect("traj write");
}

/// `megabyte-obs`: dump Megabyte obs for various states.
/// Generates states and runs them for some time to deplete pads, creating non-kickoff obs.
fn megabyte_obs(mdir: &Path, _states_path: &Path) {
    use indococo_bot::megabyte as mx;
    use rlgymppo::ExternalPolicy;
    
    init_from_default(true).expect("load collision meshes");
    let dev = tch::Device::Cuda(0);
    let mb = mx::MegabytePolicy::load(mdir, dev).expect("megabyte weights");
    
    // Generate varied states: kickoffs and mid-game states with some pads consumed
    let mix = vec![WeightedSetter { opp_weights: Default::default(), mode_weights: Default::default(), weight: 1.0, kind: indococo_bot::setters::SetterKind::Kickoff }];
    let mut env = create_env(&RewardConfig::default(), &mix);
    
    eprintln!("Generating observations from simulated states (including depleted pads)...");
    let mut count = 0;
    for seed in 0..20 {
        let (mut state, _, _) = env.reset();
        
        // Advance the state by random steps to create varied pad states
        for _ in 0..(seed % 8) * 20 {
            let acts = vec![0; state.cars.len()];
            env.pre_step(&state, &acts);
            env.step_physics(8);
            state = env.post_step().state;
        }
        
        for car_idx in 0..state.cars.len() {
            let mut obs = vec![0.0f32; mx::OBS_SIZE];
            let mut mask = vec![false; mx::N_ACTIONS];
            mb.build(&state, car_idx, &mut obs, &mut mask);
            let actions = mb.infer(&obs, &mask, 1);
            
            let out = serde_json::json!({
                "state_idx": count,
                "car": car_idx,
                "obs": obs,
                "mask": mask,
                "action": actions[0],
            });
            println!("{out}");
            count += 1;
        }
    }
    eprintln!("✓ Generated {} observations", count);
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
    let mix = vec![WeightedSetter { opp_weights: Default::default(), mode_weights: Default::default(), weight: 1.0, kind: indococo_bot::setters::SetterKind::Kickoff }];
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
    let mut jrng = h2h_jitter(false).then(|| SmallRng::seed_from_u64(rand::rng().next_u64()));
    let mut new_game = |k: usize, started: &mut usize| -> G {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs) = jittered_reset(&mut env, jrng.as_mut());
        let live = *started < games;
        *started += 1;
        G { env, state, obs, a_team: (k % 2) as u8, live }
    };
    let mut g: Vec<G> = (0..ENVS).map(|k| new_game(k, &mut started)).collect();
    let mut mb_obs: Vec<f32> = Vec::new();
    let mut mb_mask: Vec<bool> = Vec::new();
    loop {
        let live = g.iter().filter(|e| e.live).count();
        if live == 0 { break; }
        // Candidate actions (greedy), one per candidate car (all team sizes).
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
        // TM1: Megabyte drives EVERY car of the other team (was: only the first).
        let mut rows = 0usize;
        mb_obs.clear();
        mb_mask.clear();
        for k in 0..ENVS {
            if !g[k].live { continue; }
            for (i, (info, _)) in g[k].state.cars.iter().enumerate() {
                if info.team as u8 != g[k].a_team {
                    let (off, moff) = (mb_obs.len(), mb_mask.len());
                    mb_obs.resize(off + mx::OBS_SIZE, 0.0);
                    mb_mask.resize(moff + mx::N_ACTIONS, false);
                    mb.build(&g[k].state, i, &mut mb_obs[off..off + mx::OBS_SIZE], &mut mb_mask[moff..moff + mx::N_ACTIONS]);
                    rows += 1;
                }
            }
        }
        let mb_acts = mb.infer(&mb_obs, &mb_mask, rows);
        let mut im = 0;
        // Step environments with action delay protocol
        for k in 0..ENVS {
            if !g[k].live { continue; }
            let e = &mut g[k];
            // Build acts array in game order
            let mut acts = Vec::new();
            let mut ext = Vec::new();  // (car_idx, held_controls, new_controls)
            for (i, (info, car)) in e.state.cars.iter().enumerate() {
                if info.team as u8 == e.a_team {
                    acts.push(aa[ia]);
                    ia += 1;
                } else {
                    acts.push(0);  // placeholder
                    ext.push((i, car.controls, mx::controls_of(mb_acts[im])));
                    im += 1;
                }
            }
            // Apply candidate actions
            e.env.pre_step(&e.state, &acts);
            // Hold Megabyte's previous controls for ACTION_DELAY ticks
            for &(i, held, _) in &ext {
                e.env.arena.set_car_controls(i, held);
            }
            e.env.step_physics(mx::ACTION_DELAY);
            // Apply Megabyte's new controls for remaining ticks
            for &(i, _, new) in &ext {
                e.env.arena.set_car_controls(i, new);
            }
            e.env.step_physics(mx::TICK_SKIP - mx::ACTION_DELAY);
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

    /// TM3 `widen-pred` / `load_net_any` 235 -> 270: first layers of actor
    /// AND critic keep their rows bit for bit with exact zero rows after
    /// them, everything else is identical, and the widened nets give the
    /// source's outputs whatever the 35 new inputs hold.
    #[test]
    fn widen_pred_zero_rows_keep_the_function() {
        let dev = Device::flex();
        let (from, to) = (indococo_bot::obs::TEAM_OBS_SIZE, indococo_bot::obs::PRED_OBS_SIZE);
        assert_eq!((from, to), (235, 270));
        let (n, w) = (model_with_input(from, &dev), model_with_input(to, &dev));
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut vals = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0
                })
                .collect()
        };
        let rows = 64;
        let x = vals(rows * from);
        let extra = vals(rows * (to - from));
        let mut xw = Vec::with_capacity(rows * to);
        for r in 0..rows {
            xw.extend_from_slice(&x[r * from..(r + 1) * from]);
            xw.extend_from_slice(&extra[r * (to - from)..(r + 1) * (to - from)]);
        }
        for (narrow, mut wide) in [(n.actor, w.actor), (n.critic, w.critic)] {
            let mut st = SafetensorsStore::from_bytes(None);
            narrow.clone().save_into(&mut st).expect("save");
            let bytes = widen_safetensors(&st.get_bytes().expect("bytes"), from, to);
            let mut st = SafetensorsStore::from_bytes(Some(bytes));
            wide.load_from(&mut st).expect("load widened");
            assert!(first_layer_extends(&narrow, &wide, from), "zero rows / identical tensors");
            let a = narrow.forward(Tensor::<2>::from_data(TensorData::new(x.clone(), [rows, from]), &dev));
            let b = wide.forward(Tensor::<2>::from_data(TensorData::new(xw.clone(), [rows, to]), &dev));
            let d = (a - b).abs().max().into_data().try_to_vec::<f32>().expect("diff")[0];
            assert!(d < 1e-4, "max |output diff| {d}");
        }
    }

    fn cfg(warmup: u64, iterations: u64) -> TrainConfig {
        let text = include_str!("../../config/train.toml");
        let mut c: TrainConfig = toml::from_str(text).expect("train.toml");
        c.critic_warmup_steps = warmup;
        c.iterations = iterations;
        c
    }

    /// TM2 template with the launcher placeholders filled: parses, mode mix
    /// 35/40/25, setter mix sums to 1 with mechanic drills <= 22%.
    #[test]
    fn tm2_train_template_parses() {
        let mut text = include_str!("../../experiments/train_tm2.toml").to_string();
        for (k, v) in [
            ("@RUN@", "/tmp/tm2"), ("@ITERATIONS@", "1"), ("@WARM@", "0"), ("@HOLD@", "0"),
            ("@SNAP@", "/tmp/snap.f32"), ("@TPI@", "300_000"), ("@MB@", "100_000"), ("@LR@", "5e-5"),
            ("@ENT@", "0.016"), ("@GAMMA@", "0.995"), ("@OPPCH@", "0.40"), ("@PASTW@", "1.0"),
            ("@DECAY@", "1_000_000_000"), ("@MINLR@", "0.4"),
        ] {
            text = text.replace(k, v);
        }
        assert!(!text.contains('@'), "unfilled placeholder");
        let c: TrainConfig = toml::from_str(&text).expect("train_tm2.toml");
        let mix = &c.team.as_ref().expect("[team]").mode_mix;
        assert_eq!((mix["1"], mix["2"], mix["3"]), (0.35, 0.40, 0.25));
        let total: f32 = c.state_setters.iter().map(|s| s.weight).sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
        let mech: f32 = c
            .state_setters
            .iter()
            .filter(|s| {
                ["backboard_double_tap", "reset_shot", "musty_setup", "air_near_ball", "air_reset"]
                    .contains(&s.kind.label())
            })
            .map(|s| s.weight)
            .sum();
        assert!(mech <= 0.2201, "{mech}");
        let labels: Vec<_> = c.state_setters.iter().map(|s| s.kind.label()).collect();
        assert!(labels.contains(&"reset_shot") && labels.contains(&"musty_setup"));
    }

    fn fill_train_placeholders(text: &str) -> String {
        let mut text = text.to_string();
        for (k, v) in [
            ("@RUN@", "/tmp/tm11"), ("@ITERATIONS@", "1"), ("@WARM@", "0"), ("@HOLD@", "0"),
            ("@SNAP@", "/tmp/snap.f32"), ("@TPI@", "300_000"), ("@MB@", "100_000"), ("@LR@", "5e-5"),
            ("@ENT@", "0.016"), ("@GAMMA@", "0.995"), ("@OPPCH@", "0.40"), ("@PASTW@", "1.0"),
            ("@DECAY@", "1_000_000_000"), ("@MINLR@", "0.4"),
        ] {
            text = text.replace(k, v);
        }
        text
    }

    fn share_of(shares: &[(&str, f32)], label: &str) -> f32 {
        shares.iter().find(|(l, _)| *l == label).map(|(_, w)| *w).unwrap_or(0.0)
    }

    /// TM11 train template: per-mode shares match the §2 table; replay is
    /// exactly 0 in 2v2/3v3; defend_carry is in the mix.
    #[test]
    fn tm11_train_template_parses() {
        let text = fill_train_placeholders(include_str!("../../experiments/train_tm11.toml"));
        assert!(!text.contains('@'), "unfilled placeholder");
        let c: TrainConfig = toml::from_str(&text).expect("train_tm11.toml");
        let mix = indococo_bot::setters::build_mix::<indococo_bot::env::SharedInfo>(&c.state_setters);
        let labels: Vec<_> = c.state_setters.iter().map(|s| s.kind.label()).collect();
        assert!(labels.contains(&"defend_carry"), "{labels:?}");
        let cases: &[(usize, Option<&str>, &str, f32)] = &[
            (1, None, "kickoff", 0.15 / 1.09),
            (1, None, "defend_carry", 0.09 / 1.09),
            (1, None, "replay", 0.09 / 1.09),
            (1, None, "team_play", 0.15 / 1.09),
            (2, None, "kickoff", 0.15),
            (2, None, "defend_carry", 0.09),
            (2, None, "replay", 0.0),
            (2, None, "team_play", 0.15),
            (3, None, "kickoff", 0.20 / 1.05),
            (3, None, "defend_carry", 0.09 / 1.05),
            (3, None, "replay", 0.0),
            (3, None, "team_play", 0.15 / 1.05),
            (1, Some("galactus"), "kickoff", 0.15 / 1.25),
            (1, Some("galactus"), "defend_carry", 0.25 / 1.25),
            (1, Some("galactus"), "replay", 0.09 / 1.25),
            (1, Some("galactus"), "team_play", 0.15 / 1.25),
            (2, Some("galactus"), "kickoff", 0.15 / 1.16),
            (2, Some("galactus"), "defend_carry", 0.25 / 1.16),
            (2, Some("galactus"), "replay", 0.0),
            (2, Some("galactus"), "team_play", 0.15 / 1.16),
            (3, Some("galactus"), "kickoff", 0.20 / 1.21),
            (3, Some("galactus"), "defend_carry", 0.25 / 1.21),
            (3, Some("galactus"), "replay", 0.0),
            (3, Some("galactus"), "team_play", 0.15 / 1.21),
        ];
        for &(per_team, opp, label, want) in cases {
            let shares = mix.shares(per_team, opp);
            let got = share_of(&shares, label);
            assert!((got - want).abs() < 1e-4, "{per_team}v vs {opp:?} {label}: got {got} want {want}");
        }
        for per_team in [2usize, 3] {
            let shares = mix.shares(per_team, None);
            assert_eq!(share_of(&shares, "replay"), 0.0, "replay in {per_team}v");
        }
    }

    #[test]
    fn st35_pfsp_section_parses_and_is_off_by_default() {
        assert!(cfg(0, 0).pfsp.is_none());
        let text = format!(
            "{}\n[pfsp]\nself_share = 0.30\npast_share = 0.20\nfloor = 0.03\npinned = [\"HUNT_final\", \"FR4_final\"]\nlog = \"/tmp/pfsp.jsonl\"\n[pfsp.clamp]\ngalactus = [0.20, 0.26]\n",
            include_str!("../../config/train.toml")
        );
        let c: TrainConfig = toml::from_str(&text).expect("train.toml + [pfsp]");
        let p = c.pfsp.expect("pfsp");
        assert!(p.enabled && (p.league_share() - 0.5).abs() < 1e-12);
        assert_eq!(p.clamp["galactus"], [0.20, 0.26]);
        assert_eq!(p.pinned, vec!["HUNT_final".to_string(), "FR4_final".to_string()]);
    }

    /// The PFSP hook's mix reaches rlgymppo's pick: self 0.30, past 0.20,
    /// league 0.50 with Galactus at its 20% clamp floor (prior win rates).
    #[test]
    fn st35_pfsp_hook_drives_the_rlgymppo_pick() {
        use rand::SeedableRng;
        let mut p = indococo_bot::pfsp::PfspConfig::default();
        p.clamp.insert("galactus".into(), [0.20, 0.26]);
        p.pinned = vec!["HUNT_final".into(), "FR4_final".into()];
        let names = ["FR4_final", "ST25_final", "HUNT_final", "ripple", "galactus", "megabyte"];
        p.validate(&names).unwrap();
        indococo_bot::pfsp::install(p);
        let m = pfsp_mix_hook(&names, 3).expect("pfsp on");
        assert!((m.chance - 0.70).abs() < 1e-6 && (m.past_weight - 0.20).abs() < 1e-6);
        assert!((m.weights.iter().sum::<f32>() - 0.50).abs() < 1e-5);
        assert!((m.weights[4] - 0.20).abs() < 1e-5, "galactus {:?}", m.weights);
        assert!((m.weights[0] - 0.015).abs() < 1e-5 && (m.weights[2] - 0.015).abs() < 1e-5);
        // No past versions yet: past goes to self-play, league stays 0.50.
        let m0 = pfsp_mix_hook(&names, 0).unwrap();
        assert!((m0.chance - 0.50).abs() < 1e-6 && m0.past_weight == 0.0);
        // Through SelfPlayConfig::pick_opponent_mix (no opponents loaded, so
        // only self vs past): 30% self-play, 70% past with a 0.7/0.2 mix.
        let cfg = SelfPlayConfig { train_against_old_versions: true, past_versions_weight: 0.0, ..Default::default() };
        let mix = rlgymppo::OpponentMix { chance: 0.7, past_weight: 0.2, weights: vec![] };
        let mut rng = SmallRng::seed_from_u64(5);
        let old = (0..100_000)
            .filter(|_| matches!(cfg.pick_opponent_mix(&mix, 2, &mut rng), Some(rlgymppo::OpponentPick::PastVersion(_))))
            .count();
        assert!((old as f64 / 1e5 - 0.70).abs() < 0.01, "old {old}");
        // Old versions off: never a past pick, all self-play.
        let off = SelfPlayConfig { train_against_old_versions: false, ..Default::default() };
        assert!((0..1000).all(|_| off.pick_opponent_mix(&mix, 2, &mut rng).is_none()));
        indococo_bot::pfsp::install(indococo_bot::pfsp::PfspConfig { enabled: false, ..Default::default() });
        assert!(pfsp_mix_hook(&names, 3).is_none());
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
        assert!(!c.async_collect);
        assert!(!c.collect_stream_priority);
        assert_eq!(c.nf_intra_threads, 0);
        assert_eq!(c.league_bots_device, LeagueBotsDevice::Default);
    }

    #[test]
    fn speed_flags_parse_and_default_off() {
        let c = parse("async_collect = true\nleague_bots_device = \"cuda\"\n");
        assert!(c.async_collect);
        assert_eq!(c.league_bots_device, LeagueBotsDevice::Cuda);
        let cpu = parse("league_bots_device = \"cpu\"\n");
        assert_eq!(cpu.league_bots_device, LeagueBotsDevice::Cpu);
        assert!(!cpu.async_collect);
        assert!(!cpu.collect_stream_priority);
        let prio = parse("async_collect = true\ncollect_stream_priority = true\nnf_intra_threads = 4\n");
        assert!(prio.async_collect);
        assert!(prio.collect_stream_priority);
        assert_eq!(prio.nf_intra_threads, 4);
    }

    #[test]
    fn speed_keys_before_first_table_parse() {
        let base = include_str!("../../config/train.toml");
        let tables = concat!(
            "\n[[opponents.external]]\n",
            "name = \"nexto\"\n",
            "kind = \"nexto\"\n",
            "path = \"/tmp/nexto\"\n",
            "weight = 0.3\n",
            "replicas = 8\n",
            "chunk_rows = 64\n",
        );
        let with_tables = format!("{base}{tables}");
        // Same bug as appending after the last table: these keys must be rejected.
        let after = format!(
            "{with_tables}nf_intra_threads = 2\nasync_collect = true\ncollect_stream_priority = true\n"
        );
        let err = toml::from_str::<TrainConfig>(&after)
            .expect_err("keys after [[opponents.external]] must fail")
            .to_string();
        assert!(
            err.contains("nf_intra_threads")
                || err.contains("unknown")
                || err.contains("did not expect")
                || err.contains("unused"),
            "unexpected deny_unknown_fields error: {err}"
        );

        let keys = "nf_intra_threads = 2\nasync_collect = true\ncollect_stream_priority = true\n";
        let idx = with_tables
            .find("[[opponents.external]]")
            .expect("fixture table");
        let before = format!("{}{}{}", &with_tables[..idx], keys, &with_tables[idx..]);
        let c: TrainConfig = toml::from_str(&before).expect("top-level speed keys");
        assert_eq!(c.nf_intra_threads, 2);
        assert!(c.async_collect);
        assert!(c.collect_stream_priority);
        assert_eq!(c.opponents.external.len(), 1);
        assert_eq!(c.opponents.external[0].replicas, 8);
        assert_eq!(c.opponents.external[0].chunk_rows, 64);
    }

    #[test]
    fn speed_variants_default_and_subset() {
        let default = parse_speed_variants(None);
        assert_eq!(
            default.iter().map(|(n, _, _, _)| *n).collect::<Vec<_>>(),
            ["off", "async", "async_prio", "league_cuda", "league_cpu", "both", "both_cpu"]
        );
        let subset = parse_speed_variants(Some("off,async,async_prio"));
        assert_eq!(subset.len(), 3);
        assert_eq!(subset[0].0, "off");
        assert!(!subset[0].1);
        assert!(!subset[0].2);
        assert!(subset[1].1);
        assert!(!subset[1].2);
        assert!(subset[2].1 && subset[2].2);
        assert_eq!(parse_speed_variants(Some("")).len(), 7);
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
/// (draw). Also counts flip resets and their goalward uses per side (all cars),
/// plus demos-for and demos-against.
/// Prints one JSON line: wins, losses, draws, win_rate_decisive, resets, demos_against.
/// `h2h` counters: 9 classic + TM1 team counters (see `h2h`).
/// TM5 live-play metrics: some teammate pair <800; 2+ cars within 1500 of
/// ball; 2+ within 1000 and closing >=200 uu/s; a living car goalside by
/// >=1500 uu. None while kickoff pending, a teammate demoed, or team <2.
fn live_spacing(v: &StateView, team: u8) -> Option<[bool; 4]> {
    if indococo_bot::team::kickoff_pending(v) { return None; }
    let cars: Vec<_> = v.cars.iter().filter(|c| c.team == team).collect();
    if cars.len() < 2 || cars.iter().any(|c| c.is_demoed()) { return None; }
    let pair = cars.iter().enumerate().any(|(i, c)| cars[i + 1..].iter().any(|o| c.pos.distance(o.pos) < 800.0));
    let near = cars.iter().filter(|c| c.pos.distance(v.ball.pos) < 1500.0).count() >= 2;
    let dc = cars.iter().filter(|c| {
        let d = v.ball.pos - c.pos;
        let n = d.length();
        n > 1e-6 && n < 1000.0 && c.vel.dot(d / n) >= 200.0
    }).count() >= 2;
    let deep = cars.iter().any(|c| c.attack_sign() * (v.ball.pos.y - c.pos.y) >= 1500.0);
    Some([pair, near, dc, deep])
}

#[cfg(test)]
mod spacing_tests {
    use super::*;
    use indococo_bot::state::CarView;
    #[test]
    fn live_spacing_matches_definition() {
        let mut v = StateView::default();
        v.ball.pos.y = 100.0; // not a pending kickoff
        let c = |y: f64, vy: f64| CarView { team: 0, pos: glam::DVec3::new(0.0,y,17.0), vel: glam::DVec3::new(0.0,vy,0.0), ..CarView::default() };
        v.cars = vec![c(-300.0, 1000.0), c(-500.0, 500.0)];
        assert_eq!(live_spacing(&v, 0), Some([true, true, true, false]));
        v.cars[1] = c(-2500.0, 0.0);
        assert_eq!(live_spacing(&v, 0), Some([false, false, false, true]));
        v.cars[1].demo_respawn_timer = 1.0;
        assert_eq!(live_spacing(&v, 0), None);
    }
}

/// TM6 role metrics for one team: the opponents have the ball (an
/// opponent is the closest living car to it, within 1000 uu) in `team`'s
/// half, and whether no living `team` car is within 1500 uu of the ball
/// (the "no challenger" case). None while the kickoff is pending.
fn opp_ball_own_half(v: &StateView, team: u8) -> Option<(bool, bool)> {
    if indococo_bot::team::kickoff_pending(v) { return None; }
    let closest = v.cars.iter().filter(|c| !c.is_demoed()).min_by(|a, b| {
        a.pos.distance(v.ball.pos).partial_cmp(&b.pos.distance(v.ball.pos)).unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let sign = if team == 0 { 1.0 } else { -1.0 };
    let opp_ball = closest.team != team && closest.pos.distance(v.ball.pos) < 1000.0 && sign * v.ball.pos.y < 0.0;
    let challenger = v.cars.iter().any(|c| c.team == team && !c.is_demoed() && c.pos.distance(v.ball.pos) < 1500.0);
    Some((opp_ball, opp_ball && !challenger))
}

#[cfg(test)]
mod role_metric_tests {
    use super::*;
    use indococo_bot::state::CarView;
    #[test]
    fn no_challenger_needs_opponent_ball_in_own_half_and_no_teammate_near() {
        let mut v = StateView::default();
        v.ball.pos.y = -2000.0; // blue half, not a pending kickoff
        let c = |team: u8, y: f64| CarView { team, pos: glam::DVec3::new(0.0, y, 17.0), ..CarView::default() };
        v.cars = vec![c(0, -4500.0), c(1, -1800.0), c(0, 1000.0), c(1, 2000.0)];
        assert_eq!(opp_ball_own_half(&v, 0), Some((true, true)));
        v.cars[0].pos.y = -3000.0; // teammate within 1500 of the ball
        assert_eq!(opp_ball_own_half(&v, 0), Some((true, false)));
        assert_eq!(opp_ball_own_half(&v, 1), Some((false, false)), "orange: ball in blue's half");
        v.ball.pos = glam::DVec3::new(0.0, 0.0, 93.0);
        assert_eq!(opp_ball_own_half(&v, 0), None, "pending kickoff");
    }

    #[test]
    fn h2h_role_params_from_file() {
        use indococo_bot::team::RoleParams;
        let path = format!("{}/experiments/tm11.toml", env!("CARGO_MANIFEST_DIR"));
        let got = h2h_role_params_from(Some(&path));
        assert_eq!(
            got,
            RoleParams { margin: 300.0, hold: 8, obs_tiebreak: true, touch_mode: "hysteresis".into() }
        );
        assert_eq!(h2h_role_params_from(None), RoleParams::default());
        assert_eq!(h2h_role_params_from(Some("")), RoleParams::default());
    }

    #[test]
    fn n_h2h_in_sync() {
        assert_eq!(N_H2H, 59);
        assert_eq!(N_H2H, indococo_bot::h2h_counters::N_H2H);
    }
}

/// Role hysteresis for h2h role metrics: the run's `[team.roles]` from the reward
/// config named by H2H_REWARD_CONFIG, else RoleParams::default() (legacy h2h).
fn h2h_role_params_from(path: Option<&str>) -> indococo_bot::team::RoleParams {
    use indococo_bot::team::RoleParams;
    match path {
        None | Some("") => RoleParams::default(),
        Some(p) => RewardConfig::from_file(Path::new(p))
            .unwrap_or_else(|e| panic!("H2H_REWARD_CONFIG {p}: {e}"))
            .team
            .roles
            .unwrap_or_default(),
    }
}
fn h2h_role_params() -> indococo_bot::team::RoleParams {
    h2h_role_params_from(std::env::var("H2H_REWARD_CONFIG").ok().as_deref())
}

const N_H2H: usize = 59;

fn h2h(a: &Path, b: &Path, games: usize, threads: usize) {
    init_from_default(true).expect("load collision meshes");
    let mix = vec![WeightedSetter { opp_weights: Default::default(), mode_weights: Default::default(),
        weight: 1.0,
        kind: indococo_bot::setters::SetterKind::Kickoff,
    }];
    let per = games.div_ceil(threads);
    let roles = h2h_role_params();
    let res: Vec<[u64; N_H2H]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let mix = &mix;
                let roles = roles.clone();
                s.spawn(move || h2h_worker(mix, a, b, per, 7000 + t as u64, roles))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = [0u64; N_H2H];
    for r in &res {
        for i in 0..N_H2H {
            tot[i] += r[i];
        }
    }
    let [w, l, d, ra, rb, ua, ub, da, db] = <[u64; 9]>::try_from(&tot[..9]).expect("9");
    let mut out = serde_json::json!({
        "a": a, "b": b, "team_size": indococo_bot::obs::team_size(),
        "games": w + l + d, "wins": w, "losses": l, "draws": d,
        "win_rate_decisive": if w + l > 0 { w as f64 / (w + l) as f64 } else { 0.5 },
        "resets_a": ra, "resets_b": rb, "uses_a": ua, "uses_b": ub,
        "demos_against_a": da, "demos_against_b": db,
        "bumps_a": tot[55], "bumps_b": tot[56], "demos_a": tot[57], "demos_b": tot[58],
        "roles": serde_json::to_value(&roles).expect("roles"),
    });
    // TM1 team metrics (team modes only): per side a / b.
    if indococo_bot::obs::team_size() > 1 {
        let steps = tot[19].max(1) as f64;
        let mins = steps / (120.0 / 8.0 * 60.0);
        let side = |k: usize| serde_json::json!({
            "passes_per_min": tot[9 + k] as f64 / mins,
            "aerial_passes_per_min": tot[11 + k] as f64 / mins,
            "double_taps": tot[13 + k],
            "air_double_touches_per_min": tot[15 + k] as f64 / mins,
            "double_commit_ratio": tot[17 + k] as f64 / steps,
            "kickoff_duplicate_share": tot[20 + k] as f64 / tot[22].max(1) as f64,
            "last_man_back_ratio": tot[23 + k] as f64 / steps,
            // TM5 live-play spacing (skip pending kickoff and demoed teammates).
            "spacing_steps": tot[33 + k],
            "pair_lt800_ratio": tot[25 + k] as f64 / tot[33 + k].max(1) as f64,
            "two_near_ball1500_ratio": tot[27 + k] as f64 / tot[33 + k].max(1) as f64,
            "double_commit_wide_ratio": tot[29 + k] as f64 / tot[33 + k].max(1) as f64,
            "deep_last_man_ratio": tot[31 + k] as f64 / tot[33 + k].max(1) as f64,
            "flip_resets_per_game": ([ra, rb][k] as f64) / ((w + l + d).max(1) as f64),
            // TM6 role metrics (RoleTracker 300 uu / 8 steps; touch role = role before the touch step).
            "role_swaps_per_min": tot[35 + k] as f64 / mins,
            "first_mean_stint_s": steps / 15.0 / ((tot[35 + k] + w + l + d).max(1) as f64),
            "touch_share_first": tot[37 + k] as f64 / (tot[37 + k] + tot[39 + k] + tot[41 + k]).max(1) as f64,
            "touch_share_second": tot[39 + k] as f64 / (tot[37 + k] + tot[39 + k] + tot[41 + k]).max(1) as f64,
            "touch_share_last": tot[41 + k] as f64 / (tot[37 + k] + tot[39 + k] + tot[41 + k]).max(1) as f64,
            "no_challenger_ratio": tot[43 + k] as f64 / steps,
            "opp_ball_own_half_ratio": tot[45 + k] as f64 / steps,
            // TM8 passivity (as reward-audit's behaviour block; live play, no pending kickoff):
            // ball touches per car-minute and the mean distance of the team's deepest living
            // car from its own back wall (lower = sitting deeper).
            "touches_per_car_min": tot[47 + k] as f64 / tot[49 + k].max(1) as f64 * 15.0 * 60.0,
            "deepest_car_depth_uu": tot[53 + k] as f64 / tot[51 + k].max(1) as f64,
            "bumps_per_min": tot[55 + k] as f64 / mins.max(1e-9),
            "demos_per_min": tot[57 + k] as f64 / mins.max(1e-9),
        });
        out["team"] = serde_json::json!({"a": side(0), "b": side(1), "steps": tot[19], "kickoffs": tot[22]});
    }
    println!("{out}");
}

fn h2h_worker(mix: &[WeightedSetter], a: &Path, b: &Path, games: usize, seed: u64, roles: indococo_bot::team::RoleParams) -> [u64; N_H2H] {
    const ENVS: usize = 8;
    let dev = Device::flex();
    let (na, nb) = (load_actor(a, &dev), load_actor(b, &dev));
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = [0u64; N_H2H];
    let mut started = 0usize;
    struct G {
        env: BotEnv,
        state: GameState,
        obs: Vec<Vec<f32>>,
        prev: StateView,
        a_team: u8,
        uses: FlipResetUse,
        touches: indococo_bot::team::TouchTracker,
        roles: indococo_bot::team::RoleTracker,
        live: bool,
        gid: usize,
    }
    let mut traj = traj_writer("h2h", seed);
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
        let mut touches = indococo_bot::team::TouchTracker::default();
        touches.reset(&prev);
        let live = *started < games;
        let gid = *started;
        *started += 1;
        let roles = indococo_bot::team::RoleTracker::new(roles.clone());
        G { env, state, obs, prev, a_team: (k % 2) as u8, uses, touches, roles, live, gid }
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
            traj_line(&mut traj, g.gid, g.a_team, &view, res.is_terminal || res.truncated);
            let used = g.uses.get_rewards(&view);
            let events = StepEvents::between(&g.prev, &view);
            for (i, c) in view.cars.iter().enumerate() {
                let side = usize::from(c.team != g.a_team);
                if resets::is_flip_reset(&g.prev.cars[i], c, &view.ball) {
                    out[3 + side] += 1;
                }
                if used[i] > 0.0 {
                    out[5 + side] += 1;
                }
                if events[i].demoed {
                    out[7 + side] += 1;
                }
                out[55 + side] += u64::from(events[i].bump);
                out[57 + side] += u64::from(events[i].demo);
            }
            if view.cars.len() > 2 {
                let te = g.touches.step(&view);
                let sd = |car: usize| usize::from(view.cars[car].team != g.a_team);
                for p in &te.passes {
                    out[9 + sd(p.receiver)] += 1;
                    if p.aerial {
                        out[11 + sd(p.receiver)] += 1;
                    }
                }
                if let Some((scorer, _)) = te.double_tap {
                    out[13 + sd(scorer)] += 1;
                }
                for &c in &te.air_doubles {
                    out[15 + sd(c)] += 1;
                }
                if te.kickoff_touched {
                    out[22] += 1;
                    if let Some(t) = te.kickoff_duplicate {
                        out[20 + usize::from(t != g.a_team)] += 1;
                    }
                }
                for (k, team) in [g.a_team, 1 - g.a_team].into_iter().enumerate() {
                    let near = view.cars.iter().filter(|c| c.team == team && !c.is_demoed() && c.pos.distance(view.ball.pos) < 500.0).count();
                    out[17 + k] += u64::from(near >= 2);
                    let back = view.cars.iter().any(|c| c.team == team && !c.is_demoed() && c.attack_sign() * (c.pos.y - view.ball.pos.y) < 0.0);
                    out[23 + k] += u64::from(back);
                    if let Some(m) = live_spacing(&view, team) {
                        out[33 + k] += 1;
                        out[25 + k] += u64::from(m[0]);
                        out[27 + k] += u64::from(m[1]);
                        out[29 + k] += u64::from(m[2]);
                        out[31 + k] += u64::from(m[3]);
                    }
                }
                // TM6 role metrics.
                let before = g.roles.roles().to_vec();
                let sw0 = g.roles.swaps;
                g.roles.update(&view);
                for (k, team) in [g.a_team, 1 - g.a_team].into_iter().enumerate() {
                    let t = usize::from(team);
                    out[35 + k] += g.roles.swaps[t] - sw0[t];
                    if before.len() == view.cars.len() {
                        for (i, c) in view.cars.iter().enumerate() {
                            if c.team == team && c.ball_touches > 0 {
                                match before[i] {
                                    indococo_bot::team::Role::First => out[37 + k] += 1,
                                    indococo_bot::team::Role::Second => out[39 + k] += 1,
                                    indococo_bot::team::Role::Last => out[41 + k] += 1,
                                    indococo_bot::team::Role::Out => {}
                                }
                            }
                        }
                    }
                    if let Some((ob, nc)) = opp_ball_own_half(&view, team) {
                        out[45 + k] += u64::from(ob);
                        out[43 + k] += u64::from(nc);
                    }
                    // TM8 passivity metrics.
                    if !indococo_bot::team::kickoff_pending(&view) {
                        let mut depth = f64::INFINITY;
                        for c in view.cars.iter().filter(|c| c.team == team) {
                            out[49 + k] += 1;
                            out[47 + k] += u64::from(c.ball_touches > 0);
                            if !c.is_demoed() {
                                depth = depth.min(c.attack_sign() * c.pos.y + 5120.0);
                            }
                        }
                        if depth.is_finite() {
                            out[51 + k] += 1;
                            out[53 + k] += depth.max(0.0).round() as u64;
                        }
                    }
                }
                out[19] += 1;
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
    if let Some(f) = traj.as_mut() {
        use std::io::Write;
        f.flush().expect("traj flush");
    }
    out
}

/// `ripple-h2h`: the candidate checkpoint (greedy, CPU) against the Rust Ripple port,
/// first goal wins, kickoff starts, both sides. Ripple uses tick_skip=4 internally.
#[cfg(feature = "ripple")]
fn ripple_h2h(ckpt: &Path, rdir: &Path, games: usize, threads: usize, idle_opponent: bool) {
    use indococo_bot::ripple as rx;
    init_from_default(true).expect("load collision meshes");
    println!("Starting ripple_h2h: games={}, threads={}, idle={}", games, threads, idle_opponent);
    let mix = vec![WeightedSetter { opp_weights: Default::default(), mode_weights: Default::default(), weight: 1.0, kind: indococo_bot::setters::SetterKind::Kickoff }];
    let per = games.div_ceil(threads);
    let t0 = std::time::Instant::now();
    let res: Vec<[u64; 4]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let mix = &mix;
            s.spawn(move || ripple_worker(mix, ckpt, rdir, per, idle_opponent))
        }).collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = [0u64; 4];
    for r in &res { for i in 0..4 { tot[i] += r[i]; } }
    let [w, l, d, steps] = tot;
    let out = serde_json::json!({
        "a": ckpt, "b": "ripple(rust)", "games": w + l + d, "wins": w, "losses": l, "draws": d,
        "win_rate_decisive": if w + l > 0 { w as f64 / (w + l) as f64 } else { 0.5 },
        "steps": steps, "seconds": t0.elapsed().as_secs_f64(),
        "steps_per_second": steps as f64 / t0.elapsed().as_secs_f64(),
    });
    println!("{out}");
}

#[cfg(not(feature = "ripple"))]
fn ripple_h2h(_ckpt: &Path, _rdir: &Path, _games: usize, _threads: usize, _idle_opponent: bool) {
    panic!("ripple-h2h requires --features ripple");
}

#[cfg(feature = "ripple")]
fn ripple_worker(mix: &[WeightedSetter], ckpt: &Path, rdir: &Path, games: usize, idle_opponent: bool) -> [u64; 4] {
    use indococo_bot::ripple as rx;
    use rlgymppo::ExternalPolicy;
    use std::io::{self, Write};
    const ENVS: usize = 8;
    let dev = Device::flex();
    let na = load_actor(ckpt, &dev);
    let ripple = rx::RipplePolicy::load(rdir).expect("ripple weights");
    let mut out = [0u64; 4];
    let mut started = 0usize;
    println!("Worker starting: games={}, idle={}", games, idle_opponent);
    io::stdout().flush().unwrap();
    struct G { env: BotEnv, state: GameState, obs: Vec<Vec<f32>>, a_team: u8, live: bool }
    let mut new_game = |k: usize, started: &mut usize| -> G {
        let mut env = create_env(&RewardConfig::default(), mix);
        let (state, obs, _) = env.reset();
        let live = *started < games;
        *started += 1;
        println!("  New game k={}, started={}, live={}", k, *started, live);
        io::stdout().flush().unwrap();
        G { env, state, obs, a_team: (k % 2) as u8, live }
    };
    let mut gs: Vec<G> = (0..ENVS).map(|k| new_game(k, &mut started)).collect();
    let mut robs = vec![0.0f32; ripple.obs_width()];
    let mut rmask = vec![false; rx::N_ACTIONS];
    println!("  Starting game loop, live games: {}", gs.iter().filter(|g| g.live).count());
    io::stdout().flush().unwrap();
    let mut step_count = 0;
    while gs.iter().any(|g| g.live) {
        step_count += 1;
        if step_count % 100 == 1 {
            println!("  Step {}, live: {}", step_count, gs.iter().filter(|g| g.live).count());
            io::stdout().flush().unwrap();
        }
        let aa = if idle_opponent {
            Vec::new() // Not needed for idle opponent
        } else {
            let mut oa = Vec::new();
            for g in gs.iter().filter(|g| g.live) {
                for (i, (info, _)) in g.state.cars.iter().enumerate() {
                    if info.team as u8 == g.a_team {
                        oa.push(&g.obs[i]);
                    }
                }
            }
            argmax_actions(&na, &oa, &dev)
        };
        let mut ia = 0;
        for k in 0..ENVS {
            if !gs[k].live { continue; }
            if step_count == 1 && k == 0 {
                println!("    Env 0 processing...");
                io::stdout().flush().unwrap();
            }
            let g = &mut gs[k];
            let mut acts = Vec::new();
            let mut ripple_car = None;
            for (i, (info, _)) in g.state.cars.iter().enumerate() {
                if info.team as u8 == g.a_team {
                    acts.push(if idle_opponent { 0 } else { aa[ia] });
                    if !idle_opponent { ia += 1; }
                } else {
                    acts.push(0); // Will be overridden by Ripple
                    ripple_car = Some(i);
                }
            }
            
            if step_count == 1 && k == 0 {
                println!("    Pre-step...");
                io::stdout().flush().unwrap();
            }
            // Apply candidate actions
            g.env.pre_step(&g.state, &acts);
            
            if step_count == 1 && k == 0 {
                println!("    Ripple decision...");
                io::stdout().flush().unwrap();
            }
            
            // Ripple makes 2 decisions per 8-tick step
            // Build once to get both decisions (Ripple tracks action history internally)
            if let Some(i) = ripple_car {
                if step_count == 1 && k == 0 {
                    println!("      Ripple.build...");
                    io::stdout().flush().unwrap();
                }
                // Decision 1
                ripple.build(&g.state, i, &mut robs, &mut rmask);
                if step_count == 1 && k == 0 {
                    println!("      Ripple.infer...");
                    io::stdout().flush().unwrap();
                }
                let actions = ripple.infer(&robs, &rmask, 1);
                let action1 = actions[0];
                let controls1 = rx::controls_of(action1);
                g.env.arena.set_car_controls(i, controls1);
                if started <= 2 { 
                    println!("  Step: Ripple car={} action1={} throttle={:.2} steer={:.2} x={:.0} y={:.0}", 
                        i, action1, controls1.throttle, controls1.steer, 
                        g.state.cars[i].1.phys.pos.x, g.state.cars[i].1.phys.pos.y); 
                    io::stdout().flush().unwrap();
                }
            }
            g.env.step_physics(4);

            // Decision 2 sees the arena after those 4 ticks. Assigning into
            // g.state keeps its address, so Ripple's queue stays on the same
            // key. pre_step already applied the candidate for the whole
            // 8-tick interval; this does not build the candidate's obs.
            g.state = g.env.current_state();

            if let Some(i) = ripple_car {
                ripple.build(&g.state, i, &mut robs, &mut rmask);
                let actions = ripple.infer(&robs, &rmask, 1);
                let action2 = actions[0];
                let controls2 = rx::controls_of(action2);
                g.env.arena.set_car_controls(i, controls2);
                if k == 0 { eprintln!("Ripple car={} action2={} throttle={}", i, action2, controls2.throttle); }
            }
            g.env.step_physics(4);
            
            let res = g.env.post_step();
            out[3] += 1;
            
            // Log per-second trajectory
            if out[3] % 120 == 0 {
                let ripple_team = 1 - g.a_team;
                let ripple_car_idx = g.state.cars.iter().position(|(info, _)| info.team as u8 != g.a_team).unwrap();
                let ripple_car = &g.state.cars[ripple_car_idx].1;
                let ball = &g.state.ball;
                println!("  [T={}s] Ripple team={} x={:.0} y={:.0} vx={:.0} vy={:.0} | Ball y={:.0}",
                    out[3] / 120, ripple_team,
                    ripple_car.phys.pos.x, ripple_car.phys.pos.y,
                    ripple_car.phys.vel.x, ripple_car.phys.vel.y,
                    ball.phys.pos.y);
                io::stdout().flush().unwrap();
            }
            
            if res.is_terminal || res.truncated {
                let view = g.env.get_shared_info().view(&res.state);
                let scoring_team = view.scoring_team();
                let ripple_team = 1 - g.a_team; // Ripple is on opposite team
                let ball_y = res.state.ball.phys.pos.y;
                println!("[Game {}] Ripple team={}, Candidate team={}, Scorer={:?}, Ball Y={:.0}", 
                    started, ripple_team, g.a_team, scoring_team, ball_y);
                io::stdout().flush().unwrap();
                match scoring_team {
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

/// `ripple-dump`: writes full game state plus Ripple's chosen action as JSONL.
/// Python parity tool rebuilds obs from state and compares tensors + action.
#[cfg(feature = "ripple")]
fn ripple_dump(rdir: &Path, steps: usize, out: &Path) {
    use indococo_bot::ripple as rx;
    use rlgymppo::ExternalPolicy;
    init_from_default(true).expect("load collision meshes");
    let mix = indococo_bot::setters::default_mix();
    let ripple = rx::RipplePolicy::load(rdir).expect("ripple weights");
    let mut env = create_env(&RewardConfig::default(), &mix);
    let (mut state, mut obs, _) = env.reset();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("open dump"));
    let mut robs = vec![0.0f32; ripple.obs_width()];
    let mut rmask = vec![false; rx::N_ACTIONS];
    use std::io::Write;
    for step in 0..steps {
        ripple.build(&state, 0, &mut robs, &mut rmask);
        let actions = ripple.infer(&robs, &rmask, 1);
        let action = actions[0];
        
        // Extract Rust tensors
        let mut offset = 0;
        let rust_main: Vec<f32> = robs[offset..offset+32].to_vec(); offset += 32;
        let rust_others: Vec<f32> = robs[offset..offset+5*21].to_vec(); offset += 5*21;
        let rust_ball: Vec<f32> = robs[offset..offset+6*9].to_vec(); offset += 6*9;
        let rust_actions: Vec<f32> = robs[offset..offset+6*8].to_vec(); offset += 6*8;
        let rust_boosts: Vec<f32> = robs[offset..offset+34*5].to_vec();
        
        // Serialize full game state
        let cars: Vec<_> = state.cars.iter().map(|(ci, c)| serde_json::json!({
            "team": ci.team as u8,
            "pos": [c.phys.pos.x, c.phys.pos.y, c.phys.pos.z],
            "vel": [c.phys.vel.x, c.phys.vel.y, c.phys.vel.z],
            "ang_vel": [c.phys.ang_vel.x, c.phys.ang_vel.y, c.phys.ang_vel.z],
            "rot_mat": [
                [c.phys.rot_mat.x_axis.x, c.phys.rot_mat.x_axis.y, c.phys.rot_mat.x_axis.z],
                [c.phys.rot_mat.y_axis.x, c.phys.rot_mat.y_axis.y, c.phys.rot_mat.y_axis.z],
                [c.phys.rot_mat.z_axis.x, c.phys.rot_mat.z_axis.y, c.phys.rot_mat.z_axis.z],
            ],
            "boost": c.boost,
            "on_ground": c.is_on_ground(),
            "has_flipped": c.has_flipped,
            "has_double_jumped": c.has_double_jumped,
            "has_jumped": c.has_jumped,
            "air_time_since_jump": c.air_time_since_jump,
            "is_demoed": c.is_demoed,
            "controls": [c.controls.throttle, c.controls.steer, c.controls.pitch, c.controls.yaw,
                c.controls.roll, f32::from(c.controls.jump), f32::from(c.controls.boost), f32::from(c.controls.handbrake)],
        })).collect();
        
        let pads: Vec<_> = state.boost_pads.iter().map(|(_, ps)| serde_json::json!({
            "is_active": ps.is_active(),
            "cooldown": ps.cooldown,
        })).collect();
        
        let line = serde_json::json!({
            "step": step,
            "tick": state.tick_count,
            "ball": {
                "pos": [state.ball.phys.pos.x, state.ball.phys.pos.y, state.ball.phys.pos.z],
                "vel": [state.ball.phys.vel.x, state.ball.phys.vel.y, state.ball.phys.vel.z],
                "ang_vel": [state.ball.phys.ang_vel.x, state.ball.phys.ang_vel.y, state.ball.phys.ang_vel.z],
            },
            "cars": cars,
            "pads": pads,
            "rust_main": rust_main,
            "rust_others": rust_others,
            "rust_ball": rust_ball,
            "rust_actions": rust_actions,
            "rust_boosts": rust_boosts,
            "rust_action": action,
        });
        writeln!(f, "{}", serde_json::to_string(&line).unwrap()).expect("write");
        
        // Apply Ripple's controls directly and step
        let controls = rx::controls_of(action);
        let num_cars = state.cars.len();
        let acts = vec![0; num_cars]; // Placeholder actions for all cars
        env.pre_step(&state, &acts);
        for _ in 0..2 {
            env.arena.set_car_controls(0, controls);
            env.step_physics(4);
        }
        let res = env.post_step();
        state = res.state;
        obs = res.obs;
    }
    println!("Wrote {steps} steps to {}", out.display());
}

#[cfg(not(feature = "ripple"))]
fn ripple_dump(_rdir: &Path, _steps: usize, _out: &Path) {
    panic!("ripple-dump requires --features ripple");
}


// ---------------------------------------------------------------------------
// TM8 AUDIT (scratch only): `reward-audit <reward.toml> <setters.toml> <ckpt>
// <steps> [threads]`. Self-play of <ckpt> (both teams, actions SAMPLED from
// the policy softmax as in training rollouts) over the setter mix of
// <setters.toml> (a train_tm.toml works: only [[state_setters]] is read),
// with the run's real reward config. The learner step for team-spirit /
// ramp schedules is AUDIT_STEP (default 1e12 = every ramp at its end).
// Prints the per (team size, role, phase, term) sums from
// `rewards::combined::audit` as JSON.
fn audit_step() -> u64 {
    std::env::var("AUDIT_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000_000_000)
}

fn sample_actions(actor: &Net, obs: &[&Vec<f32>], dev: &Device, rng: &mut SmallRng) -> Vec<usize> {
    let flat: Vec<f32> = obs.iter().flat_map(|o| o.iter().copied()).collect();
    let x = Tensor::<2>::from_data(TensorData::new(flat, [obs.len(), obs_size()]), dev);
    let logits = actor.forward(x).into_data().try_to_vec::<f32>().expect("logits");
    logits
        .chunks(N_ACTIONS)
        .map(|row| {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f64> = row.iter().map(|&l| f64::from(l - m).exp()).collect();
            let z: f64 = p.iter().sum();
            let mut u = rng.random::<f64>() * z;
            for (k, &pk) in p.iter().enumerate() {
                u -= pk;
                if u <= 0.0 {
                    return k;
                }
            }
            N_ACTIONS - 1
        })
        .collect()
}

static BEH: std::sync::Mutex<[f64; 8]> = std::sync::Mutex::new([0.0; 8]);

fn reward_audit(reward_toml: &Path, setters_toml: &Path, ckpt: &Path, steps: usize, threads: usize) {
    #[derive(Deserialize)]
    struct Mix {
        state_setters: Vec<WeightedSetter>,
    }
    init_from_default(true).expect("load collision meshes");
    indococo_bot::team::set_step_source(audit_step);
    indococo_bot::obs::set_obs_shuffle(true);
    indococo_bot::rewards::combined::audit::ON.store(true, std::sync::atomic::Ordering::Relaxed);
    let cfg = RewardConfig::from_file(reward_toml).expect("reward config");
    let text = std::fs::read_to_string(setters_toml).expect("read setters");
    let mix: Mix = toml::from_str(&text).expect("parse setters");
    let per = steps.div_ceil(threads);
    let t0 = std::time::Instant::now();
    let res: Vec<[u64; 4]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (cfg, mix) = (&cfg, &mix.state_setters);
                s.spawn(move || {
                    const ENVS: usize = 8;
                    let dev = Device::flex();
                    let na = load_actor(ckpt, &dev);
                    let mut rng = SmallRng::seed_from_u64(4242 + t as u64);
                    let mut envs: Vec<(BotEnv, GameState, Vec<Vec<f32>>)> = (0..ENVS)
                        .map(|_| {
                            let mut env = create_env(cfg, mix);
                            let (st, obs, _) = env.reset();
                            (env, st, obs)
                        })
                        .collect();
                    let mut out = [0u64; 4];
                    // TM8 passivity metrics: [car-steps, touches, team-steps, deepest-car depth sum,
                    // closest-car ball dist sum, pair800 team-steps, double-commit team-steps, live steps]
                    let mut beh = [0f64; 8];
                    let mut done_steps = 0usize;
                    while done_steps < per {
                        let all: Vec<&Vec<f32>> = envs.iter().flat_map(|e| e.2.iter()).collect();
                        let acts = sample_actions(&na, &all, &dev, &mut rng);
                        let mut k = 0;
                        for e in envs.iter_mut() {
                            let n = e.2.len();
                            let a = acts[k..k + n].to_vec();
                            k += n;
                            let r = e.0.step(&e.1, &a);
                            out[0] += 1;
                            {
                                let v = indococo_bot::state::StateView::from_game_state(&r.state);
                                if !indococo_bot::team::kickoff_pending(&v) {
                                    beh[7] += 1.0;
                                    for c in &v.cars {
                                        beh[0] += 1.0;
                                        if c.ball_touches > 0 { beh[1] += 1.0; }
                                    }
                                    for team in [0u8, 1u8] {
                                        let ids: Vec<usize> = (0..v.cars.len()).filter(|&i| v.cars[i].team == team && !v.cars[i].is_demoed()).collect();
                                        if ids.is_empty() { continue; }
                                        beh[2] += 1.0;
                                        let depth = ids.iter().map(|&i| v.cars[i].attack_sign() * v.cars[i].pos.y + 5120.0).fold(f64::INFINITY, f64::min);
                                        beh[3] += depth;
                                        let bd = ids.iter().map(|&i| v.cars[i].pos.distance(v.ball.pos)).fold(f64::INFINITY, f64::min);
                                        beh[4] += bd;
                                        let mut p800 = false;
                                        let mut dc = 0;
                                        for (k, &i) in ids.iter().enumerate() {
                                            let c = &v.cars[i];
                                            let d = v.ball.pos - c.pos;
                                            let dist = d.length();
                                            if dist < 1000.0 && dist > 1e-6 && c.vel.dot(d / dist) > 500.0 { dc += 1; }
                                            for &j in &ids[k + 1..] {
                                                if c.pos.distance(v.cars[j].pos) < 800.0 { p800 = true; }
                                            }
                                        }
                                        if p800 { beh[5] += 1.0; }
                                        if dc >= 2 { beh[6] += 1.0; }
                                    }
                                }
                            }
                            if r.is_terminal || r.truncated {
                                let v = e.0.get_shared_info().view(&r.state);
                                if v.scoring_team().is_some() { out[1] += 1; } else { out[2] += 1; }
                                let (st, obs, _) = e.0.reset();
                                e.1 = st;
                                e.2 = obs;
                            } else {
                                e.1 = r.state;
                                e.2 = r.obs;
                            }
                        }
                        done_steps += ENVS;
                    }
                    for (k, b) in beh.iter().enumerate() {
                        BEH.lock().unwrap()[k] += b;
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = [0u64; 4];
    for r in &res { for i in 0..4 { tot[i] += r[i]; } }
    let agg = indococo_bot::rewards::combined::audit::AGG.lock().unwrap();
    let flags = indococo_bot::rewards::combined::audit::FLAGS.lock().unwrap();
    let rows: Vec<serde_json::Value> = agg.iter().map(|((n, role, phase, term), v)| serde_json::json!({
        "n": n, "role": role, "phase": phase, "term": term,
        "cnt": v[0], "sum_pre": v[1], "abs_pre": v[2], "sum_post": v[3], "abs_post": v[4], "absdev_post": v[5],
    })).collect();
    let fl: serde_json::Map<String, serde_json::Value> = flags.iter().map(|(k, (i, f))| (k.clone(), serde_json::json!({"individual": i, "full": f}))).collect();
    let b = *BEH.lock().unwrap();
    let behaviour = serde_json::json!({
        "touches_per_car_min": b[1] / b[0].max(1.0) * indococo_bot::consts::STEPS_PER_SECOND * 60.0,
        "deepest_car_depth_uu": b[3] / b[2].max(1.0),
        "closest_car_ball_dist_uu": b[4] / b[2].max(1.0),
        "pair800": b[5] / b[2].max(1.0),
        "double_commit": b[6] / b[2].max(1.0),
        "live_steps": b[7],
    });
    let out = serde_json::json!({
        "reward": reward_toml, "setters": setters_toml, "ckpt": ckpt, "team_size": indococo_bot::obs::team_size(),
        "audit_step": audit_step(), "env_steps": tot[0], "goals": tot[1], "timeouts": tot[2],
        "seconds": t0.elapsed().as_secs_f64(), "flags": fl, "rows": rows,
        "behaviour": behaviour,
    });
    println!("{out}");
}
