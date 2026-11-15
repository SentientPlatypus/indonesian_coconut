//! RLBot v5 agent: the indococo policies on Burn Flex (CPU).
//!
//! Which model plays is picked per match from the match's team sizes:
//!
//! * 1v1: `--1v1 <model>` (or `INDOCOCO_1V1_MODEL`), default `st25`.
//! * 2v2 / 3v3 (any team with more than one car): `--team <model>` (or
//!   `INDOCOCO_TEAM_MODEL`), default `tm2`.
//!
//! A model name is a directory under `weights/` next to the executable
//! holding `actor.bpk` (`weights/st25`, `weights/tm2`); anything containing
//! a path separator is taken as a checkpoint directory path instead.
//! `INDOCOCO_WEIGHTS` (a checkpoint directory) overrides both choices.
//!
//! The obs mode follows the chosen actor's input width: 92 base, 106 FR8
//! extended (ST runs), 235 TM1/TM2 team-ext (any team size, 1v1 included),
//! 270 TM3 team-ext + ball prediction.
//!
//! `--selftest <1|2|3>` runs the same model choice offline for that many
//! cars per team: a RocketSim match where every car is driven by the
//! policy, printing the model, obs width and an action summary.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use burn::tensor::Device;
use indococo_bot::LAYERS;
use indococo_bot::action::LookupNoMask;
use indococo_bot::env::{SharedInfo, create_env_sized};
use indococo_bot::obs::{self, V2DefaultObs};
use indococo_bot::rewards::RewardConfig;
use indococo_bot::setters::default_mix;
use rlbot_rocketsim::rlbot::RLBotConnection;
use rlbot_rocketsim::rlbot::agents::{BotAgent, run_bot_agents};
use rlbot_rocketsim::rlbot::flat::{ControllableInfo, FieldInfo, GamePacket, MatchConfiguration};
use rlbot_rocketsim::rlbot::util::{AgentEnvironment, PacketQueue};
use rlgymppo_model::{NormSelection, PolicyConfig};
use rlgymppo_rlbot::{ConfigurablePpoBot, PpoBotConfig, RlbotAction};
use rlgymppo_utils::rlgym::{Action as _, Obs as _};
use rlgymppo_utils::rocketsim::{CarControls, GameMode, init_from_mem};
use rustc_hash::FxHashMap;

/// Default 1v1 model (`weights/st25`: ST25 20728M, 106-wide 1v1 obs).
const DEFAULT_1V1: &str = "st25";
/// Default 2v2 / 3v3 model (`weights/tm2`: TM2, 235-wide team-ext obs).
const DEFAULT_TEAM: &str = "tm2";

/// Checkpoint directory picked for this process (set once, before loading).
static CHECKPOINT: OnceLock<PathBuf> = OnceLock::new();

/// Python `DiscreteFF` architecture: no shared head, no normalization.
struct IndococoConfig;

impl PpoBotConfig for IndococoConfig {
    fn policy_config(input_size: usize, action_size: usize) -> PolicyConfig {
        PolicyConfig {
            input_size,
            action_size,
            actor_layer_sizes: LAYERS.to_vec(),
            shared_head_layer_sizes: vec![],
            norm: NormSelection::None,
        }
    }

    fn checkpoint_path() -> PathBuf {
        CHECKPOINT.get().cloned().expect("model chosen before the policy loads")
    }
}

/// Newtype so the RLBot crate's trait can be implemented here.
#[derive(Default)]
struct Actions(LookupNoMask);

impl RlbotAction for Actions {
    fn get_action(&self, action_index: usize) -> CarControls {
        self.0.controls(action_index)
    }
}

impl<SI> rlgymppo_utils::rlgym::Action<SI> for Actions
where
    LookupNoMask: rlgymppo_utils::rlgym::Action<SI>,
{
    type Input = <LookupNoMask as rlgymppo_utils::rlgym::Action<SI>>::Input;

    fn get_tick_skip() -> u8 {
        <LookupNoMask as rlgymppo_utils::rlgym::Action<SI>>::get_tick_skip()
    }

    fn get_action_delay() -> u8 {
        <LookupNoMask as rlgymppo_utils::rlgym::Action<SI>>::get_action_delay()
    }

    fn get_action_space(&self, si: &SI) -> usize {
        self.0.get_action_space(si)
    }

    fn reset(&mut self, state: &rlgymppo_utils::rlgym::GameState, si: &mut SI) {
        self.0.reset(state, si);
    }

    fn parse_actions<'a>(
        &'a mut self,
        actions: &[Self::Input],
        state: &rlgymppo_utils::rlgym::GameState,
        si: &'a mut SI,
    ) -> &'a [(usize, CarControls)] {
        self.0.parse_actions(actions, state, si)
    }

    fn get_action_masks(
        &mut self,
        state: &rlgymppo_utils::rlgym::GameState,
        si: &mut SI,
    ) -> Vec<Vec<bool>> {
        self.0.get_action_masks(state, si)
    }
}

macro_rules! meshes {
    ($($n:literal),*) => {
        [$(include_bytes!(concat!("../../../collision_meshes/soccar/mesh_", $n, ".cmf")).as_slice()),*]
    };
}

const SOCCAR_MESHES: [&[u8]; 16] = meshes!(
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"
);

/// Command line / environment model choice.
struct Choice {
    one: String,
    team: String,
    selftest: Option<usize>,
}

fn usage() -> ! {
    eprintln!(
        "usage: indococo-rlbot [--1v1 st25|tm2|<dir>] [--team tm2|<dir>] [--selftest 1|2|3]\n\
         env: INDOCOCO_1V1_MODEL, INDOCOCO_TEAM_MODEL, INDOCOCO_WEIGHTS (overrides both)\n\
         defaults: --1v1 {DEFAULT_1V1} --team {DEFAULT_TEAM}"
    );
    std::process::exit(2)
}

fn parse_args() -> Choice {
    let env = |k: &str, d: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| d.into());
    let mut c = Choice {
        one: env("INDOCOCO_1V1_MODEL", DEFAULT_1V1),
        team: env("INDOCOCO_TEAM_MODEL", DEFAULT_TEAM),
        selftest: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let (key, inline) = match a.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (a.clone(), None),
        };
        let mut val = || inline.clone().or_else(|| args.next()).unwrap_or_else(|| usage());
        match key.as_str() {
            "--1v1" | "--1v1-model" | "--duel" => c.one = val(),
            "--team" | "--team-model" => c.team = val(),
            "--selftest" => {
                c.selftest = Some(val().parse().ok().filter(|n| (1..=3).contains(n)).unwrap_or_else(|| usage()));
            }
            "-h" | "--help" => usage(),
            _ => eprintln!("indococo-rlbot: ignoring unknown argument {a:?}"),
        }
    }
    c
}

static CHOICE: OnceLock<Choice> = OnceLock::new();

/// `weights/` next to the executable.
fn weights_root() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("weights")))
        .unwrap_or_else(|| PathBuf::from("weights"))
}

/// Checkpoint directory for `per_team` cars per team.
fn model_dir(per_team: usize) -> PathBuf {
    if let Some(path) = std::env::var_os("INDOCOCO_WEIGHTS") {
        return PathBuf::from(path);
    }
    let c = CHOICE.get().expect("args parsed");
    let spec = if per_team > 1 { &c.team } else { &c.one };
    if spec.contains('/') || spec.contains('\\') {
        PathBuf::from(spec)
    } else {
        weights_root().join(spec.to_ascii_lowercase())
    }
}

/// Input width of `<dir>/actor.bpk`.
fn actor_input_width(dir: &std::path::Path) -> usize {
    use burn_store::{BurnpackStore, ModuleStore};
    let path = dir.join("actor.bpk");
    assert!(
        path.is_file(),
        "{} not found (models live in weights/<name>/actor.bpk; see README)",
        path.display()
    );
    let mut store = BurnpackStore::from_file(&path);
    store
        .get_tensor("linear_layers.0.weight")
        .unwrap_or_else(|e| panic!("read {}: {e:?}", path.display()))
        .unwrap_or_else(|| panic!("{}: no linear_layers.0.weight", path.display()))
        .shape[0]
}

/// Picks the model for `per_team` cars per team and fixes the obs mode to
/// its input width. Returns the width.
fn choose_model(per_team: usize) -> usize {
    let dir = model_dir(per_team);
    let width = actor_input_width(&dir);
    // TM1/TM2 team-ext nets (235 inputs) serve 1v1, 2v2 and 3v3; TM3 nets
    // (270) add the ball prediction + nearest big pad, rolled out at runtime
    // on a ball-only arena from the embedded meshes.
    let pred = width == obs::PRED_OBS_SIZE;
    let team = pred || width == obs::TEAM_OBS_SIZE;
    let ext = if team || width == obs::ext_obs_size_for(1) {
        true
    } else if width == obs::obs_size_for(1) {
        false
    } else {
        panic!(
            "{}: actor input width {width} is no known obs (92, 106, {}, {})",
            dir.display(),
            obs::TEAM_OBS_SIZE,
            obs::PRED_OBS_SIZE
        );
    };
    assert!(
        per_team == 1 || team,
        "{}: a {width}-wide 1v1 net cannot play {per_team}v{per_team}; use a team model (--team tm2)",
        dir.display()
    );
    assert!(obs::set_obs_team(team), "obs team mode already fixed");
    assert!(obs::set_obs_pred(pred), "obs prediction mode already fixed");
    assert!(obs::set_obs_ext(ext), "obs mode already fixed");
    let first = CHECKPOINT.get().is_none();
    let fixed = CHECKPOINT.get_or_init(|| dir.clone());
    assert!(fixed == &dir, "model already fixed to {} (wanted {})", fixed.display(), dir.display());
    if first {
        eprintln!(
            "indococo-rlbot: {per_team}v{per_team} -> {} (actor input {width}, extended obs {ext}, team obs {team}, prediction obs {pred})",
            dir.display()
        );
    }
    width
}

type Inner = ConfigurablePpoBot<V2DefaultObs, Actions, SharedInfo, IndococoConfig>;

/// Picks the model from the match's team sizes, then runs the PPO bot.
struct Bot(Inner);

impl BotAgent for Bot {
    fn new(
        team: u32,
        controllable_info: ControllableInfo,
        match_config: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        packet_queue: &mut PacketQueue,
    ) -> Self {
        let mut per_team = [0usize; 2];
        for p in &match_config.player_configurations {
            if let Some(n) = per_team.get_mut(p.team as usize) {
                *n += 1;
            }
        }
        let n = per_team[0].max(per_team[1]).max(1);
        choose_model(n);
        Self(Inner::new(team, controllable_info, match_config, field_info, packet_queue))
    }

    fn tick(&mut self, packet: &GamePacket, packet_queue: &mut PacketQueue) {
        self.0.tick(packet, packet_queue);
    }
}

fn init_meshes() {
    let meshes = FxHashMap::from_iter([(
        GameMode::Soccar,
        SOCCAR_MESHES.iter().map(|m| m.to_vec()).collect::<Vec<_>>(),
    )]);
    init_from_mem(meshes, true).expect("load embedded collision meshes");
}

/// Offline check: the chosen model drives every car of a `per_team`v`per_team`
/// RocketSim match (kickoff and other default setters) for 1200 decisions.
fn selftest(per_team: usize) {
    let width = choose_model(per_team);
    init_meshes();
    let mut env = create_env_sized(&RewardConfig::default(), &default_mix(), per_team);
    let mut si = SharedInfo::default();
    let lookup = LookupNoMask::default();
    let obs_width = V2DefaultObs::default().get_obs_space(&si);
    assert_eq!(obs_width, width, "obs builder width differs from the model");
    let n_actions = lookup.get_action_space(&si);
    let device = Device::flex();
    let policy = IndococoConfig::load_policy(width, n_actions, &device);
    let (mut state, mut o, _) = env.reset();
    assert_eq!(o.len(), 2 * per_team);
    let idx: Vec<usize> = (0..o.len()).collect();
    let mut lookup_m = LookupNoMask::default();
    let mut hist = vec![0usize; n_actions];
    let (mut decisions, mut episodes, mut ms) = (0usize, 1usize, 0f64);
    for _ in 0..1200 {
        assert!(o.iter().all(|r| r.len() == width && r.iter().all(|x| x.is_finite())));
        let masks = lookup_m.get_action_masks(&state, &mut si);
        let t = std::time::Instant::now();
        let actions = policy.react_deterministic_indexed(&o, &masks, &idx, &device);
        ms += t.elapsed().as_secs_f64() * 1e3;
        for &a in &actions {
            hist[a] += 1;
        }
        decisions += actions.len();
        let res = env.step(&state, &actions);
        let done = res.is_terminal || res.truncated;
        (state, o) = (res.state, res.obs);
        if done {
            let (s, oo, _) = env.reset();
            (state, o) = (s, oo);
            episodes += 1;
        }
    }
    let distinct = hist.iter().filter(|&&c| c > 0).count();
    let top = hist.iter().enumerate().max_by_key(|(_, c)| **c).map(|(a, c)| (a, *c)).unwrap();
    println!(
        "selftest {per_team}v{per_team}: model {} width {width}, {decisions} decisions over {episodes} episodes, \
         {distinct}/{n_actions} distinct actions (top {} x{}), {:.2} ms/inference batch -> OK",
        CHECKPOINT.get().unwrap().display(),
        top.0,
        top.1,
        ms / 1200.0
    );
}

fn main() {
    let choice = parse_args();
    let selftest_n = choice.selftest;
    assert!(CHOICE.set(choice).is_ok());
    if let Some(n) = selftest_n {
        selftest(n);
        return;
    }
    init_meshes();
    let AgentEnvironment {
        server_addr,
        agent_id,
    } = AgentEnvironment::from_env();
    let agent_id = agent_id.unwrap_or_else(|| "indonesian_coconut/indococo-ppo".into());
    let connection = RLBotConnection::new(&server_addr).expect("connect to RLBot");
    run_bot_agents::<Bot>(agent_id, false, false, connection).expect("run RLBot agent");
}
