//! RLBot v5 agent: the 1v1 policy on Burn Flex (CPU).
//!
//! Weights are read from `INDOCOCO_WEIGHTS` if set, otherwise from the
//! `weights/` directory next to the executable (`actor.bpk`).

use std::path::PathBuf;

use indococo_bot::LAYERS;
use indococo_bot::action::LookupNoMask;
use indococo_bot::env::SharedInfo;
use indococo_bot::obs::V2DefaultObs;
use rlbot_rocketsim::rlbot::RLBotConnection;
use rlbot_rocketsim::rlbot::agents::run_bot_agents;
use rlbot_rocketsim::rlbot::util::AgentEnvironment;
use rlgymppo_model::{NormSelection, PolicyConfig};
use rlgymppo_rlbot::{ConfigurablePpoBot, PpoBotConfig, RlbotAction};
use rlgymppo_utils::rocketsim::{CarControls, GameMode, init_from_mem};
use rustc_hash::FxHashMap;

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
        if let Some(path) = std::env::var_os("INDOCOCO_WEIGHTS") {
            return PathBuf::from(path);
        }
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("weights")))
            .unwrap_or_else(|| PathBuf::from("weights"))
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

    fn get_action_masks(&mut self, state: &rlgymppo_utils::rlgym::GameState, si: &mut SI) -> Vec<Vec<bool>> {
        self.0.get_action_masks(state, si)
    }
}

type Bot = ConfigurablePpoBot<V2DefaultObs, Actions, SharedInfo, IndococoConfig>;

macro_rules! meshes {
    ($($n:literal),*) => {
        [$(include_bytes!(concat!("../../../collision_meshes/soccar/mesh_", $n, ".cmf")).as_slice()),*]
    };
}

const SOCCAR_MESHES: [&[u8]; 16] = meshes!("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");

fn main() {
    let meshes = FxHashMap::from_iter([(GameMode::Soccar, SOCCAR_MESHES.iter().map(|m| m.to_vec()).collect::<Vec<_>>())]);
    init_from_mem(meshes, true).expect("load embedded collision meshes");
    let AgentEnvironment { server_addr, agent_id } = AgentEnvironment::from_env();
    let agent_id = agent_id.unwrap_or_else(|| "indonesian_coconut/indococo-ppo".into());
    let connection = RLBotConnection::new(&server_addr).expect("connect to RLBot");
    run_bot_agents::<Bot>(agent_id, false, false, connection).expect("run RLBot agent");
}
