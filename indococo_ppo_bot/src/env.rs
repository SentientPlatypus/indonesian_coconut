//! Environment wiring: arena, state setters, terminals and shared info.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
use rlgymppo_utils::rlgym::rocketsim::{Arena, ArenaEvent, CarBodyConfig, GameMode, Team};
use rlgymppo_utils::rlgym::{Env, GameState, SharedInfoProvider};
use rlgymppo_utils::shared_info::{SharedInfoReport, SharedInfoRng};
use rlgymppo_utils::state_setters::{KickoffState, RandomState, WeightedState};
use rlgymppo_utils::terminal::{AnyTerminal, GameEndedCondition, NoTouchCondition, OnGoalCondition};
use rlgymppo_utils::{AvgTracker, Report, any_terminal, weighted_state};

use crate::action::LookupNoMask;
use crate::obs::V2DefaultObs;
use crate::rewards::{CombinedReward, RewardConfig};

/// `NO_TOUCH_TIMEOUT_S = 30` (truncation).
pub const NO_TOUCH_TICKS: u64 = 30 * 120;
/// `GAME_TIMEOUT_S = 300`.
pub const GAME_TICKS: u64 = 300 * 120;

/// Per-environment RNG and metrics.
pub struct SharedInfo {
    rng: SmallRng,
    metrics: Report,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self::from(rng().next_u64())
    }
}

impl From<u64> for SharedInfo {
    fn from(seed: u64) -> Self {
        Self { rng: SmallRng::seed_from_u64(seed), metrics: Report::default() }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, _state: &GameState) {}

    fn update(&mut self, gs: &GameState) {
        for (info, car) in &gs.cars {
            self.metrics["Player/Distance to ball"] += AvgTracker::from(car.phys.pos.distance(gs.ball.pos));
            self.metrics["Player/In Air Ratio"] += AvgTracker::from(!car.is_on_ground());
            self.metrics["Player/Speed"] += AvgTracker::from(car.phys.vel.length());
            self.metrics["Player/Boost"] += AvgTracker::from(car.boost);
            let touched = gs.events.iter().any(|e| matches!(e, ArenaEvent::CarHitBall(h) if h.car_idx == info.idx));
            self.metrics["Player/Ball Touch Ratio"] += AvgTracker::from(touched);
        }
    }
}

impl SharedInfoRng for SharedInfo {
    type Rng = SmallRng;

    fn rng(&mut self) -> &mut SmallRng {
        &mut self.rng
    }
}

impl SharedInfoReport for SharedInfo {
    fn report(&mut self) -> &mut Report {
        &mut self.metrics
    }
}

/// The training environment type.
pub type BotEnv = Env<
    WeightedState<SharedInfo>,
    V2DefaultObs,
    LookupNoMask,
    CombinedReward,
    AnyTerminal<SharedInfo>,
    NoTouchCondition<NO_TOUCH_TICKS>,
    SharedInfo,
>;

/// A 1v1 Octane arena with the configured rewards.
///
/// Resets are 34% kickoffs and 66% randomized states. The Python mechanic
/// curriculum (`curriculum_mutators.py`) is not ported; the random states
/// stand in for its mechanic setups.
#[must_use]
pub fn create_env(rewards: &RewardConfig) -> BotEnv {
    let mut arena = Arena::new(GameMode::Soccar);
    arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    Env::new(
        arena,
        weighted_state![
            KickoffState, 0.34;
            RandomState<true, true, false>, 0.33;
            RandomState<true, false, true>, 0.33;
        ],
        V2DefaultObs,
        LookupNoMask::default(),
        rewards.build(),
        any_terminal![OnGoalCondition, GameEndedCondition<GAME_TICKS>],
        NoTouchCondition::default(),
        SharedInfo::default(),
    )
}
