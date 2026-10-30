//! Environment wiring: arena, state setters, terminals and shared info.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
use rlgymppo_utils::rlgym::rocketsim::{Arena, ArenaEvent, CarBodyConfig, GameMode, Team};
use rlgymppo_utils::rlgym::{Env, GameState, SharedInfoProvider};
use rlgymppo_utils::shared_info::{SharedInfoReport, SharedInfoRng};
use rlgymppo_utils::state_setters::WeightedState;
use rlgymppo_utils::terminal::{
    AnyTerminal, GameEndedCondition, NoTouchCondition, OnGoalCondition,
};
use rlgymppo_utils::{AvgTracker, Report, any_terminal};

use crate::action::LookupNoMask;
use crate::events::StepEvents;
use crate::obs::V2DefaultObs;
use crate::rewards::{CombinedReward, RewardConfig};
use crate::setters::{WeightedSetter, build_mix};
use crate::state::StateView;

/// `NO_TOUCH_TIMEOUT_S = 30` (truncation).
pub const NO_TOUCH_TICKS: u64 = 30 * 120;
/// `GAME_TIMEOUT_S = 300`.
pub const GAME_TICKS: u64 = 300 * 120;

/// Steps per minute of game time (120 Hz physics, 8-tick action repeat).
const STEPS_PER_MINUTE: f64 = 120.0 / 8.0 * 60.0;

/// Per-environment RNG and metrics.
pub struct SharedInfo {
    rng: SmallRng,
    metrics: Report,
    /// Previous step, for the event counters.
    prev: Option<StateView>,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self::from(rng().next_u64())
    }
}

impl From<u64> for SharedInfo {
    fn from(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            metrics: Report::default(),
            prev: None,
        }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, state: &GameState) {
        self.prev = Some(StateView::from_game_state(state));
    }

    fn update(&mut self, gs: &GameState) {
        for (info, car) in &gs.cars {
            self.metrics["Player/Distance to ball"] +=
                AvgTracker::from(car.phys.pos.distance(gs.ball.pos));
            self.metrics["Player/In Air Ratio"] += AvgTracker::from(!car.is_on_ground());
            self.metrics["Player/Speed"] += AvgTracker::from(car.phys.vel.length());
            self.metrics["Player/Boost"] += AvgTracker::from(car.boost);
            let touched = gs
                .events
                .iter()
                .any(|e| matches!(e, ArenaEvent::CarHitBall(h) if h.car_idx == info.idx));
            self.metrics["Player/Ball Touch Ratio"] += AvgTracker::from(touched);
        }
        self.count_events(gs);
    }
}

impl SharedInfo {
    /// `Events/*`: share of car-steps with the flip available, and flip
    /// resets, bumps and demos per car per minute of game time.
    fn count_events(&mut self, gs: &GameState) {
        let cur = StateView::from_game_state(gs);
        if let Some(prev) = self
            .prev
            .as_ref()
            .filter(|p| p.cars.len() == cur.cars.len())
        {
            let per_min = |hit: bool| AvgTracker::from(if hit { STEPS_PER_MINUTE } else { 0.0 });
            for e in StepEvents::between(prev, &cur) {
                self.metrics["Events/Has Flip Ratio"] += AvgTracker::from(e.has_flip);
                self.metrics["Events/Flip Resets per min"] += per_min(e.flip_reset);
                self.metrics["Events/Bumps per min"] += per_min(e.bump);
                self.metrics["Events/Demos per min"] += per_min(e.demo);
            }
        }
        self.prev = Some(cur);
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

/// A 1v1 Octane arena with the configured rewards and reset mix
/// (see [`crate::setters`]; [`crate::setters::default_mix`] is the old fixed one).
#[must_use]
pub fn create_env(rewards: &RewardConfig, setters: &[WeightedSetter]) -> BotEnv {
    let mut arena = Arena::new(GameMode::Soccar);
    arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    Env::new(
        arena,
        build_mix(setters),
        V2DefaultObs,
        LookupNoMask::default(),
        rewards.build(),
        any_terminal![OnGoalCondition, GameEndedCondition<GAME_TICKS>],
        NoTouchCondition::default(),
        SharedInfo::default(),
    )
}
