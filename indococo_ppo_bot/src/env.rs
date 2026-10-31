//! Environment wiring: arena, state setters, terminals and shared info.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
use rlgymppo_utils::rlgym::rocketsim::{Arena, ArenaEvent, CarBodyConfig, GameMode, Team};
use rlgymppo_utils::rlgym::{Env, GameState, SharedInfoProvider};
use rlgymppo_utils::shared_info::{SharedInfoReport, SharedInfoRng};
use rlgymppo_utils::terminal::{
    AnyTerminal, GameEndedCondition, NoTouchCondition, OnGoalCondition,
};
use rlgymppo_utils::{AvgTracker, Report, any_terminal};

use crate::action::LookupNoMask;
use crate::curriculum::{self, Attempt, CurriculumLog};
use crate::events::EventTracker;
use crate::obs::V2DefaultObs;
use crate::resets::TickResetTracker;
use crate::rewards::{CombinedReward, RewardConfig};
use crate::setters::{SetterLog, SetterMix, WeightedSetter, build_mix};
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
    events: EventTracker,
    resets: TickResetTracker,
    /// Setter that started the current episode.
    setter: &'static str,
    /// Curriculum episode still being decided.
    attempt: Option<Attempt>,
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
            events: EventTracker::default(),
            resets: TickResetTracker::default(),
            setter: "",
            attempt: None,
        }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, state: &GameState) {
        self.resets.reset(state);
        self.events.reset(StateView::from_game_state(state));
    }

    fn on_tick(&mut self, arena: &Arena, _tick_events: &[ArenaEvent]) {
        self.resets.on_tick(arena);
    }

    /// Runs before the rewards of the step (rlgym_rs `post_step` order).
    fn update(&mut self, gs: &GameState) {
        self.resets.finish_step();
        if let Some(mut a) = self.attempt {
            let reset = self
                .resets
                .last_step()
                .get(a.flier)
                .copied()
                .unwrap_or(false);
            match a.step(reset) {
                Some(success) => self.finish_attempt(a, success),
                None => self.attempt = Some(a),
            }
        }
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
    /// resets, bumps, demos and air dribble bumps per car per minute.
    fn count_events(&mut self, gs: &GameState) {
        let per_min = |hit: bool| AvgTracker::from(if hit { STEPS_PER_MINUTE } else { 0.0 });
        let view = self.view(gs);
        for e in self.events.step(view) {
            self.metrics["Events/Has Flip Ratio"] += AvgTracker::from(e.has_flip);
            self.metrics["Events/Flip Resets per min"] += per_min(e.flip_reset);
            self.metrics["Events/Flip Resets (old detector) per min"] +=
                per_min(e.flip_reset_legacy);
            self.metrics[format!("Events/Flip Resets per min @{}", self.setter).as_str()] +=
                per_min(e.flip_reset);
            self.metrics["Events/Bumps per min"] += per_min(e.bump);
            self.metrics["Events/Demos per min"] += per_min(e.demo);
            self.metrics["Events/Air Dribble Bumps per min"] += per_min(e.air_dribble_bump);
        }
    }
}

impl SharedInfo {
    /// Reports a decided curriculum episode and logs it.
    fn finish_attempt(&mut self, a: Attempt, success: bool) {
        self.attempt = None;
        let level = curriculum::record(a.level, success);
        self.metrics["Curriculum/level"] += AvgTracker::from(level as f64);
        self.metrics["Curriculum/success"] += AvgTracker::from(success);
        self.metrics[format!("Curriculum/success @L{}", a.level).as_str()] +=
            AvgTracker::from(success);
    }
}

impl CurriculumLog for SharedInfo {
    fn start_curriculum(
        &mut self,
        level: usize,
        flier: usize,
        min_control_s: f32,
        deadline_s: f32,
    ) {
        self.attempt = Some(Attempt::new(level, flier, min_control_s, deadline_s));
        self.metrics["Curriculum/episode level"] += AvgTracker::from(level as f64);
    }
}

impl SetterLog for SharedInfo {
    fn record_setter(&mut self, chosen: &'static str, labels: &[&'static str]) {
        // An episode that ended before its curriculum attempt was decided
        // (goal, timeout) is a failure.
        if let Some(a) = self.attempt {
            self.finish_attempt(a, false);
        }
        self.setter = chosen;
        for &label in labels {
            self.metrics[format!("Setters/{label} share").as_str()] +=
                AvgTracker::from(label == chosen);
        }
    }
}

/// Shared info that can hand rewards the step's per-tick flip resets.
pub trait TickResets {
    /// Which cars (car order) got a flip reset during the last step.
    fn tick_resets(&self) -> &[bool];
    /// Setter kind that started the current episode.
    fn setter_kind(&self) -> &'static str;
}

impl TickResets for SharedInfo {
    fn tick_resets(&self) -> &[bool] {
        self.resets.last_step()
    }

    fn setter_kind(&self) -> &'static str {
        self.setter
    }
}

impl SharedInfo {
    /// Setter that started the current episode (its `kind`).
    #[must_use]
    pub fn setter(&self) -> &'static str {
        self.setter
    }

    /// The step's [`StateView`] with the per-tick flip resets filled in.
    #[must_use]
    pub fn view(&self, gs: &GameState) -> StateView {
        StateView::from_game_state(gs).with_tick_resets(self.resets.last_step())
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
    SetterMix<SharedInfo>,
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
