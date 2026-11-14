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
    /// ST35 PFSP: which rollout drives this episode and whether its goal
    /// counts for that rollout's opponent ([`crate::pfsp::EpisodeTracker`]).
    pfsp: crate::pfsp::EpisodeTracker,
    /// TM1 team metrics (`Team/*`), only with more than one car per team.
    team: crate::team::TouchTracker,
    prev_view: Option<StateView>,
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
            pfsp: crate::pfsp::EpisodeTracker::default(),
            team: crate::team::TouchTracker::default(),
            prev_view: None,
        }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, state: &GameState) {
        self.resets.reset(state);
        self.events.reset(StateView::from_game_state(state));
        self.prev_view = None;
        if state.cars.len() > 2 {
            self.team.reset(&StateView::from_game_state(state));
        }
        if crate::pfsp::enabled() {
            self.pfsp.reset(crate::setters::rollout_seq());
        }
    }

    fn on_tick(&mut self, arena: &Arena, _tick_events: &[ArenaEvent]) {
        self.resets.on_tick(arena);
    }

    /// Runs before the rewards of the step (rlgym_rs `post_step` order).
    fn update(&mut self, gs: &GameState) {
        self.resets.finish_step();
        // ST35 PFSP: a goal decides the episode for the opponent's win rate.
        if crate::pfsp::enabled() {
            let seq = crate::setters::rollout_seq();
            if let Some(team) = self.pfsp.step(seq, f64::from(gs.ball.pos.y)) {
                crate::pfsp::record_goal(team);
            }
        }
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
        let team_view = (gs.cars.len() > 2).then(|| view.clone());
        for e in self.events.step(view) {
            self.metrics["Events/Has Flip Ratio"] += AvgTracker::from(e.has_flip);
            self.metrics["Events/Flip Resets per min"] += per_min(e.flip_reset);
            self.metrics["Events/Flip Resets (old detector) per min"] +=
                per_min(e.flip_reset_legacy);
            self.metrics[format!("Events/Flip Resets per min @{}", self.setter).as_str()] +=
                per_min(e.flip_reset);
            self.metrics["Events/Bumps per min"] += per_min(e.bump);
            self.metrics["Events/Demos per min"] += per_min(e.demo);
            self.metrics["Events/Demoed per min"] += per_min(e.demoed);
            self.metrics["Events/Air Dribble Bumps per min"] += per_min(e.air_dribble_bump);
            self.metrics["Events/Goals per min"] += per_min(e.goal);
            self.metrics["Events/Aerial Goals per min"] += per_min(e.aerial_goal);
        }
        if let Some(v) = team_view {
            self.count_team(&v);
        }
    }

    /// `Team/*` (team modes): per game-minute counts and per-step ratios,
    /// also split by mode (`@2v2`, `@3v3`).
    fn count_team(&mut self, view: &StateView) {
        let per_min = |n: usize| n as f64 * STEPS_PER_MINUTE;
        let mode = format!("@{0}v{0}", view.cars.len() / 2);
        let ev = self.team.step(view);
        let put = |m: &mut Report, key: &str, v: f64| {
            m[key] += AvgTracker::from(v);
            m[format!("{key} {mode}").as_str()] += AvgTracker::from(v);
        };
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let aerial = ev.passes.iter().filter(|p| p.aerial).count();
        put(&mut self.metrics, "Team/Passes per game-min", per_min(ev.passes.len()));
        put(&mut self.metrics, "Team/Aerial passes per game-min", per_min(aerial));
        put(&mut self.metrics, "Team/Giveaways per game-min", per_min(ev.giveaways.len()));
        put(&mut self.metrics, "Team/Air double touches per game-min", per_min(ev.air_doubles.len()));
        put(&mut self.metrics, "Team/Double taps per game-min", per_min(usize::from(ev.double_tap.is_some())));
        if ev.kickoff_touched {
            put(&mut self.metrics, "Team/Kickoff duplicate share", b(ev.kickoff_duplicate.is_some()));
        }
        // Double commit: 2+ teammates within 500 uu of the ball.
        let mut bumps = 0;
        for team in [0u8, 1u8] {
            let near = view
                .cars
                .iter()
                .filter(|c| c.team == team && !c.is_demoed() && c.pos.distance(view.ball.pos) < 500.0)
                .count();
            put(&mut self.metrics, "Team/Double commit ratio", b(near >= 2));
            // Someone goal-side of the ball (between it and the own goal).
            let back = view.cars.iter().any(|c| {
                c.team == team && !c.is_demoed() && c.attack_sign() * (c.pos.y - view.ball.pos.y) < 0.0
            });
            put(&mut self.metrics, "Team/Last man back ratio", b(back));
        }
        if let Some(prev) = &self.prev_view {
            for (i, c) in view.cars.iter().enumerate() {
                if let Some(v) = c.bump_victim {
                    let new = prev.cars.get(i).is_some_and(|p| p.bump_victim != Some(v));
                    if new && view.cars.get(v).is_some_and(|o| o.team == c.team) {
                        bumps += 1;
                    }
                }
            }
        }
        put(&mut self.metrics, "Team/Teammate bumps per game-min", per_min(bumps));
        let resets = view.cars.iter().filter(|c| c.tick_flip_reset == Some(true)).count();
        put(&mut self.metrics, "Team/Flip resets per game-min", per_min(resets));
        self.prev_view = Some(view.clone());
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

    fn record_opponent_mix(&mut self, opp: Option<&str>, chosen: &'static str) {
        self.metrics["Setters/opp-conditioned share"] += AvgTracker::from(opp.is_some());
        if let Some(opp) = opp {
            self.metrics[format!("Setters/@{opp} demo_threat share").as_str()] +=
                AvgTracker::from(chosen == "demo_threat");
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

/// An Octane arena (1v1, or [`crate::obs::team_size`] per team) with the configured rewards and reset mix
/// (see [`crate::setters`]; [`crate::setters::default_mix`] is the old fixed one).
#[must_use]
pub fn create_env(rewards: &RewardConfig, setters: &[WeightedSetter]) -> BotEnv {
    create_env_sized(rewards, setters, crate::obs::team_size())
}

/// TM1 mixed team sizes: per-arena cars per team, set once by the trainer
/// (`[team] mode_mix`). Empty = every arena uses [`crate::obs::team_size`].
static MODE_MIX: std::sync::OnceLock<Vec<(usize, f64)>> = std::sync::OnceLock::new();

/// Installs the mode mix as `(cars per team, share of EXPERIENCE)`. Arenas
/// are split in proportion to `share / cars per team`, so each mode gets
/// its share of the collected steps. Returns false if already set.
pub fn set_mode_mix(mix: &[(usize, f64)]) -> bool {
    let total: f64 = mix.iter().map(|&(k, p)| p / k as f64).sum();
    assert!(total > 0.0, "mode_mix: no positive share");
    let arenas: Vec<(usize, f64)> = mix
        .iter()
        .map(|&(k, p)| {
            assert!((1..=3).contains(&k), "mode_mix: team size {k}");
            (k, p / k as f64 / total)
        })
        .collect();
    MODE_MIX.set(arenas).is_ok()
}

/// Cars per team for game `id` under the mode mix: a low-discrepancy
/// (golden ratio) draw over the arena shares, so any block of games is
/// close to the mix. `None` id or no mix: [`crate::obs::team_size`].
#[must_use]
pub fn team_size_for_game(id: Option<usize>) -> usize {
    match (MODE_MIX.get(), id) {
        (Some(mix), Some(g)) if !mix.is_empty() => {
            let u = ((g as f64 + 0.5) * 0.618_033_988_749_895).fract();
            let mut acc = 0.0;
            for &(k, a) in mix {
                acc += a;
                if u < acc {
                    return k;
                }
            }
            mix.last().map_or(1, |m| m.0)
        }
        _ => crate::obs::team_size(),
    }
}

/// [`create_env`] for game `id` of a collector pool (mode mix aware).
#[must_use]
pub fn create_env_for_game(rewards: &RewardConfig, setters: &[WeightedSetter], id: Option<usize>) -> BotEnv {
    create_env_sized(rewards, setters, team_size_for_game(id))
}

/// An arena with `per_team` cars per team.
#[must_use]
pub fn create_env_sized(rewards: &RewardConfig, setters: &[WeightedSetter], per_team: usize) -> BotEnv {
    let mut arena = Arena::new(GameMode::Soccar);
    // Blue, Orange, Blue, Orange...: cars 0 and 1 stay one blue and one
    // orange, as the 1v1 setters (replay, mirrors) expect.
    for _ in 0..per_team {
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    }
    Env::new(
        arena,
        build_mix(setters),
        V2DefaultObs::default(),
        LookupNoMask::default(),
        rewards.build(),
        any_terminal![OnGoalCondition, GameEndedCondition<GAME_TICKS>],
        NoTouchCondition::default(),
        SharedInfo::default(),
    )
}
