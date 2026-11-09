//! Frozen opponents that are not an [`Actic`]: a foreign policy with its own
//! observation builder, action table and decision timing (e.g. a GigaLearn
//! bot ported to Rust). The collector builds their observations from the
//! RocketSim game state, runs their inference in one batch per decision and
//! overrides their cars' controls, so the learner's obs/action spaces do not
//! have to match.

use std::fmt;
use std::sync::Arc;

use rlgym::GameState;
use rlgym::rocketsim::CarControls;

use super::model::Actic;

/// A frozen policy driven from raw game state.
///
/// Per learner interval of `tick_skip` physics ticks:
///
/// - `action_delay > 0`: hold the car's current controls for that many
///   ticks, then apply the new action for the rest of the interval
///   (GigaLearn's `actionDelay = tickSkip - 1`). One `build`/`infer` covers
///   the whole interval.
/// - `action_delay == 0` and `tick_skip >= 2`: apply this decision for the
///   first half of the interval, rebuild from the arena, then `build`/`infer`
///   again for the second half. Ripple decides every 4 ticks this way while
///   the learner stays on one obs/action/reward per 8-tick interval.
///   Galactus and Megabyte return 7 and keep the hold-then-switch path.
pub trait ExternalPolicy: Send + Sync {
    /// Ticks the previous controls are held before the new action applies.
    fn action_delay(&self) -> u8;
    /// Observation width of one row.
    fn obs_width(&self) -> usize;
    /// Number of discrete actions (mask width).
    fn num_actions(&self) -> usize;
    /// Write the observation and action mask of car `car` (index into
    /// `state.cars`) into `obs` / `mask`. Called from collector worker
    /// threads, one game per job.
    fn build(&self, state: &GameState, car: usize, obs: &mut [f32], mask: &mut [bool]);
    /// Batched inference over `rows` rows; returns one action index per row.
    fn infer(&self, obs: &[f32], masks: &[bool], rows: usize) -> Vec<usize>;
    /// Controls of action `action`.
    fn controls(&self, action: usize) -> CarControls;
}

/// An [`ExternalPolicy`] in the self-play pool with its sampling weight.
#[derive(Clone)]
pub struct ExternalOpponent {
    pub name: String,
    pub policy: Arc<dyn ExternalPolicy>,
    pub weight: f32,
}

impl fmt::Debug for ExternalOpponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalOpponent")
            .field("name", &self.name)
            .field("weight", &self.weight)
            .finish_non_exhaustive()
    }
}

/// The opponent of one collection: a policy network or an external policy.
#[derive(Clone)]
pub enum Opponent {
    Model(Actic),
    External(Arc<dyn ExternalPolicy>),
}

impl Opponent {
    pub fn as_ref(&self) -> OpponentRef<'_> {
        match self {
            Opponent::Model(m) => OpponentRef::Model(m),
            Opponent::External(e) => OpponentRef::External(e.as_ref()),
        }
    }
}

/// Borrowed [`Opponent`], shared by the collector threads.
#[derive(Clone, Copy)]
pub enum OpponentRef<'a> {
    Model(&'a Actic),
    External(&'a dyn ExternalPolicy),
}
