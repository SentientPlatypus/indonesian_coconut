//! A Python-shaped view of one environment step.
//!
//! Rewards are written against [`StateView`], which mirrors the fields and
//! derived properties of RLGym v2's `GameState`/`Car` (`on_ground`,
//! `has_flip`, `is_flipping`, `ball_touches`, `bump_victim_id`, ...). It is
//! built once per step from the simulator state ([`StateView::from_game_state`])
//! or deserialized from a parity fixture, so the reward code is identical in
//! training and in the Python parity tests.

use glam::DVec3;
use rlgymppo_utils::rlgym::GameState;
use rlgymppo_utils::rocketsim::{ArenaEvent, Team};
use serde::{Deserialize, Serialize};

use crate::consts::{
    BLUE_TEAM, DOUBLEJUMP_MAX_DELAY, FLIP_TORQUE_TIME, GOAL_THRESHOLD, ORANGE_TEAM,
};

/// Ball physics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BallView {
    pub pos: DVec3,
    pub vel: DVec3,
    pub ang_vel: DVec3,
}

/// One car, with the raw fields RLGym's `Car` stores.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CarView {
    /// `0` blue, `1` orange.
    pub team: u8,
    pub pos: DVec3,
    pub vel: DVec3,
    pub ang_vel: DVec3,
    pub forward: DVec3,
    pub right: DVec3,
    pub up: DVec3,
    /// Boost in `[0, 100]`.
    pub boost: f64,
    /// Ball contacts during the last step (summed over its physics ticks).
    pub ball_touches: u32,
    /// Index (into `StateView::cars`) of the car this one bumped, while the
    /// bump cooldown runs.
    pub bump_victim: Option<usize>,
    pub demo_respawn_timer: f64,
    /// front-left, front-right, back-left, back-right.
    pub wheels_with_contact: [bool; 4],
    /// Seconds boost has been held continuously.
    pub boost_active_time: f64,
    pub has_jumped: bool,
    pub has_flipped: bool,
    pub has_double_jumped: bool,
    pub air_time_since_jump: f64,
    pub flip_time: f64,
}

impl CarView {
    #[must_use]
    pub fn is_orange(&self) -> bool {
        self.team == ORANGE_TEAM
    }

    #[must_use]
    pub fn is_blue(&self) -> bool {
        self.team == BLUE_TEAM
    }

    /// At least three wheels touching a surface.
    #[must_use]
    pub fn on_ground(&self) -> bool {
        self.wheels_with_contact.iter().filter(|&&w| w).count() >= 3
    }

    #[must_use]
    pub fn is_demoed(&self) -> bool {
        self.demo_respawn_timer > 0.0
    }

    #[must_use]
    pub fn is_boosting(&self) -> bool {
        self.boost_active_time > 0.0
    }

    #[must_use]
    pub fn has_flip(&self) -> bool {
        !self.has_double_jumped
            && !self.has_flipped
            && self.air_time_since_jump < DOUBLEJUMP_MAX_DELAY
    }

    #[must_use]
    pub fn is_flipping(&self) -> bool {
        self.has_flipped && self.flip_time < FLIP_TORQUE_TIME
    }

    /// `+1` for blue (attacks +y), `-1` for orange.
    #[must_use]
    pub fn attack_sign(&self) -> f64 {
        if self.is_orange() { -1.0 } else { 1.0 }
    }
}

/// Everything a reward sees for one step.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StateView {
    pub tick_count: u64,
    pub ball: BallView,
    pub cars: Vec<CarView>,
}

impl StateView {
    /// Ball inside either goal.
    #[must_use]
    pub fn goal_scored(&self) -> bool {
        self.ball.pos.y.abs() > GOAL_THRESHOLD
    }

    /// Team that scored, if a goal was scored.
    #[must_use]
    pub fn scoring_team(&self) -> Option<u8> {
        self.goal_scored().then_some(if self.ball.pos.y > 0.0 {
            BLUE_TEAM
        } else {
            ORANGE_TEAM
        })
    }

    /// Indices of the cars on the other team.
    pub fn opponents(&self, agent: usize) -> impl Iterator<Item = usize> + '_ {
        let team = self.cars[agent].team;
        (0..self.cars.len()).filter(move |&i| self.cars[i].team != team)
    }

    /// Converts the simulator state of one step.
    #[must_use]
    pub fn from_game_state(state: &GameState) -> Self {
        let cars = state
            .cars
            .iter()
            .map(|(info, car)| {
                let rot = car.phys.rot_mat;
                let wheels = car.wheels_with_contact().map(|w| w.is_some());
                let ball_touches = state
                    .events
                    .iter()
                    .filter(|e| matches!(e, ArenaEvent::CarHitBall(h) if h.car_idx == info.idx))
                    .count() as u32;
                let bump_victim = car
                    .bump_cooldown_victim
                    .filter(|_| car.bump_cooldown_timer > 0.0)
                    .and_then(|idx| state.cars.iter().position(|(i, _)| i.idx == idx));
                CarView {
                    team: if info.team == Team::Blue {
                        BLUE_TEAM
                    } else {
                        ORANGE_TEAM
                    },
                    pos: car.phys.pos.as_dvec3(),
                    vel: car.phys.vel.as_dvec3(),
                    ang_vel: car.phys.ang_vel.as_dvec3(),
                    forward: rot.x_axis.as_dvec3(),
                    right: rot.y_axis.as_dvec3(),
                    up: rot.z_axis.as_dvec3(),
                    boost: f64::from(car.boost),
                    ball_touches,
                    bump_victim,
                    demo_respawn_timer: f64::from(car.demo_respawn_timer),
                    wheels_with_contact: wheels,
                    boost_active_time: f64::from(car.boosting_time),
                    has_jumped: car.has_jumped,
                    has_flipped: car.has_flipped,
                    has_double_jumped: car.has_double_jumped,
                    air_time_since_jump: f64::from(car.air_time_since_jump),
                    flip_time: f64::from(car.flip_time),
                }
            })
            .collect();
        Self {
            tick_count: state.tick_count,
            ball: BallView {
                pos: state.ball.pos.as_dvec3(),
                vel: state.ball.vel.as_dvec3(),
                ang_vel: state.ball.ang_vel.as_dvec3(),
            },
            cars,
        }
    }
}
