//! State setters and their weighted mix, configurable per experiment.
//!
//! `train.toml` lists the mix as `[[state_setters]]` tables, each with a
//! `kind`, a `weight` and the kind's parameters:
//!
//! ```toml
//! [[state_setters]]
//! kind = "kickoff"
//! weight = 0.5
//!
//! [[state_setters]]
//! kind = "air_near_ball"   # flip reset setups
//! weight = 0.3
//! ball_height = [600.0, 1500.0]
//!
//! [[state_setters]]
//! kind = "wall_air_dribble"
//! weight = 0.2
//! ```
//!
//! With no `[[state_setters]]` the mix is [`default_mix`], the one every run
//! so far used. To add a setter, add a variant to [`SetterKind`] and a
//! `StateSetter` impl.
//!
//! [`SetterMix`] samples the mix and tells the shared info which setter
//! started each episode ([`SetterLog`]), for the `Setters/*` metrics and the
//! per-setter flip reset rates.

use std::f32::consts::PI;

use rand::RngExt;
use rlgymppo_utils::rlgym::StateSetter;
use rlgymppo_utils::rlgym::rocketsim::{Arena, BallState, CarState, Mat3A, Vec3A};
use rlgymppo_utils::shared_info::SharedInfoRng;
use rlgymppo_utils::state_setters::{KickoffState, RandomState};
use serde::Deserialize;

use crate::curriculum::CurriculumLog;

/// Side wall `|x|`, and the car's half height when resting on a surface.
const SIDE_WALL_X: f32 = 4096.0;
const CAR_REST_HEIGHT: f32 = 17.0;

/// One entry of the mix.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct WeightedSetter {
    pub weight: f32,
    #[serde(flatten)]
    pub kind: SetterKind,
}

/// The available setters.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SetterKind {
    /// A random standard kickoff.
    Kickoff,
    /// `RandomState`: random ball and cars over the whole field.
    Random {
        #[serde(default)]
        ball_speed: bool,
        #[serde(default)]
        car_speed: bool,
        #[serde(default)]
        cars_on_ground: bool,
    },
    /// One car airborne just under the ball with its wheels toward it.
    AirNearBall(AirNearBall),
    /// One car driving up a side wall with the ball on its nose.
    WallAirDribble(WallAirDribble),
    /// One car in the air 400-800 uu from the ball, moving and facing the
    /// wrong way: a flip reset needs steering and boost.
    AirReset(AirReset),
    /// Adaptive flip reset curriculum ([`crate::curriculum`]).
    ResetCurriculum(crate::curriculum::ResetCurriculum),
    /// A state recorded from real self-play (`indococo-train snapshots`):
    /// aerial situations near the ball from kickoff and random episodes.
    Replay(Replay),
    /// FR9: the same replay setter over a snapshot file filtered to
    /// reset-able geometry (`tools/snap/filter_reset_spots.py`). Its own
    /// label so its resets/uses never count as normal play.
    ResetReplay(Replay),
    /// Aerial defense drills: ball airborne moving toward our goal, opponent
    /// positioned to contest/shoot, our car at varied defensive positions.
    AerialDefense(AerialDefense),
    /// Demo threat starts: opponent supersonic heading toward our car.
    DemoThreat(DemoThreat),
}

impl SetterKind {
    /// Metric label (the `kind` key).
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Kickoff => "kickoff",
            Self::Random { .. } => "random",
            Self::AirNearBall(_) => "air_near_ball",
            Self::WallAirDribble(_) => "wall_air_dribble",
            Self::AirReset(_) => "air_reset",
            Self::ResetCurriculum(_) => "reset_curriculum",
            Self::Replay(_) => "replay",
            Self::ResetReplay(_) => "reset_replay",
            Self::AerialDefense(_) => "aerial_defense",
            Self::DemoThreat(_) => "demo_threat",
        }
    }
}

/// The mix used before setters were configurable: 34% kickoffs, 33% random
/// states with moving cars, 33% random states with cars on the ground.
#[must_use]
pub fn default_mix() -> Vec<WeightedSetter> {
    let random = |car_speed, cars_on_ground| SetterKind::Random {
        ball_speed: true,
        car_speed,
        cars_on_ground,
    };
    vec![
        WeightedSetter {
            weight: 0.34,
            kind: SetterKind::Kickoff,
        },
        WeightedSetter {
            weight: 0.33,
            kind: random(true, false),
        },
        WeightedSetter {
            weight: 0.33,
            kind: random(false, true),
        },
    ]
}

/// Shared info that wants to know which setter started an episode.
pub trait SetterLog {
    /// `chosen` started the episode; `labels` are all labels in the mix.
    fn record_setter(&mut self, chosen: &'static str, labels: &[&'static str]);
}

/// The weighted setter mix. Entries with weight 0 are dropped.
pub struct SetterMix<SI> {
    setters: Vec<Box<dyn StateSetter<SI> + Send>>,
    labels: Vec<&'static str>,
    /// Distinct labels, for the metrics.
    distinct: Vec<&'static str>,
    cumulative: Vec<f32>,
}

impl<SI> SetterMix<SI> {
    /// Index of the setter for a uniform draw `u` in `[0, total weight)`.
    fn pick(&self, u: f32) -> usize {
        self.cumulative
            .iter()
            .position(|&c| u < c)
            .unwrap_or(self.cumulative.len() - 1)
    }
}

impl<SI: SharedInfoRng + SetterLog + CurriculumLog> StateSetter<SI> for SetterMix<SI> {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        let total = *self.cumulative.last().expect("non-empty mix");
        let i = self.pick(shared_info.rng().random_range(0.0..total));
        shared_info.record_setter(self.labels[i], &self.distinct);
        self.setters[i].apply(arena, shared_info);
    }
}

/// Builds the weighted setter mix.
///
/// # Panics
/// If no entry has a positive weight.
#[must_use]
pub fn build_mix<SI: SharedInfoRng + SetterLog + CurriculumLog + 'static>(
    mix: &[WeightedSetter],
) -> SetterMix<SI> {
    let used: Vec<&WeightedSetter> = mix.iter().filter(|s| s.weight > 0.0).collect();
    assert!(
        !used.is_empty(),
        "state_setters: no entry with a positive weight"
    );
    let labels: Vec<&'static str> = used.iter().map(|s| s.kind.label()).collect();
    let mut distinct = labels.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let cumulative = used
        .iter()
        .scan(0.0, |acc, s| {
            *acc += s.weight;
            Some(*acc)
        })
        .collect();
    SetterMix {
        setters: used.iter().map(|s| s.kind.build::<SI>()).collect(),
        labels,
        distinct,
        cumulative,
    }
}

impl SetterKind {
    fn build<SI: SharedInfoRng + CurriculumLog + 'static>(
        &self,
    ) -> Box<dyn StateSetter<SI> + Send> {
        match *self {
            Self::Kickoff => Box::new(KickoffState),
            Self::Random {
                ball_speed,
                car_speed,
                cars_on_ground,
            } => match (ball_speed, car_speed, cars_on_ground) {
                (false, false, false) => Box::new(RandomState::<false, false, false>),
                (false, false, true) => Box::new(RandomState::<false, false, true>),
                (false, true, false) => Box::new(RandomState::<false, true, false>),
                (false, true, true) => Box::new(RandomState::<false, true, true>),
                (true, false, false) => Box::new(RandomState::<true, false, false>),
                (true, false, true) => Box::new(RandomState::<true, false, true>),
                (true, true, false) => Box::new(RandomState::<true, true, false>),
                (true, true, true) => Box::new(RandomState::<true, true, true>),
            },
            Self::AirNearBall(ref s) => Box::new(s.clone()),
            Self::WallAirDribble(ref s) => Box::new(s.clone()),
            Self::AirReset(ref s) => Box::new(s.clone()),
            Self::Replay(ref s) | Self::ResetReplay(ref s) => Box::new(s.clone()),
            Self::ResetCurriculum(ref s) => {
                crate::curriculum::configure(s.adapt());
                Box::new(s.clone())
            }
            Self::AerialDefense(ref s) => Box::new(s.clone()),
            Self::DemoThreat(ref s) => Box::new(s.clone()),
        }
    }
}

/// Flip reset setup: the ball in the air and one car (random team) under it,
/// flip used, wheels within `max_tilt_deg` of pointing at the ball, closing
/// in at `approach_speed`. The other car starts on the ground.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AirNearBall {
    pub ball_height: [f32; 2],
    /// Distance from the car's center to the ball's center.
    pub distance: [f32; 2],
    /// Angle between the car-to-ball direction and straight up.
    pub max_tilt_deg: f32,
    pub approach_speed: [f32; 2],
    /// Shared velocity of ball and car (random horizontal direction).
    pub carry_speed: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for AirNearBall {
    fn default() -> Self {
        Self {
            ball_height: [600.0, 1600.0],
            distance: [160.0, 400.0],
            max_tilt_deg: 35.0,
            approach_speed: [100.0, 500.0],
            carry_speed: [0.0, 800.0],
            boost: [30.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for AirNearBall {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let ball_pos = Vec3A::new(
            rng.random_range(-3000.0..3000.0),
            rng.random_range(-3800.0..3800.0),
            range(rng, self.ball_height),
        );
        let carry = horizontal(rng) * range(rng, self.carry_speed);
        let to_ball = tilted_up(rng, self.max_tilt_deg.to_radians());
        arena.set_ball_state(ball_at(ball_pos, carry));

        let flier = rng.random_range(0..arena.num_cars());
        for idx in 0..arena.num_cars() {
            let rng = shared_info.rng();
            let mut car = if idx == flier {
                let mut car = CarState::DEFAULT;
                car.phys.pos = ball_pos - to_ball * range(rng, self.distance);
                car.phys.vel = carry + to_ball * range(rng, self.approach_speed);
                // Wheels (-up) toward the ball.
                car.phys.rot_mat = orientation(rng, -to_ball);
                car.has_jumped = true;
                car.has_flipped = true;
                car.air_time_since_jump = 2.0;
                car
            } else {
                ground_car(rng)
            };
            car.boost = range(shared_info.rng(), self.boost);
            arena.set_car_state(idx, car.core());
        }
    }
}

/// Flip reset challenge: the ball in the air and one car (random team, jump
/// and flip used) `distance` away, below it within `max_tilt_deg` of straight
/// down. Unlike [`AirNearBall`] nothing is lined up: the car's velocity is
/// the ball's plus a random `velocity_mismatch` in any direction, and its
/// wheels point up to `max_wheel_error_deg` away from the ball. Getting a
/// reset takes steering and boost. The other car starts on the ground.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AirReset {
    pub ball_height: [f32; 2],
    pub distance: [f32; 2],
    pub max_tilt_deg: f32,
    /// Shared horizontal velocity of ball and car.
    pub carry_speed: [f32; 2],
    /// Extra car velocity in a random direction.
    pub velocity_mismatch: [f32; 2],
    pub max_wheel_error_deg: f32,
    pub boost: [f32; 2],
}

impl Default for AirReset {
    fn default() -> Self {
        Self {
            ball_height: [700.0, 1500.0],
            distance: [400.0, 800.0],
            max_tilt_deg: 50.0,
            carry_speed: [0.0, 600.0],
            velocity_mismatch: [200.0, 700.0],
            max_wheel_error_deg: 90.0,
            boost: [50.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for AirReset {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let ball_pos = Vec3A::new(
            rng.random_range(-2800.0..2800.0),
            rng.random_range(-3600.0..3600.0),
            range(rng, self.ball_height),
        );
        let carry = horizontal(rng) * range(rng, self.carry_speed);
        arena.set_ball_state(ball_at(ball_pos, carry));
        let to_ball = tilted_up(rng, self.max_tilt_deg.to_radians());
        let mut car_pos = ball_pos - to_ball * range(rng, self.distance);
        car_pos.z = car_pos.z.max(300.0);

        let flier = rng.random_range(0..arena.num_cars());
        for idx in 0..arena.num_cars() {
            let rng = shared_info.rng();
            let mut car = if idx == flier {
                let mut car = CarState::DEFAULT;
                car.phys.pos = car_pos;
                car.phys.vel = carry + random_unit(rng) * range(rng, self.velocity_mismatch);
                let wheels = tilted_toward(rng, to_ball, self.max_wheel_error_deg.to_radians());
                car.phys.rot_mat = orientation(rng, -wheels);
                car.has_jumped = true;
                car.has_flipped = true;
                car.air_time_since_jump = 2.0;
                car
            } else {
                ground_car(rng)
            };
            car.boost = range(shared_info.rng(), self.boost);
            arena.set_car_state(idx, car.core());
        }
    }
}

/// Uniform random unit vector.
pub(crate) fn random_unit(rng: &mut impl RngExt) -> Vec3A {
    let z: f32 = rng.random_range(-1.0..1.0);
    horizontal(rng) * (1.0 - z * z).sqrt() + Vec3A::Z * z
}

/// Unit vector within `max_angle` radians of `axis` (unit), uniform in azimuth.
pub(crate) fn tilted_toward(rng: &mut impl RngExt, axis: Vec3A, max_angle: f32) -> Vec3A {
    let local = tilted_up(rng, max_angle);
    let helper = if axis.z.abs() < 0.9 {
        Vec3A::Z
    } else {
        Vec3A::X
    };
    let a = axis.cross(helper).normalize();
    let b = axis.cross(a);
    (a * local.x + b * local.y + axis * local.z).normalize()
}

/// Wall air dribble setup: one car (random team) driving up a side wall
/// with the ball resting just ahead of its nose. The other car starts on the
/// ground.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct WallAirDribble {
    pub car_height: [f32; 2],
    pub speed: [f32; 2],
    /// Largest angle of the driving direction from straight up the wall.
    pub max_heading_deg: f32,
    /// Ball offset from the car center: along the heading, and off the wall.
    pub ball_ahead: f32,
    pub ball_off_wall: f32,
    pub boost: [f32; 2],
}

impl Default for WallAirDribble {
    fn default() -> Self {
        Self {
            car_height: [250.0, 900.0],
            speed: [800.0, 1500.0],
            max_heading_deg: 45.0,
            ball_ahead: 170.0,
            ball_off_wall: 110.0,
            boost: [50.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for WallAirDribble {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let side = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let up = Vec3A::new(-side, 0.0, 0.0); // away from the wall
        let heading = rng.random_range(-1.0..1.0) * self.max_heading_deg.to_radians();
        let forward = Vec3A::new(0.0, heading.sin(), heading.cos());
        let speed = range(rng, self.speed);
        let car_pos = Vec3A::new(
            side * (SIDE_WALL_X - CAR_REST_HEIGHT),
            rng.random_range(-3000.0..3000.0),
            range(rng, self.car_height),
        );
        let ball_pos = car_pos + forward * self.ball_ahead + up * self.ball_off_wall;
        arena.set_ball_state(ball_at(ball_pos, forward * speed));

        let driver = rng.random_range(0..arena.num_cars());
        for idx in 0..arena.num_cars() {
            let rng = shared_info.rng();
            let mut car = if idx == driver {
                let mut car = CarState::DEFAULT;
                car.phys.pos = car_pos;
                car.phys.vel = forward * speed;
                car.phys.rot_mat = Mat3A::from_cols(forward, up.cross(forward), up);
                car
            } else {
                ground_car(rng)
            };
            car.boost = range(shared_info.rng(), self.boost);
            arena.set_car_state(idx, car.core());
        }
    }
}

pub(crate) fn range(rng: &mut impl RngExt, [lo, hi]: [f32; 2]) -> f32 {
    if hi > lo {
        rng.random_range(lo..hi)
    } else {
        lo
    }
}

pub(crate) fn ball_at(pos: Vec3A, vel: Vec3A) -> BallState {
    let mut ball = BallState::DEFAULT;
    ball.phys.pos = pos;
    ball.phys.vel = vel;
    ball
}

/// Unit vector in the xy plane.
pub(crate) fn horizontal(rng: &mut impl RngExt) -> Vec3A {
    let a = rng.random_range(-PI..PI);
    Vec3A::new(a.cos(), a.sin(), 0.0)
}

/// Unit vector within `max_tilt` radians of +z, uniform in azimuth.
pub(crate) fn tilted_up(rng: &mut impl RngExt, max_tilt: f32) -> Vec3A {
    let tilt = rng.random_range(0.0..=max_tilt);
    horizontal(rng) * tilt.sin() + Vec3A::Z * tilt.cos()
}

/// A rotation whose up axis is `up`, with a random heading around it.
pub(crate) fn orientation(rng: &mut impl RngExt, up: Vec3A) -> Mat3A {
    let helper = if up.z.abs() < 0.9 { Vec3A::Z } else { Vec3A::X };
    let a = up.cross(helper).normalize();
    let b = up.cross(a);
    let angle = rng.random_range(-PI..PI);
    let forward = a * angle.cos() + b * angle.sin();
    Mat3A::from_cols(forward, up.cross(forward), up)
}

/// A car resting on the field at a random spot and heading.
pub(crate) fn ground_car(rng: &mut impl RngExt) -> CarState {
    let mut car = CarState::DEFAULT;
    car.phys.pos = Vec3A::new(
        rng.random_range(-3500.0..3500.0),
        rng.random_range(-4500.0..4500.0),
        CAR_REST_HEIGHT,
    );
    car.phys.rot_mat = orientation(rng, Vec3A::Z);
    car
}

/// Clamp a car position to valid field bounds and height (on ground or airborne).
pub(crate) fn clamp_car_pos(pos: Vec3A) -> Vec3A {
    Vec3A::new(
        pos.x.clamp(-4096.0 + 100.0, 4096.0 - 100.0),
        pos.y.clamp(-5120.0 + 100.0, 5120.0 - 100.0),
        pos.z.max(CAR_REST_HEIGHT),
    )
}

/// Aerial defense drill: ball airborne moving toward one goal, attacker
/// positioned to shoot/contest, defender at varied defensive positions (some
/// goal-side, some badly positioned). Trains aerial defense and positioning
/// against aerial threats. Randomizes which team defends so both agents learn.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AerialDefense {
    pub ball_height: [f32; 2],
    /// Ball's distance from the defended goal when spawned.
    pub ball_dist_from_goal: [f32; 2],
    /// Ball speed toward goal.
    pub ball_speed: [f32; 2],
    /// Lateral offset from centerline (x direction).
    pub ball_lateral: [f32; 2],
    /// Attacker distance from ball.
    pub attacker_dist: [f32; 2],
    /// Defender positioning: fraction that spawn goal-side (rest spawn out of position).
    pub defender_goal_side_frac: f32,
    /// Defender distance from goal when goal-side.
    pub defender_goal_dist: [f32; 2],
    /// Defender distance from ball when out of position.
    pub defender_ball_dist: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for AerialDefense {
    fn default() -> Self {
        Self {
            ball_height: [300.0, 1500.0],
            ball_dist_from_goal: [1500.0, 4000.0],
            ball_speed: [400.0, 1200.0],
            ball_lateral: [-800.0, 800.0],
            attacker_dist: [300.0, 800.0],
            defender_goal_side_frac: 0.6,
            defender_goal_dist: [500.0, 1500.0],
            defender_ball_dist: [800.0, 1800.0],
            boost: [20.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for AerialDefense {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        
        // Pick which car defends (randomize for symmetric training)
        let defender_idx = rng.random_range(0..arena.num_cars());
        let attacker_idx = 1 - defender_idx;
        
        // Defended goal: blue goal at (0, -5120, 0) or orange goal at (0, 5120, 0)
        let goal_y: f32 = if defender_idx == 0 { -5120.0 } else { 5120.0 };
        let goal_sign: f32 = goal_y.signum();
        
        // Ball positioned away from defended goal, moving toward it
        let ball_y = goal_y + goal_sign * range(rng, self.ball_dist_from_goal);
        let ball_x = range(rng, self.ball_lateral);
        let ball_z = range(rng, self.ball_height);
        let ball_pos = Vec3A::new(ball_x, ball_y, ball_z);
        
        // Ball velocity toward defended goal with some variation
        let to_goal = Vec3A::new(-ball_x * 0.3, goal_y - ball_y, -ball_z * 0.2).normalize();
        let speed = range(rng, self.ball_speed);
        let ball_vel = to_goal * speed;
        arena.set_ball_state(ball_at(ball_pos, ball_vel));
        
        // Attacker: positioned behind ball (far side from defended goal) to shoot toward it
        // from_goal points away from defended goal; attacker placed in that direction
        let from_goal = -to_goal;
        let attacker_offset = tilted_toward(rng, from_goal, 0.7);
        let attacker_pos = ball_pos + attacker_offset * range(rng, self.attacker_dist);
        let mut attacker = CarState::DEFAULT;
        attacker.phys.pos = clamp_car_pos(attacker_pos);
        attacker.phys.vel = to_goal * (speed * 0.7);
        attacker.phys.rot_mat = orientation(rng, Vec3A::Z);
        attacker.boost = range(rng, self.boost);
        
        // Defender: varied positioning (goal-side or out of position)
        let goal_side = rng.random_bool(self.defender_goal_side_frac as f64);
        let defender = if goal_side {
            // Goal-side: between ball and goal
            let def_y = goal_y + goal_sign * range(rng, self.defender_goal_dist);
            let def_x = ball_x * rng.random_range(0.3..0.8);
            let mut car = CarState::DEFAULT;
            car.phys.pos = Vec3A::new(def_x, def_y, CAR_REST_HEIGHT);
            car.phys.vel = Vec3A::ZERO;
            car.phys.rot_mat = orientation(rng, Vec3A::Z);
            car.boost = range(rng, self.boost);
            car
        } else {
            // Out of position: random side of ball
            let angle = rng.random_range(-PI..PI);
            let dist = range(rng, self.defender_ball_dist);
            let offset = Vec3A::new(angle.cos(), angle.sin(), 0.0) * dist;
            let def_pos = ball_pos + offset;
            let mut car = CarState::DEFAULT;
            car.phys.pos = clamp_car_pos(def_pos);
            car.phys.vel = Vec3A::ZERO;
            car.phys.rot_mat = orientation(rng, Vec3A::Z);
            car.boost = range(rng, self.boost) * 0.5;
            car
        };
        
        // Set cars by role
        arena.set_car_state(defender_idx, defender.core());
        arena.set_car_state(attacker_idx, attacker.core());
        
        // Debug assertions
        debug_assert!(ball_pos.z >= 300.0 && ball_pos.z <= 1500.0, "ball z out of range");
        debug_assert!(
            (ball_vel.y * goal_sign) < 0.0,
            "ball not moving toward defended goal"
        );
        let attacker_dist_from_goal = (attacker_pos.y - goal_y).abs();
        let ball_dist_from_goal = (ball_pos.y - goal_y).abs();
        debug_assert!(
            attacker_dist_from_goal > ball_dist_from_goal,
            "attacker not farther from goal than ball"
        );
    }
}

/// Demo threat drill: opponent supersonic heading toward our car. Trains demo
/// awareness and evasion. Randomizes which team is targeted so both agents learn.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DemoThreat {
    /// Distance between cars at start.
    pub distance: [f32; 2],
    /// Attacker speed toward target (clamped to at least 2200 for supersonic).
    pub attacker_speed: [f32; 2],
    /// Ball positioning: distance from midfield.
    pub ball_dist: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for DemoThreat {
    fn default() -> Self {
        Self {
            distance: [800.0, 1500.0],
            attacker_speed: [2250.0, 2300.0],
            ball_dist: [1000.0, 3000.0],
            boost: [30.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for DemoThreat {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        
        // Pick which car is targeted (randomize for symmetric training)
        let target_idx = rng.random_range(0..arena.num_cars());
        let attacker_idx = 1 - target_idx;
        
        // Place ball randomly in the field
        let ball_y_side = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let ball_pos = Vec3A::new(
            rng.random_range(-2000.0..2000.0),
            ball_y_side * range(rng, self.ball_dist),
            100.0,
        );
        arena.set_ball_state(ball_at(ball_pos, Vec3A::ZERO));
        
        // Target: random position on their defensive half
        let target_y_side = if target_idx == 0 { -1.0 } else { 1.0 };
        let target_pos = Vec3A::new(
            rng.random_range(-2500.0..2500.0),
            target_y_side * rng.random_range(1000.0..4000.0),
            CAR_REST_HEIGHT,
        );
        
        // Attacker: positioned to demo the target, supersonic speed
        let to_target_2d = Vec3A::new(
            target_pos.x - 0.0,
            target_pos.y - 0.0,
            0.0,
        )
        .normalize();
        let dist = range(rng, self.distance);
        let attacker_pos = target_pos - to_target_2d * dist;
        
        // Ensure supersonic speed (>= 2200 uu/s)
        let speed = range(rng, self.attacker_speed).max(2200.0);
        let mut attacker = CarState::DEFAULT;
        attacker.phys.pos = clamp_car_pos(attacker_pos);
        attacker.phys.vel = to_target_2d * speed;
        attacker.phys.rot_mat = orientation(rng, Vec3A::Z);
        attacker.boost = 100.0;
        
        let mut target = CarState::DEFAULT;
        target.phys.pos = target_pos;
        target.phys.vel = Vec3A::ZERO;
        target.phys.rot_mat = orientation(rng, Vec3A::Z);
        target.boost = range(rng, self.boost);
        
        arena.set_car_state(target_idx, target.core());
        arena.set_car_state(attacker_idx, attacker.core());
        
        // Debug assertions
        debug_assert!(attacker.phys.pos.z >= 0.0, "attacker z below 0");
        debug_assert!(target.phys.pos.z >= 0.0, "target z below 0");
        debug_assert!(speed >= 2200.0, "attacker not supersonic");
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    #[test]
    fn mix_parses_from_toml_with_defaults() {
        #[derive(Deserialize)]
        struct Cfg {
            state_setters: Vec<WeightedSetter>,
        }
        let cfg: Cfg = toml::from_str(
            r#"
            [[state_setters]]
            kind = "kickoff"
            weight = 0.5
            [[state_setters]]
            kind = "random"
            weight = 0.2
            ball_speed = true
            [[state_setters]]
            kind = "air_near_ball"
            weight = 0.2
            ball_height = [700.0, 900.0]
            [[state_setters]]
            kind = "wall_air_dribble"
            weight = 0.1
            "#,
        )
        .unwrap();
        let s = &cfg.state_setters;
        assert_eq!(s[0].kind, SetterKind::Kickoff);
        assert_eq!(
            s[1].kind,
            SetterKind::Random {
                ball_speed: true,
                car_speed: false,
                cars_on_ground: false
            }
        );
        let SetterKind::AirNearBall(air) = &s[2].kind else {
            panic!("{:?}", s[2])
        };
        assert_eq!(air.ball_height, [700.0, 900.0]);
        assert_eq!(air.max_tilt_deg, AirNearBall::default().max_tilt_deg);
        assert_eq!(
            s[3].kind,
            SetterKind::WallAirDribble(WallAirDribble::default())
        );
    }

    #[test]
    fn unknown_kind_or_parameter_is_rejected() {
        let parse = |t: &str| toml::from_str::<WeightedSetter>(t);
        assert!(parse("kind = \"teleport\"\nweight = 1.0").is_err());
        assert!(parse("kind = \"air_near_ball\"\nweight = 1.0\nheight = [1.0, 2.0]").is_err());
    }

    #[test]
    fn mix_samples_by_weight() {
        let mix = build_mix::<crate::env::SharedInfo>(&[
            WeightedSetter {
                weight: 0.25,
                kind: SetterKind::Kickoff,
            },
            WeightedSetter {
                weight: 0.0,
                kind: SetterKind::AirNearBall(AirNearBall::default()),
            },
            WeightedSetter {
                weight: 0.75,
                kind: SetterKind::AirReset(AirReset::default()),
            },
        ]);
        assert_eq!(mix.labels, ["kickoff", "air_reset"]);
        assert_eq!(mix.pick(0.1), 0);
        assert_eq!(mix.pick(0.3), 1);
        assert_eq!(mix.pick(0.999), 1);
    }

    #[test]
    fn tilted_toward_stays_within_the_cone() {
        let mut rng = SmallRng::seed_from_u64(5);
        for _ in 0..200 {
            let axis = random_unit(&mut rng);
            let v = tilted_toward(&mut rng, axis, 0.5);
            assert!((v.length() - 1.0).abs() < 1e-4);
            assert!(v.dot(axis) >= 0.5f32.cos() - 1e-4);
        }
    }

    #[test]
    fn orientations_are_rotations_with_the_requested_up() {
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..200 {
            let up = tilted_up(&mut rng, 1.2);
            assert!(up.z >= 1.2f32.cos() - 1e-5);
            let m = orientation(&mut rng, -up);
            assert!((m.z_axis - -up).length() < 1e-5);
            assert!((m.determinant() - 1.0).abs() < 1e-4);
            assert!(m.x_axis.dot(m.z_axis).abs() < 1e-5);
        }
    }
}

/// Floats per snapshot record: ball pos/vel/ang_vel (9), then per car
/// pos, vel, ang_vel, rot_mat columns (18), boost, has_jumped,
/// has_double_jumped, has_flipped, air_time_since_jump, flip_time (6).
pub const SNAPSHOT_CAR: usize = 24;
pub const SNAPSHOT_LEN: usize = 9 + 2 * SNAPSHOT_CAR;

/// Encodes a 1v1 game state as one snapshot record.
#[must_use]
pub fn encode_snapshot(ball: &BallState, cars: &[&CarState]) -> Vec<f32> {
    let mut v = Vec::with_capacity(SNAPSHOT_LEN);
    for x in [ball.phys.pos, ball.phys.vel, ball.phys.ang_vel] {
        v.extend_from_slice(&x.to_array());
    }
    for c in cars {
        let m = c.phys.rot_mat;
        for x in [c.phys.pos, c.phys.vel, c.phys.ang_vel, m.x_axis, m.y_axis, m.z_axis] {
            v.extend_from_slice(&x.to_array());
        }
        v.extend_from_slice(&[
            c.boost,
            f32::from(u8::from(c.has_jumped)),
            f32::from(u8::from(c.has_double_jumped)),
            f32::from(u8::from(c.has_flipped)),
            c.air_time_since_jump,
            c.flip_time,
        ]);
    }
    v
}

static SNAPSHOTS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<Vec<f32>>>>> =
    std::sync::OnceLock::new();

/// Loads (once per process) a raw little-endian f32 snapshot file.
///
/// # Panics
/// If the file is missing, empty or not a whole number of records.
fn snapshots(path: &str) -> std::sync::Arc<Vec<f32>> {
    let map = SNAPSHOTS.get_or_init(Default::default);
    let mut map = map.lock().expect("snapshots lock");
    map.entry(path.to_string())
        .or_insert_with(|| {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("replay {path}: {e}"));
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert!(
                !v.is_empty() && v.len() % SNAPSHOT_LEN == 0 && bytes.len() % 4 == 0,
                "replay {path}: not a snapshot file"
            );
            std::sync::Arc::new(v)
        })
        .clone()
}

/// Real-play replay setter: a random recorded snapshot, mirrored (180 deg
/// about z, cars swapped) half the time, with small position/velocity noise.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Replay {
    pub path: String,
    pub pos_noise: f32,
    pub vel_noise: f32,
}

impl Default for Replay {
    fn default() -> Self {
        Self {
            path: String::new(),
            pos_noise: 30.0,
            vel_noise: 50.0,
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for Replay {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let data = snapshots(&self.path);
        let rng = shared_info.rng();
        let n = data.len() / SNAPSHOT_LEN;
        let r = &data[rng.random_range(0..n) * SNAPSHOT_LEN..][..SNAPSHOT_LEN];
        let mirror = rng.random_bool(0.5);
        let f = if mirror { Vec3A::new(-1.0, -1.0, 1.0) } else { Vec3A::ONE };
        let v3 = |i: usize| Vec3A::new(r[i], r[i + 1], r[i + 2]) * f;
        let mut noise = |s: f32| {
            Vec3A::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)) * s
        };
        let mut ball = BallState::DEFAULT;
        ball.phys.pos = v3(0) + noise(self.pos_noise);
        ball.phys.pos.z = ball.phys.pos.z.max(100.0);
        ball.phys.vel = v3(3) + noise(self.vel_noise);
        ball.phys.ang_vel = v3(6);
        arena.set_ball_state(ball);
        for idx in 0..arena.num_cars().min(2) {
            let src = if mirror { 1 - idx } else { idx };
            let o = 9 + src * SNAPSHOT_CAR;
            let mut car = CarState::DEFAULT;
            car.phys.pos = v3(o);
            car.phys.vel = v3(o + 3);
            car.phys.ang_vel = v3(o + 6);
            car.phys.rot_mat = Mat3A::from_cols(v3(o + 9), v3(o + 12), v3(o + 15));
            car.boost = r[o + 18];
            car.has_jumped = r[o + 19] > 0.5;
            car.has_double_jumped = r[o + 20] > 0.5;
            car.has_flipped = r[o + 21] > 0.5;
            car.air_time_since_jump = r[o + 22];
            car.flip_time = r[o + 23];
            arena.set_car_state(idx, car.core());
        }
    }
}
