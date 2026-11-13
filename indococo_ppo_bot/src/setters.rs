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

/// Side wall `|x|`, back wall `|y|`, and the car's half height when resting
/// on a surface.
const SIDE_WALL_X: f32 = 4096.0;
const BACK_WALL_Y: f32 = 5120.0;
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
    /// Mid air dribble, with the other car on a wall between the ball and
    /// that car's own goal, ready to jump off and challenge.
    AirDribbleWallChallenge(AirDribbleWallChallenge),
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
            Self::AirDribbleWallChallenge(_) => "air_dribble_wall_challenge",
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
            Self::AirDribbleWallChallenge(ref s) => Box::new(s.clone()),
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
        let ball_y = goal_y - goal_sign * range(rng, self.ball_dist_from_goal);
        let ball_x = range(rng, self.ball_lateral);
        let ball_z = range(rng, self.ball_height);
        let ball_pos = Vec3A::new(
            ball_x.clamp(-4096.0 + 100.0, 4096.0 - 100.0),
            ball_y.clamp(-5120.0 + 100.0, 5120.0 - 100.0),
            ball_z.clamp(300.0, 1500.0),
        );
        
        // Ball velocity toward defended goal with some variation
        let to_goal = Vec3A::new(-ball_pos.x * 0.3, goal_y - ball_pos.y, -ball_pos.z * 0.2).normalize();
        let speed = range(rng, self.ball_speed);
        let ball_vel = to_goal * speed;
        arena.set_ball_state(ball_at(ball_pos, ball_vel));
        
        // Attacker: positioned behind ball (far side from defended goal) to shoot toward it
        // from_goal points away from defended goal; attacker placed in that direction
        let from_goal = -to_goal;
        let attacker_offset = tilted_toward(rng, from_goal, 0.7);
        let attacker_pos_raw = ball_pos + attacker_offset * range(rng, self.attacker_dist);
        let attacker_pos = clamp_car_pos(attacker_pos_raw);
        let mut attacker = CarState::DEFAULT;
        attacker.phys.pos = attacker_pos;
        attacker.phys.vel = to_goal * (speed * 0.7);
        attacker.phys.rot_mat = orientation(rng, Vec3A::Z);
        attacker.boost = range(rng, self.boost);
        
        // Defender: varied positioning (goal-side or out of position)
        let goal_side = rng.random_bool(self.defender_goal_side_frac as f64);
        let defender = if goal_side {
            // Goal-side: between ball and goal
            let def_y = goal_y - goal_sign * range(rng, self.defender_goal_dist);
            let def_x = ball_x * rng.random_range(0.3..0.8);
            let def_pos = Vec3A::new(
                def_x.clamp(-4096.0 + 100.0, 4096.0 - 100.0),
                def_y.clamp(-5120.0 + 100.0, 5120.0 - 100.0),
                CAR_REST_HEIGHT,
            );
            let mut car = CarState::DEFAULT;
            car.phys.pos = def_pos;
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
        
        // Validate invariants
        assert!(
            ball_pos.z >= 300.0 && ball_pos.z <= 1500.0,
            "ball z out of range: {}",
            ball_pos.z
        );
        assert!(
            ball_pos.x.abs() <= 4096.0 && ball_pos.y.abs() <= 5120.0,
            "ball outside field: ({}, {})",
            ball_pos.x,
            ball_pos.y
        );
        assert!(
            (ball_vel.y * goal_sign) > 0.0,
            "ball not moving toward defended goal: ball_vel.y={}, goal_sign={}",
            ball_vel.y,
            goal_sign
        );
        let attacker_dist_from_goal = (attacker_pos.y - goal_y).abs();
        let ball_dist_from_goal = (ball_pos.y - goal_y).abs();
        assert!(
            attacker_dist_from_goal > ball_dist_from_goal,
            "attacker not farther from goal than ball: attacker_dist={}, ball_dist={}",
            attacker_dist_from_goal,
            ball_dist_from_goal
        );
        assert!(
            attacker_pos.x.abs() <= 4096.0 && attacker_pos.y.abs() <= 5120.0 && attacker_pos.z >= CAR_REST_HEIGHT,
            "attacker outside bounds: ({}, {}, {})",
            attacker_pos.x,
            attacker_pos.y,
            attacker_pos.z
        );
        assert!(
            defender.phys.pos.x.abs() <= 4096.0 && defender.phys.pos.y.abs() <= 5120.0 && defender.phys.pos.z >= CAR_REST_HEIGHT,
            "defender outside bounds: ({}, {}, {})",
            defender.phys.pos.x,
            defender.phys.pos.y,
            defender.phys.pos.z
        );
    }
}

/// Half-angle of the recoverable approach cone, around the defender-to-midfield
/// direction. 55° sits in the ±45–60° band: wide enough that the close is not
/// the single midfield ray, narrow enough that the attacker still comes from
/// upfield with about a second to react.
const DEMO_APPROACH_CONE_DEG: f32 = 55.0;
/// Keeper lateral spread when [`DemoThreat::target_goal_dist`] is set.
const DEMO_KEEPER_X: f32 = 900.0;
/// Grounded keeper boost in the goal-side drill (real Galactus victims sit ~90).
const DEMO_KEEPER_BOOST: [f32; 2] = [50.0, 100.0];
/// Ball distance from the defended goal line when `ball_threat` is on.
/// 3200 uu at the 1450 uu/s cap is ~2.2 s, inside the ~1–2 s scoring window
/// once slow far shots are sped up (see [`DemoThreat::ball_threat_state`]).
const BALL_THREAT_DIST: [f32; 2] = [1800.0, 3200.0];
/// Horizontal ball speed toward the net when `ball_threat` is on.
const BALL_THREAT_SPEED: [f32; 2] = [900.0, 1450.0];
/// Desired time for the ball to reach the goal line, before the speed clamp.
const BALL_THREAT_ETA: [f32; 2] = [1.05, 1.95];

/// Demo threat drill: opponent heading toward our car. Trains demo awareness
/// and evasion. Randomizes which team is targeted so both agents learn.
///
/// With no optional fields this is the original drill: a supersonic car on
/// the ray from midfield, a static ball, and the target at least ~1120 uu
/// from its goal line. ST32 turns on [`Self::target_goal_dist`],
/// [`Self::attacker_speed_range`] and [`Self::ball_threat`] so the keeper is
/// goal-side, the close is a general angle at a realistic speed, and failing
/// to dodge actually concedes.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DemoThreat {
    /// Distance between cars at start. `dist_range` is an alias so ST32 toml
    /// can use that name without a second field.
    #[serde(alias = "dist_range")]
    pub distance: [f32; 2],
    /// Attacker speed toward the target. When this range sits at or below
    /// 2200 uu/s it is used as written. Otherwise it is clamped up to 2200
    /// so the legacy supersonic drill is unchanged. Prefer
    /// [`Self::attacker_speed_range`] for an explicit closing speed.
    pub attacker_speed: [f32; 2],
    /// Ball positioning for the static ball: distance from midfield.
    /// Ignored when [`Self::ball_threat`] is set.
    pub ball_dist: [f32; 2],
    pub boost: [f32; 2],
    /// When set, place the defender this far from its own goal line (uu),
    /// goal-side, grounded, facing the ball. `None` keeps the legacy
    /// midfield-relative placement.
    #[serde(default)]
    pub target_goal_dist: Option<[f32; 2]>,
    /// Closing speed (uu/s) used as written, with no 2200 floor. `None`
    /// falls back to [`Self::attacker_speed`].
    #[serde(default)]
    pub attacker_speed_range: Option<[f32; 2]>,
    /// When `true`, the ball is in the defensive half with a velocity into
    /// the defender's net, so a demo concedes and a dodge can still save.
    /// `None` and `false` keep the static ball. Default `false`.
    #[serde(default)]
    pub ball_threat: Option<bool>,
}

impl Default for DemoThreat {
    fn default() -> Self {
        Self {
            distance: [800.0, 1500.0],
            attacker_speed: [2250.0, 2300.0],
            ball_dist: [1000.0, 3000.0],
            boost: [30.0, 100.0],
            target_goal_dist: None,
            attacker_speed_range: None,
            ball_threat: None,
        }
    }
}

impl DemoThreat {
    /// Legacy geometry, static ball, and the supersonic floor.
    fn use_legacy(&self) -> bool {
        self.target_goal_dist.is_none()
            && self.attacker_speed_range.is_none()
            && !self.ball_threat.unwrap_or(false)
            && self.attacker_speed[1] > 2200.0
    }

    /// Closing speed. The 2200 floor applies only to the legacy supersonic
    /// default, so an explicit sub-supersonic `attacker_speed` or
    /// `attacker_speed_range` is honored.
    fn sample_speed(&self, rng: &mut impl RngExt) -> f32 {
        if let Some(band) = self.attacker_speed_range {
            return range(rng, band);
        }
        let speed = range(rng, self.attacker_speed);
        if self.attacker_speed[1] > 2200.0 {
            speed.max(2200.0)
        } else {
            speed
        }
    }

    /// Ball in the half being attacked, traveling into that net on a small arc.
    /// Speed is chosen from the 1.05–1.95 s window and clamped to
    /// 900–1450 uu/s, so a 3200 uu ball still arrives in ~2.2 s instead of
    /// the 3.5 s a 900 uu/s shot would take.
    fn ball_threat_state(rng: &mut impl RngExt, goal_y: f32, goal_sign: f32) -> (Vec3A, Vec3A) {
        let ball_dist = range(rng, BALL_THREAT_DIST);
        let eta = range(rng, BALL_THREAT_ETA);
        let speed = (ball_dist / eta).clamp(BALL_THREAT_SPEED[0], BALL_THREAT_SPEED[1]);
        let ball_x: f32 = rng.random_range(-700.0..700.0);
        let ball_y = goal_y - goal_sign * ball_dist;
        let ball_z = rng.random_range(93.15..220.0);
        let ball_pos = Vec3A::new(
            ball_x.clamp(-SIDE_WALL_X + 100.0, SIDE_WALL_X - 100.0),
            ball_y.clamp(-BACK_WALL_Y + 100.0, BACK_WALL_Y - 100.0),
            ball_z,
        );
        // Pull the aim point toward the mouth center so the shot is on net
        // and the sideways component stays small (the y-speed carries the ETA).
        let aim_x = (ball_x * 0.35).clamp(-450.0, 450.0);
        let aim_z = rng.random_range(140.0..320.0);
        let aim = Vec3A::new(aim_x, goal_y, aim_z);
        let to_aim = aim - ball_pos;
        let horiz = Vec3A::new(to_aim.x, to_aim.y, 0.0);
        let horiz = if horiz.length_squared() < 1.0 {
            Vec3A::new(0.0, -goal_sign, 0.0)
        } else {
            horiz.normalize()
        };
        let vz = rng.random_range(120.0..420.0);
        let ball_vel = Vec3A::new(horiz.x * speed, horiz.y * speed, vz);
        (ball_pos, ball_vel)
    }
}

/// Grounded rotation whose nose is `dir` (xy). Falls back to +y.
fn face_xy(dir: Vec3A) -> Mat3A {
    let mut flat = Vec3A::new(dir.x, dir.y, 0.0);
    if flat.length_squared() < 1e-6 {
        flat = Vec3A::Y;
    }
    let forward = flat.normalize();
    let up = Vec3A::Z;
    Mat3A::from_cols(forward, up.cross(forward), up)
}

impl<SI: SharedInfoRng> StateSetter<SI> for DemoThreat {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        if self.use_legacy() {
            self.apply_legacy(arena, shared_info);
        } else {
            self.apply_recoverable(arena, shared_info);
        }
    }
}

impl DemoThreat {
    /// Original drill. RNG order matches the pre-ST32 setter so old configs
    /// keep the same states for a given seed.
    fn apply_legacy<SI: SharedInfoRng>(&self, arena: &mut Arena, shared_info: &mut SI) {
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
        let target_x: f32 = rng.random_range(-2500.0..2500.0);
        let target_y_raw: f32 = target_y_side * rng.random_range(1000.0..4000.0);
        let target_pos = Vec3A::new(
            target_x.clamp(-4096.0 + 100.0, 4096.0 - 100.0),
            target_y_raw.clamp(-5120.0 + 100.0, 5120.0 - 100.0),
            CAR_REST_HEIGHT,
        );

        // Attacker: positioned to demo the target, supersonic speed
        let to_target_2d = Vec3A::new(target_pos.x, target_pos.y, 0.0).normalize();
        let dist = range(rng, self.distance);
        let attacker_pos_raw = target_pos - to_target_2d * dist;
        let attacker_pos = clamp_car_pos(attacker_pos_raw);

        // Ensure supersonic speed (>= 2200 uu/s)
        let speed = range(rng, self.attacker_speed).max(2200.0);
        let mut attacker = CarState::DEFAULT;
        attacker.phys.pos = attacker_pos;
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

        // Validate invariants
        assert!(speed >= 2200.0, "attacker not supersonic: speed={}", speed);
        assert!(
            attacker_pos.x.abs() <= 4096.0
                && attacker_pos.y.abs() <= 5120.0
                && attacker_pos.z >= CAR_REST_HEIGHT,
            "attacker outside bounds: ({}, {}, {})",
            attacker_pos.x,
            attacker_pos.y,
            attacker_pos.z
        );
        assert!(
            target_pos.x.abs() <= 4096.0
                && target_pos.y.abs() <= 5120.0
                && target_pos.z >= CAR_REST_HEIGHT,
            "target outside bounds: ({}, {}, {})",
            target_pos.x,
            target_pos.y,
            target_pos.z
        );
        let heading_dot = attacker.phys.vel.normalize().dot(to_target_2d);
        assert!(
            heading_dot > 0.99,
            "attacker not heading at target: dot={}",
            heading_dot
        );
    }

    /// Goal-side keeper, a cone approach, an optional threatening ball.
    /// Each optional field falls back to the legacy piece when it is unset,
    /// so a config can turn on one of them.
    fn apply_recoverable<SI: SharedInfoRng>(&self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();

        let target_idx = rng.random_range(0..arena.num_cars());
        let attacker_idx = 1 - target_idx;
        // Blue (car 0) defends y = -5120. Orange defends y = +5120.
        let goal_y: f32 = if target_idx == 0 { -BACK_WALL_Y } else { BACK_WALL_Y };
        let goal_sign = goal_sign_of(goal_y);

        let target_pos = if let Some(goal_dist) = self.target_goal_dist {
            let x = rng.random_range(-DEMO_KEEPER_X..DEMO_KEEPER_X);
            let along = range(rng, goal_dist);
            Vec3A::new(
                x.clamp(-SIDE_WALL_X + 100.0, SIDE_WALL_X - 100.0),
                (goal_y - goal_sign * along).clamp(-BACK_WALL_Y + 100.0, BACK_WALL_Y - 100.0),
                CAR_REST_HEIGHT,
            )
        } else {
            let target_y_side = if target_idx == 0 { -1.0 } else { 1.0 };
            let target_x: f32 = rng.random_range(-2500.0..2500.0);
            let target_y_raw: f32 = target_y_side * rng.random_range(1000.0..4000.0);
            Vec3A::new(
                target_x.clamp(-SIDE_WALL_X + 100.0, SIDE_WALL_X - 100.0),
                target_y_raw.clamp(-BACK_WALL_Y + 100.0, BACK_WALL_Y - 100.0),
                CAR_REST_HEIGHT,
            )
        };

        let (ball_pos, ball_vel) = if self.ball_threat.unwrap_or(false) {
            Self::ball_threat_state(rng, goal_y, goal_sign)
        } else {
            let ball_y_side = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            let ball_pos = Vec3A::new(
                rng.random_range(-2000.0..2000.0),
                ball_y_side * range(rng, self.ball_dist),
                100.0,
            );
            (ball_pos, Vec3A::ZERO)
        };
        arena.set_ball_state(ball_at(ball_pos, ball_vel));

        let dist = range(rng, self.distance);
        let (attacker_pos, to_target_2d) = if self.target_goal_dist.is_some() {
            // Cone around defender → midfield, then drive straight at the keeper.
            let max_ang = DEMO_APPROACH_CONE_DEG.to_radians();
            let ang = rng.random_range(-max_ang..max_ang);
            let (s, c) = ang.sin_cos();
            // At ang = 0 this is (0, -goal_sign): straight upfield from the net.
            let approach = Vec3A::new(goal_sign * s, -goal_sign * c, 0.0);
            let attacker_pos = clamp_car_pos(target_pos + approach * dist);
            let to_target = target_pos - attacker_pos;
            let to_target_2d = Vec3A::new(to_target.x, to_target.y, 0.0).normalize();
            (attacker_pos, to_target_2d)
        } else {
            let to_target_2d = Vec3A::new(target_pos.x, target_pos.y, 0.0).normalize();
            let attacker_pos = clamp_car_pos(target_pos - to_target_2d * dist);
            (attacker_pos, to_target_2d)
        };

        let speed = self.sample_speed(rng);
        let mut attacker = CarState::DEFAULT;
        attacker.phys.pos = attacker_pos;
        attacker.phys.vel = to_target_2d * speed;
        attacker.phys.rot_mat = if self.target_goal_dist.is_some() {
            face_xy(to_target_2d)
        } else {
            orientation(rng, Vec3A::Z)
        };
        attacker.boost = 100.0;

        let mut target = CarState::DEFAULT;
        target.phys.pos = target_pos;
        target.phys.vel = Vec3A::ZERO;
        target.phys.rot_mat = if self.target_goal_dist.is_some() {
            face_xy(ball_pos - target_pos)
        } else {
            orientation(rng, Vec3A::Z)
        };
        target.boost = if self.target_goal_dist.is_some() {
            range(rng, DEMO_KEEPER_BOOST)
        } else {
            range(rng, self.boost)
        };

        arena.set_car_state(target_idx, target.core());
        arena.set_car_state(attacker_idx, attacker.core());

        if self.attacker_speed_range.is_none() && self.attacker_speed[1] > 2200.0 {
            assert!(speed >= 2200.0, "attacker not supersonic: speed={}", speed);
        }
        assert!(
            attacker_pos.x.abs() <= SIDE_WALL_X
                && attacker_pos.y.abs() <= BACK_WALL_Y
                && attacker_pos.z >= CAR_REST_HEIGHT,
            "attacker outside bounds: ({}, {}, {})",
            attacker_pos.x,
            attacker_pos.y,
            attacker_pos.z
        );
        assert!(
            target_pos.x.abs() <= SIDE_WALL_X
                && target_pos.y.abs() <= BACK_WALL_Y
                && target_pos.z >= CAR_REST_HEIGHT,
            "target outside bounds: ({}, {}, {})",
            target_pos.x,
            target_pos.y,
            target_pos.z
        );
        let heading_dot = attacker.phys.vel.normalize().dot(to_target_2d);
        assert!(
            heading_dot > 0.99,
            "attacker not heading at target: dot={}",
            heading_dot
        );
        if self.ball_threat.unwrap_or(false) {
            assert!(
                ball_vel.y * goal_sign > 0.0,
                "ball not moving toward defended goal: vel.y={}, goal_sign={}",
                ball_vel.y,
                goal_sign
            );
            assert!(
                ball_pos.x.abs() <= SIDE_WALL_X && ball_pos.y.abs() <= BACK_WALL_Y && ball_pos.z > 90.0,
                "ball outside field: ({}, {}, {})",
                ball_pos.x,
                ball_pos.y,
                ball_pos.z
            );
        }
        if let Some(goal_dist) = self.target_goal_dist {
            let along = (target_pos.y - goal_y).abs();
            assert!(
                along + 1.0 >= goal_dist[0] && along <= goal_dist[1] + 1.0,
                "keeper not in goal-dist window: along={along}, range={goal_dist:?}"
            );
            let from_keeper = attacker_pos - target_pos;
            let from_xy = Vec3A::new(from_keeper.x, from_keeper.y, 0.0).normalize();
            let upfield = Vec3A::new(0.0, -goal_sign, 0.0);
            assert!(
                from_xy.dot(upfield) >= DEMO_APPROACH_CONE_DEG.to_radians().cos() - 1e-3,
                "attacker outside the approach cone: dot={}",
                from_xy.dot(upfield)
            );
        }
    }
}

fn goal_sign_of(goal_y: f32) -> f32 {
    if goal_y >= 0.0 { 1.0 } else { -1.0 }
}

/// Which wall the challenger defends from.
#[derive(Clone, Copy)]
enum ChallengeWall {
    /// Same side wall as the dribble, ahead of the ball.
    Same,
    /// Back wall beside the challenger's own goal.
    Back,
    /// The other side wall, still goal-side of the ball.
    Opposite,
}

/// Air-dribble 50/50: one car carries the ball in the air along a side, and
/// the other is on a wall between the ball and its own goal, wheels on the
/// wall and jump still available. Which car dribbles is random, and the
/// field is mirrored, so each team practices both roles. Weight 0 leaves the
/// default mix unchanged.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AirDribbleWallChallenge {
    /// Dribbler height.
    pub dribbler_height: [f32; 2],
    /// Gap between the dribbler and the side wall when it hugs the wall.
    pub off_wall: [f32; 2],
    /// Chance the dribbler is farther into the field instead of hugging the wall.
    pub into_field_frac: f32,
    /// Gap from the side wall on that farther spawn.
    pub into_field: [f32; 2],
    /// Speed of the car and the ball along the nose.
    pub speed: [f32; 2],
    /// Cone of the nose around the direction to the opponent goal, in degrees.
    pub nose_spread_deg: f32,
    /// Ball center ahead of the car center, along the nose.
    pub ball_ahead: [f32; 2],
    /// Ball center above the car center, along the car's up.
    pub ball_above: [f32; 2],
    pub dribbler_boost: [f32; 2],
    /// Share of challenges on the same side wall. Normalized with the other two.
    pub same_wall_frac: f32,
    /// Share on the back wall by the challenger's own goal.
    pub back_wall_frac: f32,
    /// Share on the opposite side wall.
    pub opposite_wall_frac: f32,
    pub challenger_height: [f32; 2],
    /// Distance from the ball toward the challenger's goal, along the field.
    pub challenger_dist: [f32; 2],
    /// Speed along the wall.
    pub challenger_speed: [f32; 2],
    /// Cone of the challenger's nose around the ball, in degrees.
    pub face_spread_deg: f32,
    pub challenger_boost: [f32; 2],
}

impl Default for AirDribbleWallChallenge {
    fn default() -> Self {
        Self {
            dribbler_height: [400.0, 1400.0],
            off_wall: [50.0, 250.0],
            into_field_frac: 0.35,
            into_field: [500.0, 1600.0],
            speed: [900.0, 1700.0],
            nose_spread_deg: 25.0,
            ball_ahead: [120.0, 180.0],
            ball_above: [40.0, 80.0],
            dribbler_boost: [20.0, 80.0],
            same_wall_frac: 0.60,
            back_wall_frac: 0.25,
            opposite_wall_frac: 0.15,
            challenger_height: [300.0, 1200.0],
            challenger_dist: [600.0, 2000.0],
            challenger_speed: [0.0, 600.0],
            face_spread_deg: 20.0,
            challenger_boost: [30.0, 100.0],
        }
    }
}

/// Up axis for a car whose nose is `forward` and whose roof points mostly up.
fn upright_from_forward(forward: Vec3A) -> Vec3A {
    let forward = forward.normalize();
    let right = forward.cross(Vec3A::Z);
    let right = if right.length() < 1e-4 {
        forward.cross(Vec3A::X).normalize()
    } else {
        right.normalize()
    };
    right.cross(forward).normalize()
}

/// Move `a` and `b` by the same xy delta so both sit inside the walls.
fn shift_pair_inside(a: Vec3A, b: Vec3A) -> (Vec3A, Vec3A) {
    let lim_x = SIDE_WALL_X - 1.0;
    let lim_y = BACK_WALL_Y - 1.0;
    let mut dx: f32 = 0.0;
    let mut dy: f32 = 0.0;
    for p in [a, b] {
        if p.x > lim_x {
            dx = dx.min(lim_x - p.x);
        }
        if p.x < -lim_x {
            dx = dx.max(-lim_x - p.x);
        }
        if p.y > lim_y {
            dy = dy.min(lim_y - p.y);
        }
        if p.y < -lim_y {
            dy = dy.max(-lim_y - p.y);
        }
    }
    let d = Vec3A::new(dx, dy, 0.0);
    (a + d, b + d)
}

impl<SI: SharedInfoRng> StateSetter<SI> for AirDribbleWallChallenge {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let cars = arena.num_cars().max(1);
        let dribbler_idx = rng.random_range(0..cars.min(2));
        let challenger_idx = 1 - dribbler_idx;
        // Blue (car 0) attacks +y. Orange attacks -y. The challenger defends
        // the goal the dribbler is driving at.
        let attack_sign: f32 = if dribbler_idx == 0 { 1.0 } else { -1.0 };
        let wall_side: f32 = if rng.random_bool(0.5) { 1.0 } else { -1.0 };

        let same = self.same_wall_frac.max(0.0);
        let back = self.back_wall_frac.max(0.0);
        let opposite = self.opposite_wall_frac.max(0.0);
        let sum = (same + back + opposite).max(f32::EPSILON);
        let draw = rng.random_range(0.0..sum);
        let wall = if draw < same {
            ChallengeWall::Same
        } else if draw < same + back {
            ChallengeWall::Back
        } else {
            ChallengeWall::Opposite
        };

        let into = rng.random_bool(self.into_field_frac.clamp(0.0, 1.0) as f64);
        let off = if into {
            range(rng, self.into_field)
        } else {
            range(rng, self.off_wall)
        };
        let attack = Vec3A::new(0.0, attack_sign, 0.0);
        let forward = tilted_toward(rng, attack, self.nose_spread_deg.to_radians());
        let up = upright_from_forward(forward);
        let ahead = range(rng, self.ball_ahead);
        let above = range(rng, self.ball_above);
        let speed = range(rng, self.speed);
        let dist = range(rng, self.challenger_dist);

        // Anchor the car, not the ball, so the dribbler is the one 50+ uu off
        // the side wall. The ball sits on the nose and may be closer to the wall.
        let car_x = wall_side * (SIDE_WALL_X - off);
        let car_z = range(rng, self.dribbler_height);
        let car_y = if matches!(wall, ChallengeWall::Back) {
            let ball_y = attack_sign * (BACK_WALL_Y - CAR_REST_HEIGHT - dist);
            ball_y - forward.y * ahead - up.y * above
        } else {
            attack_sign * rng.random_range(-1200.0..2400.0)
        };
        let dribbler_pos = Vec3A::new(car_x, car_y, car_z);
        let ball_pos = dribbler_pos + forward * ahead + up * above;
        let (dribbler_pos, ball_pos) = shift_pair_inside(dribbler_pos, ball_pos);

        let vel = forward * speed;
        let mut dribbler = CarState::DEFAULT;
        dribbler.phys.pos = dribbler_pos;
        dribbler.phys.vel = vel;
        dribbler.phys.rot_mat = Mat3A::from_cols(forward, up.cross(forward), up);
        dribbler.boost = range(rng, self.dribbler_boost);
        dribbler.has_jumped = true;
        arena.set_ball_state(ball_at(ball_pos, vel));
        arena.set_car_state(dribbler_idx, dribbler.core());

        if challenger_idx >= arena.num_cars() {
            return;
        }
        let (ch_pos, ch_up) = match wall {
            ChallengeWall::Same => {
                let x = wall_side * (SIDE_WALL_X - CAR_REST_HEIGHT);
                let y = ball_pos.y + attack_sign * dist;
                (
                    Vec3A::new(x, y, range(rng, self.challenger_height)),
                    Vec3A::new(-wall_side, 0.0, 0.0),
                )
            }
            ChallengeWall::Opposite => {
                let side = -wall_side;
                let x = side * (SIDE_WALL_X - CAR_REST_HEIGHT);
                let y = ball_pos.y + attack_sign * dist;
                (
                    Vec3A::new(x, y, range(rng, self.challenger_height)),
                    Vec3A::new(-side, 0.0, 0.0),
                )
            }
            ChallengeWall::Back => {
                let y = attack_sign * (BACK_WALL_Y - CAR_REST_HEIGHT);
                let x = wall_side * rng.random_range(1400.0..3600.0);
                (
                    Vec3A::new(x, y, range(rng, self.challenger_height)),
                    Vec3A::new(0.0, -attack_sign, 0.0),
                )
            }
        };
        let lim_y = BACK_WALL_Y - CAR_REST_HEIGHT;
        let ch_pos = Vec3A::new(
            ch_pos.x.clamp(-(SIDE_WALL_X - CAR_REST_HEIGHT), SIDE_WALL_X - CAR_REST_HEIGHT),
            ch_pos.y.clamp(-lim_y, lim_y),
            ch_pos.z.clamp(
                self.challenger_height[0],
                self.challenger_height[1].max(self.challenger_height[0]),
            ),
        );
        let to_ball = ball_pos - ch_pos;
        let in_plane = to_ball - ch_up * to_ball.dot(ch_up);
        let base = if in_plane.length() < 1e-3 {
            ch_up.cross(Vec3A::Z).normalize()
        } else {
            in_plane.normalize()
        };
        let ang = rng.random_range(-1.0..1.0) * self.face_spread_deg.to_radians();
        let ch_forward = (base * ang.cos() + ch_up.cross(base) * ang.sin()).normalize();
        let tangent = ch_up.cross(Vec3A::Z);
        let tangent = if tangent.length() < 1e-4 {
            Vec3A::X
        } else {
            tangent.normalize()
        };
        let along = tangent.dot(to_ball).signum();
        let along = if along == 0.0 { 1.0 } else { along };
        let mut challenger = CarState::DEFAULT;
        challenger.phys.pos = ch_pos;
        challenger.phys.vel = tangent * along * range(rng, self.challenger_speed);
        challenger.phys.rot_mat = Mat3A::from_cols(ch_forward, ch_up.cross(ch_forward), ch_up);
        challenger.boost = range(rng, self.challenger_boost);
        challenger.has_jumped = false;
        arena.set_car_state(challenger_idx, challenger.core());
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rlgymppo_utils::rlgym::rocketsim::Vec3A;
    use rlgymppo_utils::shared_info::SharedInfoRng;

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
        assert!(
            parse("kind = \"air_dribble_wall_challenge\"\nweight = 1.0\nnope = 1.0").is_err()
        );
        assert_eq!(
            parse("kind = \"air_dribble_wall_challenge\"\nweight = 1.0")
                .unwrap()
                .kind,
            SetterKind::AirDribbleWallChallenge(AirDribbleWallChallenge::default())
        );
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

    #[test]
    fn aerial_defense_invariants() {
        use rlgymppo_utils::rlgym::rocketsim::{Arena, GameMode, init_from_default};
        use rlgymppo_utils::shared_info::SharedInfoRng;
        
        struct TestSharedInfo {
            rng: SmallRng,
        }
        
        impl SharedInfoRng for TestSharedInfo {
            type Rng = SmallRng;
            fn rng(&mut self) -> &mut SmallRng {
                &mut self.rng
            }
        }
        
        use rlgymppo_utils::rlgym::rocketsim::{CarBodyConfig, Team};
        
        init_from_default(true).unwrap();
        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
        let mut setter = AerialDefense::default();
        let mut shared_info = TestSharedInfo {
            rng: SmallRng::seed_from_u64(42),
        };
        
        let mut defender_0_count = 0;
        
        for _ in 0..10_000 {
            setter.apply(&mut arena, &mut shared_info);
            
            let ball = arena.get_ball_state();
            let car0 = arena.get_car_state(0);
            let car1 = arena.get_car_state(1);
            
            // Ball height in range 300-1500
            assert!(
                ball.phys.pos.z >= 300.0 && ball.phys.pos.z <= 1500.0,
                "ball z out of range: {}",
                ball.phys.pos.z
            );
            
            // Ball inside field
            assert!(
                ball.phys.pos.x.abs() <= 4096.0 && ball.phys.pos.y.abs() <= 5120.0,
                "ball outside field: ({}, {})",
                ball.phys.pos.x,
                ball.phys.pos.y
            );
            
            // Ball must be moving toward one of the two goals
            let goal_blue: f32 = -5120.0;
            let goal_orange: f32 = 5120.0;
            let moving_to_blue = (goal_blue - ball.phys.pos.y).signum() == ball.phys.vel.y.signum();
            let moving_to_orange = (goal_orange - ball.phys.pos.y).signum() == ball.phys.vel.y.signum();
            assert!(
                moving_to_blue || moving_to_orange,
                "ball not moving toward either goal: ball_pos.y={}, ball_vel.y={}",
                ball.phys.pos.y,
                ball.phys.vel.y
            );
            
            // Determine which goal the ball is moving toward
            let (defended_goal, _attacking_goal) = if moving_to_blue {
                (goal_blue, goal_orange)
            } else {
                (goal_orange, goal_blue)
            };
            
            // Determine attacker and defender based on which goal is being attacked
            // The attacker should be farther from the defended goal than the ball
            let car0_dist = (car0.phys.pos.y - defended_goal).abs();
            let car1_dist = (car1.phys.pos.y - defended_goal).abs();
            let ball_dist = (ball.phys.pos.y - defended_goal).abs();
            
            let (attacker, _defender, defender_idx) = if car0_dist > car1_dist {
                (&car0, &car1, 1)
            } else {
                (&car1, &car0, 0)
            };
            
            if defender_idx == 0 {
                defender_0_count += 1;
            }
            
            // Attacker farther from defended goal than ball
            let attacker_dist_from_goal = (attacker.phys.pos.y - defended_goal).abs();
            assert!(
                attacker_dist_from_goal > ball_dist,
                "attacker not farther from goal than ball: attacker_dist={}, ball_dist={}",
                attacker_dist_from_goal,
                ball_dist
            );
            
            // Both cars in bounds with z >= 17
            for (idx, car) in [(0, &car0), (1, &car1)] {
                assert!(
                    car.phys.pos.x.abs() <= 4096.0 && car.phys.pos.y.abs() <= 5120.0 && car.phys.pos.z >= CAR_REST_HEIGHT,
                    "car {} outside bounds: ({}, {}, {})",
                    idx,
                    car.phys.pos.x,
                    car.phys.pos.y,
                    car.phys.pos.z
                );
            }
        }
        
        // Defender car index should be randomized ~50/50
        let defender_0_frac = defender_0_count as f32 / 10_000.0;
        assert!(
            defender_0_frac >= 0.45 && defender_0_frac <= 0.55,
            "defender selection not balanced: {:.2}% chose car 0",
            defender_0_frac * 100.0
        );
    }

    #[test]
    fn demo_threat_invariants() {
        use rlgymppo_utils::rlgym::rocketsim::{Arena, GameMode, init_from_default};
        use rlgymppo_utils::shared_info::SharedInfoRng;
        
        struct TestSharedInfo {
            rng: SmallRng,
        }
        
        impl SharedInfoRng for TestSharedInfo {
            type Rng = SmallRng;
            fn rng(&mut self) -> &mut SmallRng {
                &mut self.rng
            }
        }
        
        use rlgymppo_utils::rlgym::rocketsim::{CarBodyConfig, Team};
        
        init_from_default(true).unwrap();
        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
        let mut setter = DemoThreat::default();
        let mut shared_info = TestSharedInfo {
            rng: SmallRng::seed_from_u64(123),
        };
        
        let mut target_0_count = 0;
        
        for _ in 0..10_000 {
            setter.apply(&mut arena, &mut shared_info);
            
            let car0 = arena.get_car_state(0);
            let car1 = arena.get_car_state(1);
            
            // Determine which car is attacker (supersonic) and which is target
            let speed0 = car0.phys.vel.length();
            let speed1 = car1.phys.vel.length();
            
            let (attacker, target, target_idx) = if speed0 > speed1 {
                (&car0, &car1, 1)
            } else {
                (&car1, &car0, 0)
            };
            
            if target_idx == 0 {
                target_0_count += 1;
            }
            
            let speed = attacker.phys.vel.length();
            
            // Attacker supersonic >= 2200
            assert!(
                speed >= 2200.0,
                "attacker not supersonic: speed={}",
                speed
            );
            
            // Both cars in bounds
            for (idx, car) in [(0, &car0), (1, &car1)] {
                assert!(
                    car.phys.pos.x.abs() <= 4096.0 && car.phys.pos.y.abs() <= 5120.0 && car.phys.pos.z >= CAR_REST_HEIGHT,
                    "car {} outside bounds: ({}, {}, {})",
                    idx,
                    car.phys.pos.x,
                    car.phys.pos.y,
                    car.phys.pos.z
                );
            }
            
            // Attacker heading at target
            let to_target = (target.phys.pos - attacker.phys.pos).normalize();
            let heading = attacker.phys.vel.normalize();
            let heading_dot = heading.dot(to_target);
            assert!(
                heading_dot > 0.95,
                "attacker not heading at target: dot={}",
                heading_dot
            );

            // Legacy approach is the midfield ray: attacker and target are
            // colinear with the origin (clamp can move a car by a few uu).
            let cross = attacker.phys.pos.x * target.phys.pos.y
                - attacker.phys.pos.y * target.phys.pos.x;
            assert!(
                cross.abs() < 5.0,
                "legacy approach not radial from midfield: cross={cross}"
            );
            // Static ball, and the keeper is never close to the goal line.
            let ball = arena.get_ball_state();
            assert!(ball.phys.vel.length() < 1.0, "legacy ball is moving");
            assert!((ball.phys.pos.z - 100.0).abs() < 1.0, "legacy ball z");
            let goal_y = if target_idx == 0 { -5120.0 } else { 5120.0 };
            let goal_gap = (target.phys.pos.y - goal_y).abs();
            assert!(
                goal_gap >= 1120.0 - 1.0,
                "legacy keeper closer than 1120 uu to the goal: {goal_gap}"
            );
        }
        
        // Target car index should be randomized ~50/50
        let target_0_frac = target_0_count as f32 / 10_000.0;
        assert!(
            target_0_frac >= 0.45 && target_0_frac <= 0.55,
            "target selection not balanced: {:.2}% chose car 0",
            target_0_frac * 100.0
        );
    }

    fn demo_arena(seed: u64) -> (rlgymppo_utils::rlgym::rocketsim::Arena, DemoRng) {
        use rlgymppo_utils::rlgym::rocketsim::{Arena, CarBodyConfig, GameMode, Team, init_from_default};
        init_from_default(true).unwrap();
        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
        (
            arena,
            DemoRng {
                rng: SmallRng::seed_from_u64(seed),
            },
        )
    }

    struct DemoRng {
        rng: SmallRng,
    }

    impl SharedInfoRng for DemoRng {
        type Rng = SmallRng;
        fn rng(&mut self) -> &mut SmallRng {
            &mut self.rng
        }
    }

    fn st32_demo() -> DemoThreat {
        DemoThreat {
            distance: [1800.0, 3200.0],
            target_goal_dist: Some([250.0, 900.0]),
            attacker_speed_range: Some([1950.0, 2150.0]),
            ball_threat: Some(true),
            ..DemoThreat::default()
        }
    }

    #[test]
    fn demo_threat_toml_is_backwards_compatible_and_accepts_st32() {
        let legacy: WeightedSetter = toml::from_str(
            r#"
            kind = "demo_threat"
            weight = 0.15
            distance = [1800.0, 3000.0]
            "#,
        )
        .unwrap();
        let SetterKind::DemoThreat(demo) = legacy.kind else {
            panic!("kind");
        };
        assert_eq!(demo.distance, [1800.0, 3000.0]);
        assert_eq!(demo.attacker_speed, [2250.0, 2300.0]);
        assert_eq!(demo.target_goal_dist, None);
        assert_eq!(demo.attacker_speed_range, None);
        assert_eq!(demo.ball_threat, None);
        assert!(demo.use_legacy());

        let bare: WeightedSetter = toml::from_str("kind = \"demo_threat\"\nweight = 0.05").unwrap();
        assert_eq!(bare.kind, SetterKind::DemoThreat(DemoThreat::default()));

        let st32: WeightedSetter = toml::from_str(
            r#"
            kind = "demo_threat"
            weight = 0.15
            dist_range = [1800.0, 3200.0]
            target_goal_dist = [250.0, 900.0]
            attacker_speed_range = [1950.0, 2150.0]
            ball_threat = true
            "#,
        )
        .unwrap();
        let SetterKind::DemoThreat(demo) = st32.kind else {
            panic!("kind");
        };
        assert_eq!(demo.distance, [1800.0, 3200.0]);
        assert_eq!(demo.target_goal_dist, Some([250.0, 900.0]));
        assert_eq!(demo.attacker_speed_range, Some([1950.0, 2150.0]));
        assert_eq!(demo.ball_threat, Some(true));
        assert!(!demo.use_legacy());

        let off: WeightedSetter = toml::from_str(
            "kind = \"demo_threat\"\nweight = 0.15\nball_threat = false\ndistance = [1800.0, 3000.0]",
        )
        .unwrap();
        let SetterKind::DemoThreat(demo) = off.kind else {
            panic!("kind");
        };
        assert_eq!(demo.ball_threat, Some(false));
        assert!(demo.use_legacy());

        assert!(
            toml::from_str::<WeightedSetter>("kind = \"demo_threat\"\nweight = 1.0\nnope = 1.0")
                .is_err()
        );
    }

    #[test]
    fn demo_threat_st32_is_a_saveable_goal_side_threat() {
        let (mut arena, mut shared) = demo_arena(32);
        let mut setter = st32_demo();
        let mut target_0 = 0;
        for _ in 0..2_000 {
            setter.apply(&mut arena, &mut shared);
            let ball = arena.get_ball_state();
            let car0 = arena.get_car_state(0);
            let car1 = arena.get_car_state(1);
            let (attacker, keeper, keeper_idx) = if car0.phys.vel.length() > car1.phys.vel.length() {
                (&car0, &car1, 1)
            } else {
                (&car1, &car0, 0)
            };
            if keeper_idx == 0 {
                target_0 += 1;
            }
            let goal_y = if keeper_idx == 0 { -5120.0 } else { 5120.0 };
            let goal_sign = if goal_y < 0.0 { -1.0 } else { 1.0 };

            for (idx, car) in [(0, &car0), (1, &car1)] {
                assert!(
                    car.phys.pos.x.abs() <= 4096.0
                        && car.phys.pos.y.abs() <= 5120.0
                        && car.phys.pos.z >= CAR_REST_HEIGHT - 0.1,
                    "car {idx} out of bounds: {:?}",
                    car.phys.pos
                );
            }
            assert!(
                ball.phys.pos.x.abs() <= 4096.0
                    && ball.phys.pos.y.abs() <= 5120.0
                    && ball.phys.pos.z > 90.0,
                "ball out of bounds: {:?}",
                ball.phys.pos
            );

            let goal_gap = (keeper.phys.pos.y - goal_y).abs();
            assert!(
                (250.0..900.0).contains(&goal_gap),
                "keeper goal gap {goal_gap}"
            );
            assert!(keeper.phys.pos.x.abs() < 900.0, "keeper x {}", keeper.phys.pos.x);
            assert!(
                (keeper.phys.pos.z - CAR_REST_HEIGHT).abs() < 0.1,
                "keeper not grounded"
            );
            assert!(
                (50.0..100.0).contains(&keeper.boost),
                "keeper boost {}",
                keeper.boost
            );
            assert!(keeper.phys.vel.length() < 1.0, "keeper already moving");

            let to_ball = ball.phys.pos - keeper.phys.pos;
            let to_ball_xy = Vec3A::new(to_ball.x, to_ball.y, 0.0).normalize();
            assert!(
                keeper.phys.rot_mat.x_axis.dot(to_ball_xy) > 0.95,
                "keeper not facing the ball"
            );

            let speed = attacker.phys.vel.length();
            assert!(
                (1950.0..2150.0).contains(&speed),
                "closing speed {speed}"
            );
            let to_keeper = (keeper.phys.pos - attacker.phys.pos).normalize();
            assert!(
                attacker.phys.vel.normalize().dot(to_keeper) > 0.95,
                "attacker not heading at the keeper"
            );
            assert!(
                attacker.phys.rot_mat.x_axis.dot(attacker.phys.vel.normalize()) > 0.95,
                "attacker not facing its velocity"
            );
            let from_keeper = attacker.phys.pos - keeper.phys.pos;
            let from_xy = Vec3A::new(from_keeper.x, from_keeper.y, 0.0).normalize();
            let upfield = Vec3A::new(0.0, -goal_sign, 0.0);
            let cone = from_xy.dot(upfield);
            assert!(
                cone >= DEMO_APPROACH_CONE_DEG.to_radians().cos() - 1e-3,
                "approach outside the cone: dot={cone}"
            );
            let sep = (attacker.phys.pos - keeper.phys.pos).length();
            assert!((1800.0..3400.0).contains(&sep), "separation {sep}");
            let react = sep / speed;
            assert!(
                (0.8..1.8).contains(&react),
                "reaction window {react} s"
            );

            let ball_from_goal = (ball.phys.pos.y - goal_y).abs();
            assert!(
                (1800.0..3200.0).contains(&ball_from_goal),
                "ball dist {ball_from_goal}"
            );
            assert!(
                ball_from_goal > goal_gap + 500.0,
                "keeper not goal-side of the ball"
            );
            assert!(
                ball.phys.vel.y * goal_sign > 0.0,
                "ball vel.y {} not toward goal {}",
                ball.phys.vel.y,
                goal_sign
            );
            let horiz = Vec3A::new(ball.phys.vel.x, ball.phys.vel.y, 0.0).length();
            assert!(
                horiz >= 900.0 - 1e-2 && horiz <= 1450.0 + 1e-2,
                "ball horizontal speed {horiz}"
            );
            assert!(ball.phys.vel.z > 0.0, "ball has no loft");
            let eta = ball_from_goal / ball.phys.vel.y.abs();
            assert!((1.0..2.5).contains(&eta), "ball eta {eta} s");
            let t_cross = (goal_y - ball.phys.pos.y) / ball.phys.vel.y;
            let cross_x = ball.phys.pos.x + ball.phys.vel.x * t_cross;
            assert!(
                cross_x.abs() < 800.0,
                "shot misses the mouth: cross_x={cross_x}"
            );
        }
        let frac = target_0 as f32 / 2_000.0;
        assert!((0.45..0.55).contains(&frac), "keeper side balance {frac}");
    }

    #[test]
    fn demo_threat_honors_explicit_subsonic_attacker_speed() {
        let (mut arena, mut shared) = demo_arena(7);
        let mut setter = DemoThreat {
            attacker_speed: [1950.0, 2150.0],
            ..DemoThreat::default()
        };
        assert!(!setter.use_legacy());
        for _ in 0..200 {
            setter.apply(&mut arena, &mut shared);
            let car0 = arena.get_car_state(0);
            let car1 = arena.get_car_state(1);
            let speed = car0.phys.vel.length().max(car1.phys.vel.length());
            assert!(
                (1950.0..2150.0).contains(&speed),
                "explicit attacker_speed was clamped: {speed}"
            );
            assert!(
                car0.phys.pos.x.abs() <= 4096.0
                    && car0.phys.pos.y.abs() <= 5120.0
                    && car1.phys.pos.x.abs() <= 4096.0
                    && car1.phys.pos.y.abs() <= 5120.0,
                "car outside the field"
            );
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

#[test]
fn air_dribble_wall_challenge_invariants() {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rlgymppo_utils::rlgym::rocketsim::{Arena, CarBodyConfig, GameMode, Team, init_from_default};
    use rlgymppo_utils::shared_info::SharedInfoRng;

    struct TestSharedInfo {
        rng: SmallRng,
    }

    impl SharedInfoRng for TestSharedInfo {
        type Rng = SmallRng;
        fn rng(&mut self) -> &mut SmallRng {
            &mut self.rng
        }
    }

    init_from_default(true).unwrap();
    let mut arena = Arena::new(GameMode::Soccar);
    arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    let mut setter = AirDribbleWallChallenge::default();
    let mut shared_info = TestSharedInfo {
        rng: SmallRng::seed_from_u64(7),
    };

    let on_side = |x: f32| (SIDE_WALL_X - x.abs() - CAR_REST_HEIGHT).abs() < 2.0;
    let on_back = |y: f32| (BACK_WALL_Y - y.abs() - CAR_REST_HEIGHT).abs() < 2.0;
    let in_bounds = |p: Vec3A| {
        p.x.abs() <= SIDE_WALL_X && p.y.abs() <= BACK_WALL_Y && p.z >= 0.0 && p.z <= 2044.0
    };

    let mut dribbler_0 = 0;
    for _ in 0..10_000 {
        setter.apply(&mut arena, &mut shared_info);
        let ball = arena.get_ball_state().phys.pos;
        let cars = [arena.get_car_state(0), arena.get_car_state(1)];
        assert!(in_bounds(ball), "ball out of bounds: {:?}", ball);
        for (idx, car) in cars.iter().enumerate() {
            assert!(
                in_bounds(car.phys.pos),
                "car {idx} out of bounds: {:?}",
                car.phys.pos
            );
        }
        let wheels_on_wall = |car: &rlgymppo_utils::rlgym::rocketsim::CarState| {
            let up = car.phys.rot_mat.z_axis;
            (on_side(car.phys.pos.x) && up.x * -car.phys.pos.x.signum() > 0.9)
                || (on_back(car.phys.pos.y) && up.y * -car.phys.pos.y.signum() > 0.9)
        };
        let c0_wall = wheels_on_wall(&cars[0]);
        let c1_wall = wheels_on_wall(&cars[1]);
        assert!(
            c0_wall ^ c1_wall,
            "expected exactly one car on a wall: c0 {:?} wall {c0_wall} c1 {:?} wall {c1_wall} ball {:?}",
            cars[0].phys.pos,
            cars[1].phys.pos,
            ball
        );
        let (challenger, dribbler, dribbler_idx) = if c0_wall {
            (&cars[0], &cars[1], 1)
        } else {
            (&cars[1], &cars[0], 0)
        };
        if dribbler_idx == 0 {
            dribbler_0 += 1;
        }
        assert!(
            dribbler.phys.pos.z >= 400.0 && dribbler.phys.pos.z <= 1400.0,
            "dribbler not airborne: z={}",
            dribbler.phys.pos.z
        );
        assert!(
            (ball - dribbler.phys.pos).length() <= 250.0,
            "ball not on the nose: {}",
            (ball - dribbler.phys.pos).length()
        );
        let up = challenger.phys.rot_mat.z_axis;
        if on_side(challenger.phys.pos.x) {
            let away = -challenger.phys.pos.x.signum();
            assert!(up.x * away > 0.9, "side-wall up not away from the wall: {up:?}");
        } else {
            let away = -challenger.phys.pos.y.signum();
            assert!(up.y * away > 0.9, "back-wall up not away from the wall: {up:?}");
        }
        let toward = (challenger.phys.pos.y - ball.y).signum();
        let goal = toward * BACK_WALL_Y;
        assert!(
            toward != 0.0 && (challenger.phys.pos.y - goal).abs() < (ball.y - goal).abs(),
            "challenger not goal-side of the ball: car {} ball {} goal {goal}",
            challenger.phys.pos.y,
            ball.y
        );
        assert!(!challenger.has_jumped, "challenger already used its jump");
    }
    let frac = dribbler_0 as f32 / 10_000.0;
    assert!(
        (0.45..=0.55).contains(&frac),
        "dribbler selection not balanced: {frac:.3}"
    );
}
