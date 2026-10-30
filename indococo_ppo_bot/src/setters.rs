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

use std::f32::consts::PI;

use rand::RngExt;
use rlgymppo_utils::rlgym::StateSetter;
use rlgymppo_utils::rlgym::rocketsim::{Arena, BallState, CarState, Mat3A, Vec3A};
use rlgymppo_utils::shared_info::SharedInfoRng;
use rlgymppo_utils::state_setters::{KickoffState, RandomState, WeightedState};
use serde::Deserialize;

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

/// Builds the weighted setter. Entries with weight 0 are dropped.
///
/// # Panics
/// If no entry has a positive weight.
#[must_use]
pub fn build_mix<SI: SharedInfoRng + 'static>(mix: &[WeightedSetter]) -> WeightedState<SI> {
    let setters: Vec<(Box<dyn StateSetter<SI> + Send>, f32)> = mix
        .iter()
        .filter(|s| s.weight > 0.0)
        .map(|s| (s.kind.build::<SI>(), s.weight))
        .collect();
    assert!(
        !setters.is_empty(),
        "state_setters: no entry with a positive weight"
    );
    WeightedState::new(setters)
}

impl SetterKind {
    fn build<SI: SharedInfoRng + 'static>(&self) -> Box<dyn StateSetter<SI> + Send> {
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

fn range(rng: &mut impl RngExt, [lo, hi]: [f32; 2]) -> f32 {
    if hi > lo {
        rng.random_range(lo..hi)
    } else {
        lo
    }
}

fn ball_at(pos: Vec3A, vel: Vec3A) -> BallState {
    let mut ball = BallState::DEFAULT;
    ball.phys.pos = pos;
    ball.phys.vel = vel;
    ball
}

/// Unit vector in the xy plane.
fn horizontal(rng: &mut impl RngExt) -> Vec3A {
    let a = rng.random_range(-PI..PI);
    Vec3A::new(a.cos(), a.sin(), 0.0)
}

/// Unit vector within `max_tilt` radians of +z, uniform in azimuth.
fn tilted_up(rng: &mut impl RngExt, max_tilt: f32) -> Vec3A {
    let tilt = rng.random_range(0.0..=max_tilt);
    horizontal(rng) * tilt.sin() + Vec3A::Z * tilt.cos()
}

/// A rotation whose up axis is `up`, with a random heading around it.
fn orientation(rng: &mut impl RngExt, up: Vec3A) -> Mat3A {
    let helper = if up.z.abs() < 0.9 { Vec3A::Z } else { Vec3A::X };
    let a = up.cross(helper).normalize();
    let b = up.cross(a);
    let angle = rng.random_range(-PI..PI);
    let forward = a * angle.cos() + b * angle.sin();
    Mat3A::from_cols(forward, up.cross(forward), up)
}

/// A car resting on the field at a random spot and heading.
fn ground_car(rng: &mut impl RngExt) -> CarState {
    let mut car = CarState::DEFAULT;
    car.phys.pos = Vec3A::new(
        rng.random_range(-3500.0..3500.0),
        rng.random_range(-4500.0..4500.0),
        CAR_REST_HEIGHT,
    );
    car.phys.rot_mat = orientation(rng, Vec3A::Z);
    car
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
