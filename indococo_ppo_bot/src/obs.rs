//! Observation builder: RLGym v2 `DefaultObs(zero_padding=None)` with the
//! `v4_env` coefficients, matching the input of `1v1.pt`.
//!
//! 1v1 layout (92): ball (9) + boost pad timers (34) + own flags (9) + own
//! car (20) + opponent car (20). Orange sees the field mirrored in x and y.
//!
//! `air_time_since_jump` is clamped to 1.25 s as in
//! `v4_env._obs_builder(clamp_air_time=True)`: RLBot's compat layer freezes
//! it there, RocketSim keeps counting, so the clamp keeps training and
//! deployment inputs identical.
use rlgymppo_utils::rlgym::rocketsim::{CarState, Team, Vec3A};
use rlgymppo_utils::rlgym::{FullObs, GameState, Obs};

pub const POS_COEF: [f32; 3] = [1.0 / 4096.0, 1.0 / 6000.0, 1.0 / 2044.0]; // SIDE_WALL_X, BACK_NET_Y, CEILING_Z
pub const LIN_VEL_COEF: f32 = 1.0 / 2300.0;
pub const ANG_VEL_COEF: f32 = 1.0 / 5.5;
pub const PAD_TIMER_COEF: f32 = 1.0 / 10.0;
pub const BOOST_COEF: f32 = 1.0 / 100.0;
pub const DOUBLEJUMP_MAX_DELAY: f32 = 1.25;
pub const FLIP_TORQUE_TIME: f32 = 0.65;

#[inline]
fn inv(v: Vec3A, invert: bool) -> Vec3A {
    if invert { Vec3A::new(-v.x, -v.y, v.z) } else { v }
}
#[inline]
fn push_pos(o: &mut Vec<f32>, p: Vec3A) {
    o.push(p.x * POS_COEF[0]);
    o.push(p.y * POS_COEF[1]);
    o.push(p.z * POS_COEF[2]);
}
#[inline]
fn push_scaled(o: &mut Vec<f32>, v: Vec3A, s: f32) {
    o.push(v.x * s);
    o.push(v.y * s);
    o.push(v.z * s);
}

/// Fields of the python `rlgym.rocket_league.api.Car` derived from a RocketSim CarState.
pub struct CarFlags {
    pub on_ground: bool,
    pub is_boosting: bool,
    pub is_supersonic: bool,
    pub is_holding_jump: bool,
    pub has_flip: bool,
    pub can_flip: bool,
    pub is_flipping: bool,
}
pub fn flags(c: &CarState) -> CarFlags {
    // RLGym v2 `Car.on_ground`: at least three wheels in contact (not RocketSim's `is_on_ground`).
    let on_ground = c.wheels_with_contact().iter().filter(|w| w.is_some()).count() >= 3;
    let has_flip = !c.has_double_jumped && !c.has_flipped && c.air_time_since_jump < DOUBLEJUMP_MAX_DELAY;
    let is_holding_jump = c.controls.jump;
    CarFlags {
        on_ground,
        is_boosting: c.boosting_time > 0.0,
        is_supersonic: c.is_supersonic,
        is_holding_jump,
        has_flip,
        can_flip: !on_ground && !is_holding_jump && has_flip,
        is_flipping: c.has_flipped && c.flip_time < FLIP_TORQUE_TIME,
    }
}

fn car_obs(o: &mut Vec<f32>, c: &CarState, invert: bool) {
    let f = flags(c);
    push_pos(o, inv(c.phys.pos, invert));
    let fwd = inv(c.phys.rot_mat.x_axis, invert);
    let up = inv(c.phys.rot_mat.z_axis, invert);
    o.extend([fwd.x, fwd.y, fwd.z, up.x, up.y, up.z]);
    push_scaled(o, inv(c.phys.vel, invert), LIN_VEL_COEF);
    push_scaled(o, inv(c.phys.ang_vel, invert), ANG_VEL_COEF);
    o.push(c.boost * BOOST_COEF);
    o.push(c.demo_respawn_timer);
    o.push(f32::from(f.on_ground));
    o.push(f32::from(f.is_boosting));
    o.push(f32::from(f.is_supersonic));
}

/// rlgym `common_values.BOOST_LOCATIONS` (x, y) order; RocketSim pads are matched by nearest position.
pub const BOOST_LOCATIONS: [(f32, f32); 34] = [
    (0.0, -4240.0), (-1792.0, -4184.0), (1792.0, -4184.0), (-3072.0, -4096.0), (3072.0, -4096.0),
    (-940.0, -3308.0), (940.0, -3308.0), (0.0, -2816.0), (-3584.0, -2484.0), (3584.0, -2484.0),
    (-1788.0, -2300.0), (1788.0, -2300.0), (-2048.0, -1036.0), (0.0, -1024.0), (2048.0, -1036.0),
    (-3584.0, 0.0), (-1024.0, 0.0), (1024.0, 0.0), (3584.0, 0.0), (-2048.0, 1036.0), (0.0, 1024.0),
    (2048.0, 1036.0), (-1788.0, 2300.0), (1788.0, 2300.0), (-3584.0, 2484.0), (3584.0, 2484.0),
    (0.0, 2816.0), (-940.0, 3310.0), (940.0, 3308.0), (-3072.0, 4096.0), (3072.0, 4096.0),
    (-1792.0, 4184.0), (1792.0, 4184.0), (0.0, 4240.0),
];

/// RocketSim pad index for each rlgym BOOST_LOCATIONS entry.
pub fn pad_order(state: &GameState) -> Vec<usize> {
    BOOST_LOCATIONS
        .iter()
        .map(|&(x, y)| {
            (0..state.boost_pads.len())
                .min_by(|&a, &b| {
                    let pa = state.boost_pads[a].0.pos;
                    let pb = state.boost_pads[b].0.pos;
                    let da = (pa.x - x).powi(2) + (pa.y - y).powi(2);
                    let db = (pb.x - x).powi(2) + (pb.y - y).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap()
        })
        .collect()
}

pub fn build_one(state: &GameState, i: usize, order: &[usize]) -> Vec<f32> {
    let (info, car) = &state.cars[i];
    let invert = info.team == Team::Orange;
    let mut o = Vec::with_capacity(92);
    let b = &state.ball;
    push_pos(&mut o, inv(b.pos, invert));
    push_scaled(&mut o, inv(b.vel, invert), LIN_VEL_COEF);
    push_scaled(&mut o, inv(b.ang_vel, invert), ANG_VEL_COEF);
    let timers: Vec<f32> = order.iter().map(|&k| state.boost_pads[k].1.cooldown).collect();
    if invert {
        o.extend(timers.iter().rev().map(|t| t * PAD_TIMER_COEF));
    } else {
        o.extend(timers.iter().map(|t| t * PAD_TIMER_COEF));
    }
    let f = flags(car);
    o.extend([
        f32::from(f.is_holding_jump),
        car.handbrake_val,
        f32::from(car.has_jumped),
        f32::from(car.is_jumping),
        f32::from(car.has_flipped),
        f32::from(f.is_flipping),
        f32::from(car.has_double_jumped),
        f32::from(f.can_flip),
        car.air_time_since_jump.min(1.25),
    ]);
    car_obs(&mut o, car, invert);
    // allies then enemies, in state order (python iterates state.cars dict order)
    for (j, (oi, oc)) in state.cars.iter().enumerate() {
        if j != i && oi.team == info.team { car_obs(&mut o, oc, invert); }
    }
    for (j, (oi, oc)) in state.cars.iter().enumerate() {
        if j != i && oi.team != info.team { car_obs(&mut o, oc, invert); }
    }
    o
}

#[derive(Default)]
pub struct V2DefaultObs;

/// Observation size for 1v1.
pub const OBS_SIZE: usize = 92;
impl<SI> Obs<SI> for V2DefaultObs {
    fn get_obs_space(&self, _si: &SI) -> usize { OBS_SIZE }
    fn reset(&mut self, _s: &GameState, _si: &mut SI) {}
    fn build_obs(&mut self, state: &GameState, _si: &mut SI) -> FullObs {
        let order = pad_order(state);
        (0..state.cars.len()).map(|i| build_one(state, i, &order)).collect()
    }
}
