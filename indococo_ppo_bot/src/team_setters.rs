//! TM1 team drills and the support placer for 2v2 / 3v3.
//!
//! Car order is Blue, Orange, Blue, Orange, ... (`env::create_env`), so car
//! `i` is on team `i % 2`; blue attacks +y. Every car is kept at least
//! [`MIN_CAR_GAP`] uu from the others.

use rand::RngExt;
use rlgymppo_utils::rlgym::StateSetter;
use rlgymppo_utils::rlgym::rocketsim::{Arena, CarState, Mat3A, Vec3A};
use rlgymppo_utils::shared_info::SharedInfoRng;
use serde::Deserialize;

use crate::setters::{ball_at, face_xy, range};

const SIDE_WALL_X: f32 = 4096.0;
const BACK_WALL_Y: f32 = 5120.0;
const CAR_REST_HEIGHT: f32 = 17.0;
/// Minimum distance between any two cars at spawn.
pub const MIN_CAR_GAP: f32 = 300.0;

fn attack_sign(idx: usize) -> f32 {
    if idx % 2 == 0 { 1.0 } else { -1.0 }
}

fn clamp_field(p: Vec3A) -> Vec3A {
    Vec3A::new(
        p.x.clamp(-SIDE_WALL_X + 200.0, SIDE_WALL_X - 200.0),
        p.y.clamp(-BACK_WALL_Y + 200.0, BACK_WALL_Y - 200.0),
        p.z.max(CAR_REST_HEIGHT),
    )
}

/// Draws spots from `spot` until one is `MIN_CAR_GAP` from `placed` and
/// `min_ball` from the ball (30 tries, then the last draw).
fn free_spot(
    rng: &mut impl RngExt,
    placed: &[Vec3A],
    ball: Vec3A,
    min_ball: f32,
    mut spot: impl FnMut(&mut dyn FnMut() -> f32) -> Vec3A,
) -> Vec3A {
    let mut pos = Vec3A::ZERO;
    for _ in 0..30 {
        let mut u = || rng.random_range(0.0f32..1.0);
        pos = clamp_field(spot(&mut u));
        if placed.iter().all(|p| p.distance(pos) >= MIN_CAR_GAP) && pos.distance(ball) >= min_ball {
            break;
        }
    }
    pos
}

fn grounded(pos: Vec3A, face: Vec3A, boost: f32) -> CarState {
    let mut car = CarState::DEFAULT;
    car.phys.pos = Vec3A::new(pos.x, pos.y, CAR_REST_HEIGHT);
    car.phys.rot_mat = face_xy(face);
    car.boost = boost;
    car
}

/// Support spots for cars `from..`: between the ball and their own goal,
/// 1200-3500 uu behind it, facing the ball, 20-100 boost.
pub fn place_support_cars(arena: &mut Arena, rng: &mut impl RngExt, from: usize) {
    let n = arena.num_cars();
    if n <= from {
        return;
    }
    let ball = arena.get_ball_state().phys.pos;
    let mut placed: Vec<Vec3A> = (0..from).map(|i| arena.get_car_state(i).phys.pos).collect();
    for idx in from..n {
        let s = attack_sign(idx);
        let pos = free_spot(rng, &placed, ball, 600.0, |u| {
            let back = 1200.0 + 2300.0 * u();
            Vec3A::new(ball.x * 0.3 + (u() - 0.5) * 5000.0, ball.y - s * back, CAR_REST_HEIGHT)
        });
        let boost = 20.0 + 80.0 * rng.random_range(0.0f32..1.0);
        arena.set_car_state(idx, grounded(pos, ball - pos, boost).core());
        placed.push(pos);
    }
}

/// Open play: ball anywhere (rolling or bouncing), per team one car
/// 600-1500 uu from the ball, one "last back" near its own goal and (3v3)
/// one mid support. Also fine in 1v1 (then only the near cars).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TeamPlay {
    pub ball_speed: [f32; 2],
    pub near_dist: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for TeamPlay {
    fn default() -> Self {
        Self {
            ball_speed: [0.0, 1500.0],
            near_dist: [600.0, 1500.0],
            boost: [20.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for TeamPlay {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let ball = Vec3A::new(
            rng.random_range(-3000.0..3000.0),
            rng.random_range(-3500.0..3500.0),
            rng.random_range(93.0..600.0),
        );
        let a = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        let v = Vec3A::new(a.cos(), a.sin(), 0.2) * range(rng, self.ball_speed);
        arena.set_ball_state(ball_at(ball, v));
        let n = arena.num_cars();
        let mut placed = Vec::new();
        for idx in 0..n {
            let s = attack_sign(idx);
            let role = idx / 2; // 0 near, 1 back, 2 mid
            let near = range(rng, self.near_dist);
            let pos = free_spot(rng, &placed, ball, 400.0, |u| match role {
                0 => {
                    // Own side of the ball, within a 120 deg cone toward own goal.
                    let ang = (u() - 0.5) * 2.1;
                    let back = Vec3A::new(ang.sin(), -s * ang.cos(), 0.0);
                    ball + back * near
                }
                1 => Vec3A::new((u() - 0.5) * 1600.0, -s * (BACK_WALL_Y - 600.0 - 1400.0 * u()), CAR_REST_HEIGHT),
                _ => Vec3A::new((u() - 0.5) * 5000.0, ball.y - s * (1500.0 + 1500.0 * u()), CAR_REST_HEIGHT),
            });
            let boost = range(rng, self.boost);
            arena.set_car_state(idx, grounded(pos, ball - pos, boost).core());
            placed.push(pos);
        }
    }
}

/// Aerial pass: the passer (a random team's first car) airborne under a high
/// wing ball, ball drifting toward the middle; the passer's teammate (if
/// any) on the ground 1500-2600 uu away in front of the opponent goal,
/// facing the ball; opponents back. Without a teammate it is a wing aerial.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AerialPass {
    pub ball_height: [f32; 2],
    pub receiver_dist: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for AerialPass {
    fn default() -> Self {
        Self {
            ball_height: [500.0, 1200.0],
            receiver_dist: [1500.0, 2600.0],
            boost: [40.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for AerialPass {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let n = arena.num_cars();
        let passer = rng.random_range(0..n.min(2));
        let s = attack_sign(passer);
        let side = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let ball = Vec3A::new(
            side * rng.random_range(2000.0..3400.0),
            s * rng.random_range(500.0..3200.0),
            range(rng, self.ball_height),
        );
        // Drift toward the middle in front of the goal.
        let target = Vec3A::new(0.0, s * 3800.0, 300.0);
        let v = (target - ball).normalize() * rng.random_range(300.0..900.0);
        arena.set_ball_state(ball_at(ball, v));
        let mut placed = Vec::new();
        for idx in 0..n {
            let rng = shared_info.rng();
            let boost = range(rng, self.boost);
            let car = if idx == passer {
                let mut c = CarState::DEFAULT;
                c.phys.pos = clamp_field(ball - Vec3A::new(side * 150.0, s * 250.0, 260.0));
                let fwd = (ball - c.phys.pos).normalize();
                let up = fwd.cross(Vec3A::Z).cross(fwd).normalize();
                c.phys.rot_mat = Mat3A::from_cols(fwd, up.cross(fwd), up);
                c.phys.vel = v + fwd * 400.0;
                c.has_jumped = true;
                c.boost = boost;
                c
            } else if idx % 2 == passer % 2 && idx / 2 == 1 {
                let d = range(rng, self.receiver_dist);
                let pos = free_spot(rng, &placed, ball, 800.0, |u| {
                    Vec3A::new(-side * 600.0 * u(), ball.y + s * 0.4 * d, CAR_REST_HEIGHT)
                });
                grounded(Vec3A::new(pos.x, pos.y.clamp(-4300.0, 4300.0), 0.0), ball - pos, boost)
            } else if idx % 2 != passer % 2 && idx / 2 == 0 {
                // First defender: near its goal.
                let pos = free_spot(rng, &placed, ball, 800.0, |u| {
                    Vec3A::new((u() - 0.5) * 1400.0, s * (BACK_WALL_Y - 500.0 - 900.0 * u()), CAR_REST_HEIGHT)
                });
                grounded(pos, ball - pos, boost)
            } else {
                // Everyone else on its own half.
                let own = -attack_sign(idx);
                let pos = free_spot(rng, &placed, ball, 800.0, |u| {
                    Vec3A::new((u() - 0.5) * 6000.0, own * (500.0 + 3500.0 * u()), CAR_REST_HEIGHT)
                });
                grounded(pos, ball - pos, boost)
            };
            placed.push(car.phys.pos);
            arena.set_car_state(idx, car.core());
        }
    }
}

/// Backboard double tap: ball 2600-3800 uu up the opponent half flying at
/// the backboard (above the goal), the attacker airborne behind and below
/// it with boost; its teammates mid; opponents near their goal on the ground.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BackboardDoubleTap {
    pub ball_y: [f32; 2],
    pub ball_height: [f32; 2],
    pub ball_speed: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for BackboardDoubleTap {
    fn default() -> Self {
        Self {
            ball_y: [2600.0, 3800.0],
            ball_height: [700.0, 1300.0],
            ball_speed: [1400.0, 2200.0],
            boost: [50.0, 100.0],
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for BackboardDoubleTap {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let n = arena.num_cars();
        let att = rng.random_range(0..n.min(2));
        let s = attack_sign(att);
        let ball = Vec3A::new(
            rng.random_range(-1200.0..1200.0),
            s * range(rng, self.ball_y),
            range(rng, self.ball_height),
        );
        // Aim above the crossbar (z 900-1500) on the back wall.
        let aim = Vec3A::new(ball.x * 0.3, s * BACK_WALL_Y, rng.random_range(900.0..1500.0));
        let v = (aim - ball).normalize() * range(rng, self.ball_speed);
        arena.set_ball_state(ball_at(ball, v));
        let mut placed = Vec::new();
        for idx in 0..n {
            let rng = shared_info.rng();
            let boost = range(rng, self.boost);
            let car = if idx == att {
                let mut c = CarState::DEFAULT;
                let dir = v.normalize();
                c.phys.pos = clamp_field(ball - dir * rng.random_range(500.0..900.0) - Vec3A::new(0.0, 0.0, 150.0));
                let fwd = (ball - c.phys.pos).normalize();
                let up = fwd.cross(Vec3A::Z).cross(fwd).normalize();
                c.phys.rot_mat = Mat3A::from_cols(fwd, up.cross(fwd), up);
                c.phys.vel = v * 0.8;
                c.has_jumped = true;
                c.boost = boost;
                c
            } else if idx % 2 == att % 2 {
                let pos = free_spot(rng, &placed, ball, 600.0, |u| {
                    Vec3A::new((u() - 0.5) * 4000.0, s * (300.0 + 1500.0 * u()), CAR_REST_HEIGHT)
                });
                grounded(pos, ball - pos, boost)
            } else {
                let pos = free_spot(rng, &placed, ball, 600.0, |u| {
                    Vec3A::new((u() - 0.5) * 1800.0, s * (BACK_WALL_Y - 400.0 - 1600.0 * u()), CAR_REST_HEIGHT)
                });
                grounded(pos, ball - pos, boost)
            };
            placed.push(car.phys.pos);
            arena.set_car_state(idx, car.core());
        }
    }
}

/// TM8 breakaway: a random attacking team's car just behind a grounded or
/// bouncing ball 1500-4500 uu from the attacked goal line, both moving
/// goalward; every defender 1200-3000 uu behind the ball and out of the
/// 1100 uu lane (the defender is beaten); attacker teammates at support spots.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Breakaway {
    pub goal_dist: [f32; 2],
    pub speed: [f32; 2],
    pub defender_behind: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for Breakaway {
    fn default() -> Self {
        Self { goal_dist: [1500.0, 4500.0], speed: [1000.0, 1800.0], defender_behind: [1200.0, 3000.0], boost: [20.0, 80.0] }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for Breakaway {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let n = arena.num_cars();
        let att = usize::from(rng.random_range(0.0f32..1.0) < 0.5);
        let s = attack_sign(att);
        let gd = range(rng, self.goal_dist);
        let ball = Vec3A::new(rng.random_range(-1500.0..1500.0), s * (BACK_WALL_Y - gd), rng.random_range(93.0..300.0));
        let speed = range(rng, self.speed);
        // Aim roughly at the net with some spread.
        let aim = Vec3A::new(rng.random_range(-600.0..600.0), s * BACK_WALL_Y, 0.0) - Vec3A::new(ball.x, ball.y, 0.0);
        let dir = aim.normalize_or_zero();
        arena.set_ball_state(ball_at(ball, dir * speed + Vec3A::new(0.0, 0.0, rng.random_range(0.0..250.0))));
        let mut placed = Vec::new();
        for idx in 0..n {
            let mine = idx % 2 == att;
            let (pos, vel, face) = if mine && idx == att {
                let p = ball - dir * rng.random_range(250.0..450.0);
                (Vec3A::new(p.x, p.y, CAR_REST_HEIGHT), dir * (speed * rng.random_range(0.9..1.05)), dir)
            } else if mine {
                let p = free_spot(rng, &placed, ball, 900.0, |u| Vec3A::new((u() - 0.5) * 5000.0, ball.y - s * (1500.0 + 2500.0 * u()), CAR_REST_HEIGHT));
                (p, dir * 800.0, dir)
            } else {
                let behind = range(rng, self.defender_behind);
                let p = free_spot(rng, &placed, ball, 1000.0, |u| {
                    let side = if u() < 0.5 { -1.0 } else { 1.0 };
                    Vec3A::new(ball.x + side * (1200.0 + 1500.0 * u()), ball.y - s * behind, CAR_REST_HEIGHT)
                });
                (p, dir * rng.random_range(800.0..1600.0), dir)
            };
            let mut car = grounded(pos, face, range(rng, self.boost));
            car.phys.vel = vel;
            arena.set_car_state(idx, car.core());
            placed.push(pos);
        }
    }
}

/// TM8 goalie practice (Necto GoaliePracticeState style): a shot already
/// travelling at a random team's net from 2000-4500 uu out; that team's last
/// car in or near its goal mouth, the others upfield recovering; the
/// shooting team's cars following the ball.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct GoalieShot {
    pub shot_dist: [f32; 2],
    pub shot_speed: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for GoalieShot {
    fn default() -> Self {
        Self { shot_dist: [2000.0, 4500.0], shot_speed: [1400.0, 2600.0], boost: [20.0, 100.0] }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for GoalieShot {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(None);
        let rng = shared_info.rng();
        let n = arena.num_cars();
        let def = usize::from(rng.random_range(0.0f32..1.0) < 0.5);
        let s = attack_sign(def); // defender attacks +s, defends -s.
        let d = range(rng, self.shot_dist);
        let ball = Vec3A::new(rng.random_range(-2500.0..2500.0), -s * (BACK_WALL_Y - d), rng.random_range(100.0..900.0));
        let target = Vec3A::new(rng.random_range(-850.0..850.0), -s * BACK_WALL_Y, rng.random_range(100.0..550.0));
        let dir = (target - ball).normalize_or_zero();
        let speed = range(rng, self.shot_speed);
        arena.set_ball_state(ball_at(ball, dir * speed + Vec3A::new(0.0, 0.0, 150.0)));
        let mut placed = Vec::new();
        let mut first_def = true;
        for idx in 0..n {
            let (pos, face) = if idx % 2 == def {
                if first_def {
                    first_def = false;
                    let p = Vec3A::new(rng.random_range(-1000.0..1000.0), -s * (BACK_WALL_Y - rng.random_range(150.0..900.0)), CAR_REST_HEIGHT);
                    (p, ball - p)
                } else {
                    let p = free_spot(rng, &placed, ball, 800.0, |u| Vec3A::new((u() - 0.5) * 6000.0, ball.y + s * (500.0 + 3000.0 * u()), CAR_REST_HEIGHT));
                    (p, Vec3A::new(0.0, -s, 0.0))
                }
            } else {
                let p = free_spot(rng, &placed, ball, 500.0, |u| {
                    let back = 700.0 + 2500.0 * u();
                    Vec3A::new(ball.x + (u() - 0.5) * 2000.0, ball.y + s * back, CAR_REST_HEIGHT)
                });
                (p, ball - p)
            };
            let mut car = grounded(pos, face, range(rng, self.boost));
            car.phys.vel = face.normalize_or_zero() * rng.random_range(0.0..1400.0);
            arena.set_car_state(idx, car.core());
            placed.push(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rlgymppo_utils::rlgym::rocketsim::{CarBodyConfig, GameMode, Team, init_from_default};

    struct Si(SmallRng);
    impl SharedInfoRng for Si {
        type Rng = SmallRng;
        fn rng(&mut self) -> &mut SmallRng {
            &mut self.0
        }
    }

    fn arena(per_team: usize) -> Arena {
        init_from_default(true).unwrap();
        let mut a = Arena::new(GameMode::Soccar);
        for _ in 0..per_team {
            a.add_car(Team::Blue, CarBodyConfig::OCTANE);
            a.add_car(Team::Orange, CarBodyConfig::OCTANE);
        }
        a
    }

    fn check(a: &Arena) {
        let n = a.num_cars();
        for i in 0..n {
            let p = a.get_car_state(i).phys.pos;
            assert!(p.x.abs() <= SIDE_WALL_X && p.y.abs() <= BACK_WALL_Y && p.z >= 0.0 && p.z <= 2044.0, "{p:?}");
        }
    }

    #[test]
    fn team_drills_place_every_car_in_bounds_and_apart() {
        let mut si = Si(SmallRng::seed_from_u64(3));
        for per_team in 1..=3 {
            let mut a = arena(per_team);
            let mut close = 0usize;
            let mut total = 0usize;
            for _ in 0..300 {
                for k in 0..4 {
                    match k {
                        0 => TeamPlay::default().apply(&mut a, &mut si),
                        1 => AerialPass::default().apply(&mut a, &mut si),
                        2 => BackboardDoubleTap::default().apply(&mut a, &mut si),
                        _ => {
                            a.reset_to_random_kickoff(None);
                            place_support_cars(&mut a, &mut si.0, 2);
                        }
                    }
                    check(&a);
                    let n = a.num_cars();
                    for i in 0..n {
                        for j in 0..i {
                            total += 1;
                            let d = a.get_car_state(i).phys.pos.distance(a.get_car_state(j).phys.pos);
                            if d < MIN_CAR_GAP - 1.0 {
                                close += 1;
                            }
                        }
                    }
                }
            }
            assert!(close * 100 <= total.max(1), "{close} of {total} pairs closer than the gap ({per_team} per team)");
        }
    }

    #[test]
    fn aerial_pass_receiver_is_a_teammate_in_front_of_goal() {
        let mut si = Si(SmallRng::seed_from_u64(5));
        let mut a = arena(2);
        for _ in 0..200 {
            AerialPass::default().apply(&mut a, &mut si);
            let ball = a.get_ball_state().phys.pos;
            // The airborne car is the passer (car 0 or 1); its teammate is car passer+2.
            let passer = (0..2).find(|&i| a.get_car_state(i).phys.pos.z > 100.0).expect("airborne passer");
            let s = attack_sign(passer);
            let recv = a.get_car_state(passer + 2).phys.pos;
            assert!(s * recv.y > s * ball.y, "receiver ahead of the ball: {recv:?} ball {ball:?}");
        }
    }
}
