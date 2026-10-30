//! Shared geometry helpers from `rewards/freestyleMechs.py` (top of file).
//! They are used by several mechanic rewards and must match the Python exactly.

use glam::DVec3;

use crate::consts::BACK_NET_Y;
use crate::state::{CarView, StateView};

const WALL_X: f64 = 3500.0;
const WALL_Z: f64 = 250.0;

/// y of the net this car attacks.
#[must_use]
pub fn goal_y_for_car(car: &CarView) -> f64 {
    car.attack_sign() * BACK_NET_Y
}

/// `|goal_y - pos.y|` for the net this car attacks.
#[must_use]
pub fn dist_to_opp_goal_y(car: &CarView, pos: DVec3) -> f64 {
    (goal_y_for_car(car) - pos.y).abs()
}

/// Velocity component toward the attacked net along y, floored at 0.
#[must_use]
pub fn goalward_y_speed(car: &CarView, vel: DVec3) -> f64 {
    (vel.y * car.attack_sign()).max(0.0)
}

/// Near a side wall or up off the floor.
#[must_use]
pub fn on_wallish(pos: DVec3) -> bool {
    pos.x.abs() >= WALL_X || pos.z >= WALL_Z
}

/// Distance to the closest non-demoed opponent car, `1e9` if none.
#[must_use]
pub fn min_opp_car_dist(agent: usize, state: &StateView) -> f64 {
    let me = &state.cars[agent];
    state
        .cars
        .iter()
        .enumerate()
        .filter(|(i, o)| *i != agent && o.team != me.team && !o.is_demoed())
        .map(|(_, o)| (o.pos - me.pos).length())
        .reduce(f64::min)
        .unwrap_or(1e9)
}

/// Goalward speed required before an air-dribble start, rising linearly with
/// the remaining distance to the net.
#[must_use]
pub fn takeoff_need_speed(dist_to_goal: f64, near: f64, far: f64, dist_ref: f64) -> f64 {
    let t = (dist_to_goal / dist_ref.max(1.0)).clamp(0.0, 1.0);
    near + t * (far - near)
}

/// Launch momentum: goalward car or ball speed; wall rides count 80% of
/// total car speed.
#[must_use]
pub fn takeoff_have_speed(car: &CarView, ball_vel: DVec3) -> f64 {
    let mut have = goalward_y_speed(car, car.vel).max(goalward_y_speed(car, ball_vel));
    if on_wallish(car.pos) {
        have = have.max(0.80 * car.vel.length());
    }
    have
}

/// The car is goal-side of the ball by at least `margin`.
#[must_use]
pub fn car_in_front_of_ball(car: &CarView, ball_pos: DVec3, margin: f64) -> bool {
    car.attack_sign() * (car.pos.y - ball_pos.y) >= margin
}

/// How much the nose points at the attacked net, in `[0, 1]`.
#[must_use]
pub fn nose_goalward(car: &CarView) -> f64 {
    let n = car.forward.length();
    if n < 1e-6 {
        return 0.0;
    }
    (car.attack_sign() * car.forward.y / n).max(0.0)
}

/// `> 0` when the bumper faces `target`, `< 0` when the wheels do.
#[must_use]
pub fn nose_into_not_wheels(car: &CarView, target: DVec3) -> f64 {
    let to_t = target - car.pos;
    let n = to_t.length();
    if n < 1e-6 {
        return 0.0;
    }
    let d = to_t / n;
    let (fn_, un) = (car.forward.length(), car.up.length());
    if fn_ < 1e-6 || un < 1e-6 {
        return 0.0;
    }
    let nose_at = (car.forward / fn_).dot(d);
    let wheels_at = (-(car.up / un)).dot(d);
    nose_at - wheels_at.max(0.0)
}

/// Car part facing a target, by dominant local axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactPart {
    Nose,
    Back,
    Roof,
    Wheels,
    Side,
}

/// `bump_contact_part`: which part of `car` faces `target`.
#[must_use]
pub fn bump_contact_part(car: &CarView, target: DVec3) -> ContactPart {
    let d = target - car.pos;
    let n = d.length();
    if n < 1e-6 {
        return ContactPart::Side;
    }
    let d = d / n;
    let f = car.forward.dot(d);
    let u = car.up.dot(d);
    let r = car.right.dot(d);
    // Python: max((|f|,"f"), (|u|,"u"), (|r|,"r")); ties go to the larger
    // label, i.e. "u" > "r" > "f".
    let (af, au, ar) = (f.abs(), u.abs(), r.abs());
    if au >= af && au >= ar {
        if u > 0.0 {
            ContactPart::Roof
        } else {
            ContactPart::Wheels
        }
    } else if ar >= af {
        ContactPart::Side
    } else if f > 0.0 {
        ContactPart::Nose
    } else {
        ContactPart::Back
    }
}

/// Boosting now, or spent at least `0.4` boost since the previous step.
#[must_use]
pub fn spent_boost(car: &CarView, prev: Option<&CarView>) -> bool {
    if car.is_boosting() {
        return true;
    }
    prev.is_some_and(|p| p.boost - car.boost >= 0.4)
}

/// Multiplier fading a slow takeoff: `floor` at zero speed, 1 at `need`.
#[must_use]
pub fn takeoff_speed_mult(have: f64, need: f64, floor: f64) -> f64 {
    if need <= 1.0 {
        return 1.0;
    }
    let ratio = (have / need).clamp(0.0, 1.0);
    floor + (1.0 - floor) * ratio
}

/// Have we beaten every defender with an open lane to their net?
/// Returns `(clear, lead)`.
#[must_use]
pub fn advantage_clear_lane(
    agent: usize,
    state: &StateView,
    ball_pos: DVec3,
    lane_radius: f64,
    min_lead: f64,
) -> (bool, f64) {
    let car = &state.cars[agent];
    let goal_y = goal_y_for_car(car);
    let my_d = (goal_y - car.pos.y).abs();
    let ball_d = (goal_y - ball_pos.y).abs();
    let mut best: Option<f64> = None;
    let mut blocker = false;
    for (i, opp) in state.cars.iter().enumerate() {
        if i == agent || opp.team == car.team || opp.is_demoed() {
            continue;
        }
        let opp_d = (goal_y - opp.pos.y).abs();
        best = Some(best.map_or(opp_d, |b| b.max(opp_d)));
        if opp_d < ball_d && (opp.pos.x - ball_pos.x).abs() < lane_radius {
            blocker = true;
        }
    }
    match best {
        Some(best) if !blocker => {
            let lead = best - my_d;
            (lead >= min_lead, lead.max(0.0))
        }
        _ => (false, 0.0),
    }
}

/// Default lane radius / lead of `advantage_clear_lane`.
pub const LANE_RADIUS: f64 = 1100.0;
pub const MIN_LEAD: f64 = 900.0;

/// Fade an air-dribble start when the defender is already beaten.
#[must_use]
pub fn advantage_ad_mult(is_clear: bool, lead: f64, floor: f64) -> f64 {
    const FULL_LEAD: f64 = 3000.0;
    if !is_clear {
        return 1.0;
    }
    let frac = (lead / FULL_LEAD).min(1.0);
    floor.max(1.0 - frac * (1.0 - floor))
}

/// Fade an air dribble while an opponent is in flick range and the aerial is
/// not yet committed.
#[must_use]
pub fn pressure_ad_mult(
    opp_d: f64,
    ball_z: f64,
    on_ground: bool,
    have_speed: f64,
    pressure_dist: f64,
    floor: f64,
) -> f64 {
    const COMMIT_Z: f64 = 380.0;
    const COMMIT_SPEED: f64 = 700.0;
    if pressure_dist <= 1.0 || opp_d >= pressure_dist {
        return 1.0;
    }
    if !on_ground && ball_z >= COMMIT_Z && have_speed >= COMMIT_SPEED {
        return 1.0;
    }
    let fade = 1.0 - opp_d / pressure_dist;
    floor.max(1.0 - fade * (1.0 - floor))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn car(team: u8, pos: DVec3) -> CarView {
        CarView {
            team,
            pos,
            forward: DVec3::X,
            right: DVec3::Y,
            up: DVec3::Z,
            ..Default::default()
        }
    }

    #[test]
    fn takeoff_need_interpolates_and_clamps() {
        assert_eq!(takeoff_need_speed(0.0, 500.0, 1700.0, 9000.0), 500.0);
        assert_eq!(takeoff_need_speed(4500.0, 500.0, 1700.0, 9000.0), 1100.0);
        assert_eq!(takeoff_need_speed(1e5, 500.0, 1700.0, 9000.0), 1700.0);
    }

    #[test]
    fn contact_part_uses_dominant_axis() {
        let c = car(0, DVec3::ZERO);
        assert_eq!(
            bump_contact_part(&c, DVec3::new(100.0, 0.0, 0.0)),
            ContactPart::Nose
        );
        assert_eq!(
            bump_contact_part(&c, DVec3::new(0.0, 0.0, -100.0)),
            ContactPart::Wheels
        );
        assert_eq!(
            bump_contact_part(&c, DVec3::new(0.0, -100.0, 0.0)),
            ContactPart::Side
        );
    }

    #[test]
    fn clear_lane_needs_lead_and_open_lane() {
        let mut s = StateView::default();
        s.cars.push(car(0, DVec3::new(0.0, 2000.0, 17.0)));
        s.cars.push(car(1, DVec3::new(0.0, -2000.0, 17.0)));
        let (clear, lead) =
            advantage_clear_lane(0, &s, DVec3::new(0.0, 2100.0, 93.0), LANE_RADIUS, MIN_LEAD);
        assert!(clear);
        assert_eq!(lead, 4000.0);
        s.cars[1].pos.y = 4000.0;
        assert!(
            !advantage_clear_lane(0, &s, DVec3::new(0.0, 2100.0, 93.0), LANE_RADIUS, MIN_LEAD).0
        );
    }
}
