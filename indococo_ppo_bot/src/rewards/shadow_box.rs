//! `ShadowBox`: potential-based shadow / goal-side term (ST36).
//!
//! While the attack gate is on (ball in our half, ball velocity toward our
//! net above `threat_speed_min`, opponent closer to the ball), the potential
//! is
//!
//! `phi = lateral_box * longitudinal_band * shadow_line * speed` in `[0, 1]`.
//!
//! Each step pays `r = gamma * phi(s') - phi(s)` (Ng, Harada & Russell 1999).
//! When the gate is off the reward is 0 and previous phi is frozen, so a
//! clear cannot take a 1→0 penalty. A gate-on transition starts from
//! `phi_prev = phi(s)` and pays 0 on that first active step (no free jump).
//!
//! The longitudinal band is deliberately low in the 0–700 uu "parked"
//! zone off our goal line, where Galactus keeps demoing us.

use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::{BACK_WALL_Y, GOAL_CENTER_TO_POST};
use crate::math::unit;
use crate::state::{CarView, StateView};
use glam::DVec3;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShadowBoxParams {
    /// Discount used in `r = gamma * phi(s') - phi(s)`. Should match PPO.
    pub gamma: f64,
    /// Ball speed toward our goal center required for the gate, uu/s.
    pub threat_speed_min: f64,
    /// Shadow-line factor falls from 1 at 0 uu off the ball–goal line to 0
    /// at this distance.
    pub line_width: f64,
    /// Distance off our goal line (uu) where the longitudinal rise is still
    /// only [`band_low_value`].
    pub band_low_end: f64,
    /// Distance off our goal line where the longitudinal band reaches 1.
    pub band_peak_start: f64,
    /// Distance off our goal line where the longitudinal band starts to fall.
    pub band_peak_end: f64,
    /// Distance off our goal line where the longitudinal band reaches 0.
    pub band_far: f64,
    /// Longitudinal factor at [`band_low_end`] (smoothly low in 0–700 uu).
    pub band_low_value: f64,
    /// `|x|` inside this width (goal mouth half-width) keeps `lateral_box` = 1.
    pub goal_half_width: f64,
    /// `|x|` fade past the goal mouth; `lateral_box` is 0 at
    /// `goal_half_width + lateral_fade`.
    pub lateral_fade: f64,
    /// Car speed (uu/s) at which the speed factor reaches 1.
    pub speed_ref: f64,
    /// Speed factor for a stationary car (`clip(|v|/speed_ref, floor, 1)`).
    pub speed_floor: f64,
    /// Own goal line `|y|` (the line used by `diag_gal`, 5120).
    pub goal_line_y: f64,
}

impl Default for ShadowBoxParams {
    fn default() -> Self {
        Self {
            gamma: 0.998,
            threat_speed_min: 300.0,
            line_width: 1200.0,
            band_low_end: 700.0,
            band_peak_start: 800.0,
            band_peak_end: 2400.0,
            band_far: 3500.0,
            band_low_value: 0.15,
            goal_half_width: GOAL_CENTER_TO_POST,
            lateral_fade: 800.0,
            speed_ref: 800.0,
            speed_floor: 0.3,
            goal_line_y: BACK_WALL_Y,
        }
    }
}

pub struct ShadowBox {
    p: ShadowBoxParams,
    /// Previous potential, valid only while `active`.
    prev: Vec<f64>,
    /// Whether the previous step's gate was on for this car.
    active: Vec<bool>,
}

impl ShadowBox {
    #[must_use]
    pub fn new(p: ShadowBoxParams) -> Self {
        Self {
            p,
            prev: Vec::new(),
            active: Vec::new(),
        }
    }

    fn own_goal(&self, car: &CarView) -> DVec3 {
        DVec3::new(0.0, -car.attack_sign() * self.p.goal_line_y, 0.0)
    }

    /// Attack gate for `car`. False when the car is demoed or there is no
    /// living opponent closer to the ball.
    #[must_use]
    pub fn gate(&self, car: &CarView, state: &StateView) -> bool {
        if car.is_demoed() {
            return false;
        }
        let ball = state.ball.pos;
        if car.attack_sign() * ball.y >= 0.0 {
            return false;
        }
        let own = self.own_goal(car);
        let toward = state.ball.vel.dot(unit(own - ball));
        if toward <= self.p.threat_speed_min {
            return false;
        }
        let d_me = (car.pos - ball).length();
        state
            .cars
            .iter()
            .any(|o| o.team != car.team && !o.is_demoed() && (o.pos - ball).length() < d_me)
    }

    /// Unweighted potential in `[0, 1]`. 0 when the gate is off.
    #[must_use]
    pub fn potential(&self, car: &CarView, state: &StateView) -> f64 {
        if !self.gate(car, state) {
            return 0.0;
        }
        let p = &self.p;
        let own = self.own_goal(car);
        let d_line = (car.pos.y - own.y).abs();
        let long = longitudinal_band(d_line, p);
        let lat = lateral_box(car.pos.x, p);
        let line = shadow_line_factor(xy(car.pos), xy(own), xy(state.ball.pos), p.line_width);
        let speed = (car.vel.length() / p.speed_ref.max(1e-6)).clamp(p.speed_floor, 1.0);
        (lat * long * line * speed).clamp(0.0, 1.0)
    }
}

fn xy(v: DVec3) -> DVec3 {
    DVec3::new(v.x, v.y, 0.0)
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn longitudinal_band(d: f64, p: &ShadowBoxParams) -> f64 {
    let rise = if d <= p.band_low_end {
        p.band_low_value * smoothstep(0.0, p.band_low_end, d)
    } else {
        p.band_low_value
            + (1.0 - p.band_low_value) * smoothstep(p.band_low_end, p.band_peak_start, d)
    };
    let fall = 1.0 - smoothstep(p.band_peak_end, p.band_far, d);
    (rise * fall).clamp(0.0, 1.0)
}

fn lateral_box(x: f64, p: &ShadowBoxParams) -> f64 {
    let extra = x.abs() - p.goal_half_width;
    if extra <= 0.0 {
        1.0
    } else {
        (1.0 - extra / p.lateral_fade.max(1e-6)).max(0.0)
    }
}

/// 1 at 0 uu off the segment from own-goal center to the ball, 0 by `width`.
fn shadow_line_factor(car: DVec3, own: DVec3, ball: DVec3, width: f64) -> f64 {
    let seg = ball - own;
    let len = seg.length();
    let closest = if len < 1e-6 {
        own
    } else {
        let t = (car - own).dot(seg / len).clamp(0.0, len);
        own + (seg / len) * t
    };
    (1.0 - (car - closest).length() / width.max(1e-6)).max(0.0)
}

impl StepReward for ShadowBox {
    fn reset(&mut self, initial: &StateView) {
        let n = initial.cars.len();
        self.prev = vec![0.0; n];
        self.active = vec![false; n];
        for (i, car) in initial.cars.iter().enumerate() {
            if self.gate(car, initial) {
                self.prev[i] = self.potential(car, initial);
                self.active[i] = true;
            }
        }
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        if self.prev.len() != n {
            self.prev = vec![0.0; n];
            self.active = vec![false; n];
        }
        let mut rewards = vec![0.0; n];
        for (i, car) in state.cars.iter().enumerate() {
            if !self.gate(car, state) {
                // Freeze prev; a clear must not take a 1→0 penalty.
                self.active[i] = false;
                rewards[i] = 0.0;
                continue;
            }
            let phi = self.potential(car, state);
            if !self.active[i] {
                // Gate just turned on: start from phi(s), pay nothing.
                self.prev[i] = phi;
                self.active[i] = true;
                rewards[i] = 0.0;
            } else {
                let r = self.p.gamma * phi - self.prev[i];
                self.prev[i] = phi;
                rewards[i] = if r.is_finite() { r } else { 0.0 };
            }
        }
        rewards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::BallView;

    fn car(team: u8, pos: DVec3, vel: DVec3) -> CarView {
        CarView {
            team,
            pos,
            vel,
            ..CarView::default()
        }
    }

    fn state(ball_pos: DVec3, ball_vel: DVec3, blue: CarView, orange: CarView) -> StateView {
        StateView {
            ball: BallView {
                pos: ball_pos,
                vel: ball_vel,
                ..BallView::default()
            },
            cars: vec![blue, orange],
            ..StateView::default()
        }
    }

    /// Ball in blue's half, coming at blue's net at 400 uu/s.
    fn attack_ball() -> (DVec3, DVec3) {
        (DVec3::new(0.0, -2000.0, 93.0), DVec3::new(0.0, -400.0, 0.0))
    }

    /// Orange (attacker) closer to the ball than a blue car on the goal-ball line.
    fn orange_on_ball() -> CarView {
        let (b, _) = attack_ball();
        car(1, DVec3::new(0.0, b.y - 80.0, 17.0), DVec3::new(0.0, -200.0, 0.0))
    }

    /// Blue in the peak longitudinal band, on the shadow line, moving.
    fn blue_peak(speed: f64) -> CarView {
        // 1600 uu off the goal line: y = -5120 + 1600 = -3520.
        car(
            0,
            DVec3::new(0.0, -3520.0, 17.0),
            DVec3::new(0.0, -speed, 0.0),
        )
    }

    fn attack_at(blue: CarView) -> StateView {
        let (bp, bv) = attack_ball();
        state(bp, bv, blue, orange_on_ball())
    }

    #[test]
    fn peak_on_line_moving_is_near_one() {
        let r = ShadowBox::new(ShadowBoxParams::default());
        let s = attack_at(blue_peak(800.0));
        let phi = r.potential(&s.cars[0], &s);
        assert!((phi - 1.0).abs() < 1e-9, "{phi}");
        assert!(r.gate(&s.cars[0], &s));
        // The attacker (orange) is in the opposite half relative to itself,
        // so its gate is off.
        assert!(!r.gate(&s.cars[1], &s));
        assert_eq!(r.potential(&s.cars[1], &s), 0.0);
    }

    #[test]
    fn parked_band_is_low() {
        let r = ShadowBox::new(ShadowBoxParams::default());
        // ST34 demo median: 401 uu off the goal line.
        let parked = attack_at(car(
            0,
            DVec3::new(0.0, -5120.0 + 401.0, 17.0),
            DVec3::new(0.0, -800.0, 0.0),
        ));
        let peak = attack_at(blue_peak(800.0));
        let phi_p = r.potential(&parked.cars[0], &parked);
        let phi_k = r.potential(&peak.cars[0], &peak);
        assert!(phi_p < 0.2, "parked phi {phi_p}");
        assert!(phi_k > 0.9, "peak phi {phi_k}");
        assert!(phi_p < 0.25 * phi_k, "{phi_p} vs {phi_k}");
    }

    #[test]
    fn standing_still_is_less_than_moving() {
        let r = ShadowBox::new(ShadowBoxParams::default());
        let still = attack_at(blue_peak(0.0));
        let moving = attack_at(blue_peak(800.0));
        let a = r.potential(&still.cars[0], &still);
        let b = r.potential(&moving.cars[0], &moving);
        assert!((a - 0.3).abs() < 1e-9, "{a}");
        assert!((b - 1.0).abs() < 1e-9, "{b}");
        assert!(a < b);
    }

    #[test]
    fn telescopes_when_gamma_is_one_and_gate_holds() {
        let mut r = ShadowBox::new(ShadowBoxParams {
            gamma: 1.0,
            ..ShadowBoxParams::default()
        });
        let start = attack_at(blue_peak(800.0));
        r.reset(&start);
        // First step at the reset state pays 0.
        assert_eq!(r.get_rewards(&start)[0], 0.0);
        // Drop into the parked band, then return to the peak, so the
        // episode sum is a real telescope, not 0 + 0.
        let mid = attack_at(car(
            0,
            DVec3::new(0.0, -5120.0 + 401.0, 17.0),
            DVec3::new(0.0, -800.0, 0.0),
        ));
        let end = attack_at(blue_peak(800.0));
        let r1 = r.get_rewards(&mid)[0];
        let r2 = r.get_rewards(&end)[0];
        let phi0 = ShadowBox::new(ShadowBoxParams {
            gamma: 1.0,
            ..ShadowBoxParams::default()
        })
        .potential(&start.cars[0], &start);
        let phi2 = ShadowBox::new(ShadowBoxParams {
            gamma: 1.0,
            ..ShadowBoxParams::default()
        })
        .potential(&end.cars[0], &end);
        assert!(((r1 + r2) - (phi2 - phi0)).abs() < 1e-9, "{r1} {r2} {phi0} {phi2}");
        // Hovering pays 0 at gamma = 1.
        assert_eq!(r.get_rewards(&end)[0], 0.0);
    }

    #[test]
    fn pbrs_uses_configured_gamma() {
        let mut r = ShadowBox::new(ShadowBoxParams::default());
        assert!((r.p.gamma - 0.998).abs() < 1e-12);
        let a = attack_at(blue_peak(800.0));
        let b = attack_at(blue_peak(0.0));
        r.reset(&a);
        // Production calls get_rewards on the next state, not the reset
        // state. r = gamma * phi(s1) - phi(s0).
        let got = r.get_rewards(&b)[0];
        let sb = ShadowBox::new(ShadowBoxParams::default());
        let want = 0.998 * sb.potential(&b.cars[0], &b) - sb.potential(&a.cars[0], &a);
        assert!((got - want).abs() < 1e-12, "{got} {want}");
        // A self-transition pays (gamma-1)*phi, not zero, when gamma != 1.
        let mut r2 = ShadowBox::new(ShadowBoxParams::default());
        r2.reset(&a);
        let hover = r2.get_rewards(&a)[0];
        let phi = sb.potential(&a.cars[0], &a);
        assert!((hover - (0.998 * phi - phi)).abs() < 1e-12, "{hover} {phi}");
    }

    #[test]
    fn gate_off_freezes_and_a_clear_has_no_penalty() {
        let mut r = ShadowBox::new(ShadowBoxParams::default());
        let on = attack_at(blue_peak(800.0));
        r.reset(&on);
        // Clear: ball is in their half, going away. Would be a 1→0 drop if
        // the potential were applied without the freeze.
        let (bp, _) = attack_ball();
        let cleared = state(
            DVec3::new(0.0, 1500.0, 93.0),
            DVec3::new(0.0, 800.0, 0.0),
            blue_peak(800.0),
            car(1, bp, DVec3::ZERO),
        );
        assert!(!r.gate(&cleared.cars[0], &cleared));
        assert_eq!(r.get_rewards(&cleared)[0], 0.0);
        assert_eq!(r.get_rewards(&cleared)[0], 0.0);
        // Gate-on transition starts from the current phi: first step pays 0,
        // no free jump back to the previous high potential.
        assert_eq!(r.get_rewards(&on)[0], 0.0);
        // And then a real change pays the PBRS difference, not a jump from 0.
        let still = attack_at(blue_peak(0.0));
        let got = r.get_rewards(&still)[0];
        let sb = ShadowBox::new(ShadowBoxParams::default());
        let want = 0.998 * sb.potential(&still.cars[0], &still) - sb.potential(&on.cars[0], &on);
        assert!((got - want).abs() < 1e-12, "{got} {want}");
        assert!(got < 0.0, "slowing down from the peak should cost, got {got}");
    }

    #[test]
    fn midfield_and_own_possession_are_off() {
        let r = ShadowBox::new(ShadowBoxParams::default());
        let (bp, bv) = attack_ball();
        // Ball at midfield.
        let mid = state(
            DVec3::new(0.0, 0.0, 93.0),
            bv,
            blue_peak(800.0),
            orange_on_ball(),
        );
        assert!(!r.gate(&mid.cars[0], &mid));
        // We are closer to the ball than the opponent.
        let us_closer = state(
            bp,
            bv,
            car(0, DVec3::new(0.0, bp.y + 50.0, 17.0), DVec3::new(0.0, -800.0, 0.0)),
            car(1, DVec3::new(0.0, 0.0, 17.0), DVec3::ZERO),
        );
        assert!(!r.gate(&us_closer.cars[0], &us_closer));
        // Ball leaving our half (velocity away from our net).
        let leaving = state(
            bp,
            DVec3::new(0.0, 400.0, 0.0),
            blue_peak(800.0),
            orange_on_ball(),
        );
        assert!(!r.gate(&leaving.cars[0], &leaving));
    }

    #[test]
    fn orange_attack_gate_is_symmetric() {
        let r = ShadowBox::new(ShadowBoxParams::default());
        // Mirror: ball in orange's half (y > 0), coming at +y, blue closer to it.
        let s = state(
            DVec3::new(0.0, 2000.0, 93.0),
            DVec3::new(0.0, 400.0, 0.0),
            car(0, DVec3::new(0.0, 2080.0, 17.0), DVec3::new(0.0, 200.0, 0.0)),
            car(1, DVec3::new(0.0, 3520.0, 17.0), DVec3::new(0.0, 800.0, 0.0)),
        );
        assert!(r.gate(&s.cars[1], &s));
        assert!(!r.gate(&s.cars[0], &s));
        let phi = r.potential(&s.cars[1], &s);
        assert!((phi - 1.0).abs() < 1e-9, "{phi}");
    }

    #[test]
    fn rewards_are_finite_and_not_zero_sum_wrapped() {
        let mut r = ShadowBox::new(ShadowBoxParams::default());
        let s = attack_at(blue_peak(800.0));
        r.reset(&s);
        let out = r.get_rewards(&attack_at(blue_peak(0.0)));
        assert!(out.iter().all(|x| x.is_finite()));
        // Only blue is gated; orange's reward is 0, so the raw term is not
        // antisymmetric (a ZeroSum wrapper would subtract the opponent mean).
        assert_eq!(out[1], 0.0);
        assert_ne!(out[0], 0.0);
    }
}
