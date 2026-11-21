//! TM10 `playmaking_demo` / `playmaking_bump`: pay the SUPPORT car (2v2
//! `Role::Last`, 3v3 `Role::Second`; the 3v3 last man is scaled by
//! `last_scale_3v3`, 0 in TM10) for demoing or bumping the opponent who is
//! in the play while its team attacks, so the first man's touch has a
//! follow-up. The existing `demo` term is flat (pays every car the same and
//! is team-spirit blended); `role_scale` can only pay support less than the
//! first man, hence this separate, individual (post team-spirit) term.
//!
//! A contact qualifies when ALL of:
//! * team mode (2+ cars on the agent's team), the agent is a support car and
//!   a living teammate is the first man; no kickoff pending;
//! * attacking: the ball is in the offensive half, OR our first man is the
//!   last toucher and the ball is not in our defensive third;
//! * neither the ball nor the agent is in our defensive third;
//! * the victim is an opponent (never a teammate) and is in the play (on the
//!   step before the contact): the closest living opponent to the ball, OR
//!   its time-to-ball is within `eta_margin_s` of our first man's, OR it is
//!   their last defender (deepest living opponent) while our first man has
//!   possession (is the last toucher);
//! * the agent is within `max_ball_dist` of the ball, except a last defender
//!   that is shading the shot (goal-side of the ball, within `shade_width`
//!   of the ball -> net line);
//! * the car has had fewer than `cap` paid playmaking events (demos + bumps)
//!   this episode.
//! A demo pays 1.0. A bump (a NEW bump victim that is not demoed) pays
//! `min(1, dv / hard_target) ^ hard_power` for bumper (nose) contacts with
//! victim dv >= `min_bump_dv` (and > the bumper's dv with `victim_gate`);
//! wheel / side / roof / back contacts pay 0. Each event pays on one step.
//!
//! The demo and the bump terms are separate instances with the same
//! detection, so both count every paid event toward the shared cap.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use super::mech::{ContactPart, bump_contact_part};
use crate::consts::BACK_WALL_Y;
use crate::state::StateView;
use crate::team::{Role, RoleParams, RoleTracker, eta_to_ball, kickoff_pending, team_counts};

/// `[playmaking]` parameters (TM10).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PlaymakingParams {
    pub max_ball_dist: f64,
    pub eta_margin_s: f64,
    pub cap: u32,
    /// Scale for the 2v2 support car (`Role::Last`).
    pub support_scale_2v2: f64,
    /// Scale for the 3v3 support car (`Role::Second`).
    pub second_scale_3v3: f64,
    /// Scale for the 3v3 last man (0: he stays home).
    pub last_scale_3v3: f64,
    pub shade_width: f64,
    pub hard_target: f64,
    pub hard_power: f64,
    pub min_bump_dv: f64,
    pub victim_gate: bool,
}

impl Default for PlaymakingParams {
    fn default() -> Self {
        Self {
            max_ball_dist: 4000.0,
            eta_margin_s: 0.4,
            cap: 2,
            support_scale_2v2: 1.0,
            second_scale_3v3: 1.0,
            last_scale_3v3: 0.0,
            shade_width: 1200.0,
            hard_target: 900.0,
            hard_power: 1.5,
            min_bump_dv: 400.0,
            victim_gate: true,
        }
    }
}

/// Which event a term instance pays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayKind {
    Demo,
    Bump,
}

/// Our defensive third: `attack_sign * y` below this.
const DEF_THIRD: f64 = -BACK_WALL_Y / 3.0;

pub struct Playmaking {
    p: PlaymakingParams,
    kind: PlayKind,
    roles: RoleTracker,
    prev: Option<StateView>,
    last_toucher: Option<usize>,
    paid: Vec<u32>,
}

impl Playmaking {
    #[must_use]
    pub fn new(p: PlaymakingParams, roles: RoleParams, kind: PlayKind) -> Self {
        Self { p, kind, roles: RoleTracker::new(roles), prev: None, last_toucher: None, paid: Vec::new() }
    }

    /// Role scale of `agent` (0 = not a paid support car).
    fn role_scale(&self, state: &StateView, agent: usize, role: Role) -> f64 {
        let n = team_counts(state)[usize::from(state.cars[agent].team.min(1))];
        match (n, role) {
            (2, Role::Last) => self.p.support_scale_2v2,
            (3.., Role::Second) => self.p.second_scale_3v3,
            (3.., Role::Last) => self.p.last_scale_3v3,
            _ => 0.0,
        }
    }

    /// The gate above (the contact itself is detected by the caller).
    /// `s0` is the step before the contact, `s1` the contact step.
    fn qualifies(&self, s0: &StateView, s1: &StateView, roles: &[Role], agent: usize, victim: usize) -> bool {
        let car = &s1.cars[agent];
        if victim >= s1.cars.len() || s1.cars[victim].team == car.team || kickoff_pending(s0) {
            return false;
        }
        let sign = car.attack_sign();
        let Some(first) = (0..s1.cars.len())
            .find(|&i| i != agent && s1.cars[i].team == car.team && roles[i] == Role::First && !s1.cars[i].is_demoed())
        else {
            return false;
        };
        let ball = s1.ball.pos;
        let first_has_it = self.last_toucher == Some(first);
        let attacking = sign * ball.y > 0.0 || (first_has_it && sign * ball.y > DEF_THIRD);
        if !attacking || sign * ball.y <= DEF_THIRD || sign * car.pos.y <= DEF_THIRD {
            return false;
        }
        // Victim geometry from the step before the contact.
        let opps: Vec<usize> = (0..s0.cars.len()).filter(|&i| s0.cars[i].team != car.team && !s0.cars[i].is_demoed()).collect();
        if !opps.contains(&victim) {
            return false;
        }
        let d = |i: usize| s0.cars[i].pos.distance(s0.ball.pos);
        let closest = opps.iter().copied().min_by(|&a, &b| d(a).total_cmp(&d(b)));
        // Their own net is ours to attack: +sign * BACK_WALL_Y. Deepest =
        // largest sign * y (closest to the net we attack).
        let last_def = opps.iter().copied().max_by(|&a, &b| (sign * s0.cars[a].pos.y).total_cmp(&(sign * s0.cars[b].pos.y)));
        let eta_ok = eta_to_ball(s0, victim) <= eta_to_ball(s0, first) + self.p.eta_margin_s;
        let is_last = last_def == Some(victim);
        let in_play = closest == Some(victim) || eta_ok || (is_last && first_has_it);
        if !in_play {
            return false;
        }
        if car.pos.distance(ball) <= self.p.max_ball_dist {
            return true;
        }
        is_last && first_has_it && shading(s0.cars[victim].pos, s0.ball.pos, DVec3::new(0.0, sign * BACK_WALL_Y, 0.0), self.p.shade_width)
    }
}

/// Victim goal-side of the ball and within `width` of the ball -> net segment.
#[must_use]
pub fn shading(victim: DVec3, ball: DVec3, net: DVec3, width: f64) -> bool {
    let seg = DVec3::new(net.x - ball.x, net.y - ball.y, 0.0);
    let l2 = seg.length_squared();
    if l2 < 1e-6 {
        return false;
    }
    let v = DVec3::new(victim.x - ball.x, victim.y - ball.y, 0.0);
    let t = v.dot(seg) / l2;
    if t <= 0.0 || t > 1.0 {
        return false;
    }
    (v - seg * t).length() <= width
}

impl StepReward for Playmaking {
    fn reset(&mut self, initial: &StateView) {
        self.roles.reset();
        self.prev = Some(initial.clone());
        self.last_toucher = None;
        self.paid = vec![0; initial.cars.len()];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let roles = self.roles.update(state).to_vec();
        let mut r = vec![0.0; n];
        let prev = match self.prev.take() {
            Some(p) if p.cars.len() == n => p,
            _ => state.clone(),
        };
        if self.paid.len() != n {
            self.paid = vec![0; n];
        }
        // The toucher of this step (closest to the ball if several).
        if let Some(t) = (0..n)
            .filter(|&i| state.cars[i].ball_touches > 0)
            .min_by(|&a, &b| state.cars[a].pos.distance(state.ball.pos).total_cmp(&state.cars[b].pos.distance(state.ball.pos)))
        {
            self.last_toucher = Some(t);
        }
        if kickoff_pending(state) {
            self.last_toucher = None;
        }
        for agent in 0..n {
            let car = &state.cars[agent];
            let Some(victim) = car.bump_victim else { continue };
            if victim >= n || self.paid[agent] >= self.p.cap {
                continue;
            }
            let demo = state.cars[victim].is_demoed() && !prev.cars[victim].is_demoed();
            let new_bump = !demo && prev.cars[agent].bump_victim != Some(victim) && !state.cars[victim].is_demoed();
            if !demo && !new_bump {
                continue;
            }
            let scale = self.role_scale(state, agent, roles[agent]);
            if scale <= 0.0 || !self.qualifies(&prev, state, &roles, agent, victim) {
                continue;
            }
            let value = if demo {
                1.0
            } else {
                let dv = (state.cars[victim].vel - prev.cars[victim].vel).length();
                let own = (car.vel - prev.cars[agent].vel).length();
                let part = bump_contact_part(&prev.cars[agent], prev.cars[victim].pos);
                if part != ContactPart::Nose || dv < self.p.min_bump_dv || (self.p.victim_gate && dv <= own) {
                    0.0
                } else {
                    (dv / self.p.hard_target.max(1.0)).min(1.0).powf(self.p.hard_power)
                }
            };
            if value <= 0.0 {
                continue;
            }
            self.paid[agent] += 1;
            let pays = matches!((self.kind, demo), (PlayKind::Demo, true) | (PlayKind::Bump, false));
            if pays {
                r[agent] = scale * value;
            }
        }
        self.prev = Some(state.clone());
        r
    }
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::state::CarView;

    fn car(team: u8, x: f64, y: f64) -> CarView {
        CarView {
            team,
            pos: DVec3::new(x, y, 17.0),
            up: DVec3::Z,
            forward: if team == 0 { DVec3::Y } else { -DVec3::Y },
            right: if team == 0 { -DVec3::X } else { DVec3::X },
            ..CarView::default()
        }
    }

    /// 2v2, blue attacks +y. Blue 0 is first man on the ball at (0, 2500),
    /// blue 1 is support at (0, 1200); orange 2 is the defender at (0, 3600)
    /// (closest opponent, last defender), orange 3 is upfield at (2000, -500).
    fn attack_2v2() -> StateView {
        let mut s = StateView { cars: vec![car(0, 0.0, 2400.0), car(0, 0.0, 1200.0), car(1, 0.0, 3600.0), car(1, 2000.0, -500.0)], ..StateView::default() };
        s.ball.pos = DVec3::new(0.0, 2500.0, 93.0);
        s.ball.vel = DVec3::new(0.0, 100.0, 0.0);
        s
    }

    fn run(kind: PlayKind, s0: &StateView, s1: &StateView, first_touch: bool) -> Vec<f64> {
        let mut t = Playmaking::new(PlaymakingParams::default(), RoleParams::default(), kind);
        t.reset(s0);
        let mut touch = s0.clone();
        if first_touch {
            touch.cars[0].ball_touches = 1;
        }
        let _ = t.get_rewards(&touch);
        let mut s0b = s0.clone();
        s0b.cars[0].ball_touches = 0;
        let _ = t.get_rewards(&s0b);
        t.get_rewards(s1)
    }

    fn demo_of(s0: &StateView, agent: usize, victim: usize) -> StateView {
        let mut s1 = s0.clone();
        s1.cars[agent].bump_victim = Some(victim);
        s1.cars[victim].demo_respawn_timer = 3.0;
        s1
    }

    #[test]
    fn support_demo_of_the_defender_pays_first_man_and_teammates_do_not() {
        let s0 = attack_2v2();
        let r = run(PlayKind::Demo, &s0, &demo_of(&s0, 1, 2), true);
        assert_eq!(r, vec![0.0, 1.0, 0.0, 0.0]);
        // The first man's own demo is not playmaking.
        let mut s0f = s0.clone();
        s0f.cars[0].pos = DVec3::new(0.0, 3400.0, 17.0);
        let r = run(PlayKind::Demo, &s0f, &demo_of(&s0f, 0, 2), true);
        assert!(r.iter().all(|x| *x == 0.0), "{r:?}");
        // A teammate "victim" never pays.
        let mut s1 = s0.clone();
        s1.cars[1].bump_victim = Some(0);
        assert!(run(PlayKind::Bump, &s0, &s1, true).iter().all(|x| *x == 0.0));
    }

    #[test]
    fn out_of_play_victim_and_defending_context_do_not_pay() {
        let s0 = attack_2v2();
        // Orange 3 upfield: not closest, slow, not last defender.
        assert!(run(PlayKind::Demo, &s0, &demo_of(&s0, 1, 3), true).iter().all(|x| *x == 0.0));
        // Ball in our defensive third: nothing pays.
        let mut d = attack_2v2();
        for c in &mut d.cars {
            c.pos.y -= 6000.0;
        }
        d.ball.pos.y -= 6000.0;
        assert!(run(PlayKind::Demo, &d, &demo_of(&d, 1, 2), true).iter().all(|x| *x == 0.0));
        // Kickoff pending: nothing pays.
        let mut k = attack_2v2();
        k.ball.pos = DVec3::new(0.0, 0.0, 93.0);
        k.ball.vel = DVec3::ZERO;
        assert!(run(PlayKind::Demo, &k, &demo_of(&k, 1, 2), false).iter().all(|x| *x == 0.0));
    }

    #[test]
    fn far_contact_pays_only_on_a_shading_last_defender_with_possession() {
        let mut s0 = attack_2v2();
        s0.cars[1].pos = DVec3::new(0.0, -1200.0, 17.0); // 3700 from ball: in range
        assert_eq!(run(PlayKind::Demo, &s0, &demo_of(&s0, 1, 2), true)[1], 1.0);
        s0.cars[1].pos = DVec3::new(0.0, -1600.0, 17.0); // 4100: needs shading
        assert_eq!(run(PlayKind::Demo, &s0, &demo_of(&s0, 1, 2), true)[1], 1.0, "defender shades the shot, first man has it");
        assert_eq!(run(PlayKind::Demo, &s0, &demo_of(&s0, 1, 2), false)[1], 0.0, "no possession: ball in their half alone is not enough far away");
        s0.cars[2].pos = DVec3::new(3000.0, 3600.0, 17.0); // off the shot line
        assert_eq!(run(PlayKind::Demo, &s0, &demo_of(&s0, 1, 2), true)[1], 0.0);
    }

    #[test]
    fn bumps_need_the_nose_and_impact_and_the_cap_holds() {
        let s0 = attack_2v2();
        let mut s1 = s0.clone();
        s1.cars[1].bump_victim = Some(2);
        s1.cars[2].vel = DVec3::new(0.0, 900.0, 0.0);
        let r = run(PlayKind::Bump, &s0, &s1, true);
        assert!((r[1] - 1.0).abs() < 1e-9, "{r:?}");
        assert_eq!(run(PlayKind::Demo, &s0, &s1, true)[1], 0.0, "the demo term ignores bumps");
        let mut soft = s1.clone();
        soft.cars[2].vel = DVec3::new(0.0, 300.0, 0.0);
        assert_eq!(run(PlayKind::Bump, &s0, &soft, true)[1], 0.0);
        // Upside down toward the victim: the wheels face it.
        let mut wheels = s0.clone();
        wheels.cars[1].up = -DVec3::Y;
        wheels.cars[1].forward = DVec3::Z;
        let mut w1 = s1.clone();
        w1.cars[1].up = -DVec3::Y;
        w1.cars[1].forward = DVec3::Z;
        assert_eq!(run(PlayKind::Bump, &wheels, &w1, true)[1], 0.0, "wheel contact");
        // Cap: 2 paid events per car per episode.
        let mut t = Playmaking::new(PlaymakingParams::default(), RoleParams::default(), PlayKind::Bump);
        t.reset(&s0);
        let mut touch = s0.clone();
        touch.cars[0].ball_touches = 1;
        let mut paid = 0;
        for _ in 0..4 {
            let _ = t.get_rewards(&touch);
            let _ = t.get_rewards(&s0);
            if t.get_rewards(&s1)[1] > 0.0 {
                paid += 1;
            }
        }
        assert_eq!(paid, 2);
    }

    #[test]
    fn three_v_three_pays_second_not_last_and_one_v_one_is_a_no_op() {
        // Blue: 0 first at the ball, 1 second at (0, 1500), 2 last at (0, -1000).
        let mut s = StateView { cars: vec![car(0, 0.0, 2400.0), car(0, 0.0, 1500.0), car(0, 0.0, -1000.0), car(1, 0.0, 3600.0), car(1, 1500.0, 0.0), car(1, -1500.0, 0.0)], ..StateView::default() };
        s.ball.pos = DVec3::new(0.0, 2500.0, 93.0);
        let r = run(PlayKind::Demo, &s, &demo_of(&s, 1, 3), true);
        assert_eq!(r[1], 1.0, "{r:?}");
        let r = run(PlayKind::Demo, &s, &demo_of(&s, 2, 3), true);
        assert_eq!(r[2], 0.0, "3v3 last man stays home");
        let mut one = StateView { cars: vec![car(0, 0.0, 1200.0), car(1, 0.0, 3600.0)], ..StateView::default() };
        one.ball.pos = DVec3::new(0.0, 2500.0, 93.0);
        assert!(run(PlayKind::Demo, &one, &demo_of(&one, 0, 1), false).iter().all(|x| *x == 0.0));
    }
}
