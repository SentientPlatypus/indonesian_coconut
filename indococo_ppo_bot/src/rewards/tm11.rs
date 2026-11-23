//! TM11 terms. `defend_carry` (shaping) and `carry_break` (event): a KRC-style
//! (Lucy-SKG kinesthetic reward combination, multiplicative so every factor
//! must be present) reward for challenging an OPPONENT's grounded roof carry
//! from the goal side. `kickoff_outcome`: the 1.5 s kickoff result (the
//! tm8_audit_traces.py definition) paid to the 1st/2nd man. All three are
//! individual (no team-spirit blend) and not zero-sum.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::{BACK_WALL_Y, CAR_MAX_SPEED};
use crate::state::{CarView, StateView};
use crate::team::{Role, RoleParams, RoleTracker, kickoff_pending, team_counts};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DefendCarryParams {
    pub roof_min: f64,       // 95: (ball - car) . up
    pub roof_max: f64,       // 175
    pub max_planar: f64,     // 160: offset perpendicular to up
    pub max_rel_speed: f64,  // 500: |ball.vel - car.vel|
    pub max_ball_z: f64,     // 320: grounded carries only (not air dribbles)
    pub min_steps: u32,      // 4: streak before it counts (0.27 s at 15 Hz)
    pub threat_dist: f64,    // 7000: ball within this of the defended back wall
    pub prox_scale: f64,     // 2000
    pub contact_window: u32, // 3: defender contact this recent counts for a break
    pub confirm_steps: u32,  // 8: the carrier must not regain the carry for this long
    pub cap: u32,            // 3 paid breaks per car per episode
}
impl Default for DefendCarryParams {
    fn default() -> Self {
        Self { roof_min: 95.0, roof_max: 175.0, max_planar: 160.0, max_rel_speed: 500.0,
               max_ball_z: 320.0, min_steps: 4, threat_dist: 7000.0, prox_scale: 2000.0,
               contact_window: 3, confirm_steps: 8, cap: 3 }
    }
}

/// True if `car` (grounded, alive) has the ball resting on its roof.
#[must_use]
pub fn on_roof(car: &CarView, ball_pos: DVec3, ball_vel: DVec3, p: &DefendCarryParams) -> bool {
    if car.is_demoed() || !car.on_world_surface() || ball_pos.z > p.max_ball_z {
        return false;
    }
    let rel = ball_pos - car.pos;
    let up = rel.dot(car.up);
    let planar = (rel - car.up * up).length();
    (p.roof_min..=p.roof_max).contains(&up) && planar <= p.max_planar && (ball_vel - car.vel).length() <= p.max_rel_speed
}

/// Per-car roof streaks. `update` returns the carrier: the longest active
/// streak >= min_steps whose ball is within threat_dist of the net that car attacks.
#[derive(Clone, Debug, Default)]
pub struct CarryTracker { streak: Vec<u32> }
impl CarryTracker {
    pub fn reset(&mut self) { self.streak.clear(); }
    pub fn update(&mut self, s: &StateView, p: &DefendCarryParams) -> Option<usize> {
        let n = s.cars.len();
        if self.streak.len() != n { self.streak = vec![0; n]; }
        if kickoff_pending(s) { self.streak.iter_mut().for_each(|x| *x = 0); return None; }
        for i in 0..n {
            self.streak[i] = if on_roof(&s.cars[i], s.ball.pos, s.ball.vel, p) { self.streak[i].saturating_add(1) } else { 0 };
        }
        (0..n)
            .filter(|&i| self.streak[i] >= p.min_steps.max(1))
            .filter(|&i| BACK_WALL_Y - s.cars[i].attack_sign() * s.ball.pos.y <= p.threat_dist)
            .max_by_key(|&i| (self.streak[i], std::cmp::Reverse(i)))
    }
}

/// KRC challenge shape in [0,1]: cbrt(G * C * P).
/// G = goal-side: cos(car-ball, own_net-ball) in xy, clipped at 0.
/// C = closing: velocity toward the ball / 2300, clamped to [0,1].
/// P = proximity: exp(-dist / prox_scale).
/// Any factor at 0 gives 0, so an idle goalie, a ball-side chaser and a far car all get 0.
#[must_use]
pub fn challenge_krc(def: &CarView, ball: DVec3, prox_scale: f64) -> f64 {
    let net_y = -def.attack_sign() * BACK_WALL_Y; // the net this car defends
    let to_net = DVec3::new(-ball.x, net_y - ball.y, 0.0);
    let off = DVec3::new(def.pos.x - ball.x, def.pos.y - ball.y, 0.0);
    let (ln, lo) = (to_net.length(), off.length());
    let d = ball - def.pos;
    let dist = d.length();
    if ln < 1e-6 || lo < 1e-6 || dist < 1e-6 { return 0.0; }
    let g = (off.dot(to_net) / (ln * lo)).max(0.0);
    let c = (def.vel.dot(d / dist) / CAR_MAX_SPEED).clamp(0.0, 1.0);
    let pr = (-dist / prox_scale).exp();
    (g * c * pr).cbrt()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarryPart { Shape, Break }

#[derive(Clone, Debug)]
struct Pending { payee: usize, carrier: usize, left: u32 }

pub struct DefendCarry {
    p: DefendCarryParams,
    part: CarryPart,
    roles: RoleTracker,
    carry: CarryTracker,
    prev_carrier: Option<usize>,
    touch_age: Vec<u32>,                  // steps since the car's last ball touch (u32::MAX = never)
    bump_on: Vec<Option<(usize, u32)>>,   // (victim, steps since the bump)
    prev_bump: Vec<Option<usize>>,
    pending: Option<Pending>,
    paid: Vec<u32>,
}

impl DefendCarry {
    #[must_use]
    pub fn new(p: DefendCarryParams, roles: RoleParams, part: CarryPart) -> Self {
        Self { p, part, roles: RoleTracker::new(roles), carry: CarryTracker::default(), prev_carrier: None,
               touch_age: Vec::new(), bump_on: Vec::new(), prev_bump: Vec::new(), pending: None, paid: Vec::new() }
    }
    fn resize(&mut self, n: usize) {
        if self.touch_age.len() != n {
            self.touch_age = vec![u32::MAX; n]; self.bump_on = vec![None; n];
            self.prev_bump = vec![None; n]; self.paid = vec![0; n];
        }
    }
}

impl StepReward for DefendCarry {
    fn reset(&mut self, initial: &StateView) {
        self.roles.reset(); self.carry.reset(); self.prev_carrier = None; self.pending = None;
        self.touch_age.clear(); self.resize(initial.cars.len());
        for (i, c) in initial.cars.iter().enumerate() { self.prev_bump[i] = c.bump_victim; }
    }

    fn get_rewards(&mut self, s: &StateView) -> Vec<f64> {
        let n = s.cars.len();
        self.resize(n);
        let roles = self.roles.update(s).to_vec();
        let mut r = vec![0.0; n];
        let team = |i: usize| s.cars[i].team.min(1);
        // 1. Contact ages.
        for i in 0..n {
            let c = &s.cars[i];
            self.touch_age[i] = if c.ball_touches > 0 { 0 } else { self.touch_age[i].saturating_add(1) };
            self.bump_on[i] = self.bump_on[i].map(|(v, a)| (v, a.saturating_add(1)));
            if c.bump_victim.is_some() && c.bump_victim != self.prev_bump[i] {
                self.bump_on[i] = c.bump_victim.map(|v| (v, 0));
            }
            self.prev_bump[i] = c.bump_victim;
        }
        let carrier = self.carry.update(s, &self.p);
        // 2. Resolve a pending break (before detecting new ones, so a new pending isn't decremented this step).
        if let Some(pd) = self.pending.as_mut() {
            let resumed = carrier == Some(pd.carrier);
            let conceded = s.scoring_team().map(|t| t.min(1)) == Some(team(pd.carrier));
            if resumed || conceded || s.goal_scored() && conceded {
                self.pending = None;
            } else {
                pd.left = pd.left.saturating_sub(1);
                if pd.left == 0 {
                    let k = pd.payee;
                    if self.paid[k] < self.p.cap {
                        self.paid[k] += 1;
                        if self.part == CarryPart::Break { r[k] = 1.0; }
                    }
                    self.pending = None;
                }
            }
        }
        // 3. Detect a break: the carry by `pc` ended this step (no carrier, or a different one).
        if let Some(pc) = self.prev_carrier {
            if carrier != Some(pc) && self.pending.is_none() {
                let w = self.p.contact_window;
                let dteam = 1 - team(pc);
                let payee = (0..n)
                    .filter(|&k| team(k) == dteam)
                    .filter(|&k| self.touch_age[k] <= w || matches!(self.bump_on[k], Some((v, a)) if v == pc && a <= w))
                    .min_by(|&a, &b| s.cars[a].pos.distance(s.ball.pos).total_cmp(&s.cars[b].pos.distance(s.ball.pos)));
                if let Some(k) = payee {
                    self.pending = Some(Pending { payee: k, carrier: pc, left: self.p.confirm_steps.max(1) });
                }
            }
        }
        // 4. Shaping: only the defending team's first man, only during a live threatening carry.
        if self.part == CarryPart::Shape {
            if let Some(c) = carrier {
                let dteam = 1 - team(c);
                for d in 0..n {
                    if team(d) == dteam && roles[d] == Role::First && !s.cars[d].is_demoed() {
                        r[d] = challenge_krc(&s.cars[d], s.ball.pos, self.p.prox_scale);
                    }
                }
            }
        }
        self.prev_carrier = carrier;
        r
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct KickoffOutcomeParams {
    pub horizon_steps: u32,   // 22 = int(1.5 * 15), same as tm8_audit_traces.py
    pub y_scale: f64,         // 1500: team-frame ball y giving +-1 (the audit's win/loss line is 800)
    pub min_team: usize,      // 2: 1v1 gets nothing
    pub last_scale_2v2: f64,  // 0.5: the 2v2 non-taker (Role::Last) gets half
    pub last_scale_3v3: f64,  // 0.0: the 3v3 last man is not paid (stays back)
}
impl Default for KickoffOutcomeParams {
    fn default() -> Self { Self { horizon_steps: 22, y_scale: 1500.0, min_team: 2, last_scale_2v2: 0.5, last_scale_3v3: 0.0 } }
}

enum KoState { Off, Pending, Counting { left: u32, roles: Vec<Role> }, Done }

pub struct KickoffOutcome { p: KickoffOutcomeParams, roles: RoleTracker, st: KoState }

impl KickoffOutcome {
    #[must_use]
    pub fn new(p: KickoffOutcomeParams, roles: RoleParams) -> Self { Self { p, roles: RoleTracker::new(roles), st: KoState::Off } }
}

impl StepReward for KickoffOutcome {
    fn reset(&mut self, initial: &StateView) {
        self.roles.reset();
        self.st = if kickoff_pending(initial) { KoState::Pending } else { KoState::Off };
    }
    fn get_rewards(&mut self, s: &StateView) -> Vec<f64> {
        let n = s.cars.len();
        let before = self.roles.roles().to_vec();   // roles as of the previous step
        self.roles.update(s);
        let mut r = vec![0.0; n];
        match &mut self.st {
            KoState::Pending => {
                if !kickoff_pending(s) {
                    let roles = if before.len() == n { before } else { self.roles.roles().to_vec() };
                    self.st = KoState::Counting { left: self.p.horizon_steps.max(1), roles };
                }
            }
            KoState::Counting { left, roles } => {
                if s.goal_scored() {
                    self.st = KoState::Done;
                } else {
                    *left -= 1;
                    if *left == 0 {
                        let counts = team_counts(s);
                        for i in 0..n {
                            let c = &s.cars[i];
                            let ts = counts[usize::from(c.team.min(1))];
                            if ts < self.p.min_team { continue; }
                            let scale = match (ts, roles[i]) {
                                (_, Role::First) => 1.0,
                                (2, Role::Last) => self.p.last_scale_2v2,
                                (t, Role::Second) if t >= 3 => 1.0,
                                (t, Role::Last) if t >= 3 => self.p.last_scale_3v3,
                                _ => 0.0, // Out (demoed) or anything else
                            };
                            r[i] = scale * (c.attack_sign() * s.ball.pos.y / self.p.y_scale).clamp(-1.0, 1.0);
                        }
                        self.st = KoState::Done;
                    }
                }
            }
            KoState::Off | KoState::Done => {}
        }
        r
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::GOAL_THRESHOLD;
    use crate::state::{CarView, StateView};
    use glam::DVec3;

    fn car(team: u8, x: f64, y: f64) -> CarView {
        CarView {
            team,
            pos: DVec3::new(x, y, 17.0),
            up: DVec3::Z,
            forward: if team == 0 { DVec3::Y } else { -DVec3::Y },
            right: if team == 0 { -DVec3::X } else { DVec3::X },
            wheels_with_contact: [true; 4],
            ..CarView::default()
        }
    }

    fn roles_fast() -> RoleParams {
        RoleParams { margin: 0.0, hold: 1, obs_tiebreak: false, touch_mode: "instant".into() }
    }

    fn carry_2v2() -> StateView {
        // Orange car 1 carries at y=-3000 (attacking blue's net at y=-5120).
        let mut s = StateView {
            cars: vec![
                car(0, 0.0, -4300.0),
                car(1, 0.0, -3000.0),
                car(0, 0.0, -5000.0),
                car(1, 2000.0, 0.0),
            ],
            ..StateView::default()
        };
        s.cars[0].vel = DVec3::new(0.0, 1500.0, 0.0);
        s.ball.pos = DVec3::new(0.0, -3000.0, 147.0); // 17 + 130
        s.ball.vel = s.cars[1].vel;
        s
    }

    fn shape() -> DefendCarry {
        DefendCarry::new(DefendCarryParams::default(), roles_fast(), CarryPart::Shape)
    }
    fn brk() -> DefendCarry {
        DefendCarry::new(DefendCarryParams::default(), roles_fast(), CarryPart::Break)
    }

    #[test]
    fn on_roof_cases() {
        let p = DefendCarryParams::default();
        let mut c = car(0, 0.0, 0.0);
        let vel = DVec3::ZERO;
        assert!(on_roof(&c, c.pos + DVec3::new(0.0, 0.0, 130.0), vel, &p));
        assert!(!on_roof(&c, c.pos + DVec3::new(0.0, 0.0, 200.0), vel, &p));
        assert!(!on_roof(&c, c.pos + DVec3::new(200.0, 0.0, 130.0), vel, &p));
        assert!(!on_roof(&c, c.pos + DVec3::new(0.0, 0.0, 130.0), DVec3::new(600.0, 0.0, 0.0), &p));
        c.demo_respawn_timer = 1.0;
        assert!(!on_roof(&c, c.pos + DVec3::new(0.0, 0.0, 130.0), vel, &p));
        let mut c = car(0, 0.0, 0.0);
        c.pos.z = 0.0;
        assert!(!on_roof(&c, DVec3::new(0.0, 0.0, 400.0), vel, &p));
    }

    #[test]
    fn shape_pays_goalside_closing_first_only() {
        let mut t = shape();
        let s = carry_2v2();
        t.reset(&s);
        for step in 1..=5 {
            let r = t.get_rewards(&s);
            if step < 4 {
                assert!(r.iter().all(|x| *x == 0.0), "step {step}: {r:?}");
            } else {
                assert!(r[0] > 0.3, "step {step} r[0]={}", r[0]);
                assert_eq!(r[2], 0.0, "{r:?}");
                assert_eq!(r[1], 0.0);
                assert_eq!(r[3], 0.0);
            }
        }
    }

    fn shape_on(s: &StateView) -> Vec<f64> {
        let mut t = shape();
        t.reset(s);
        let mut last = vec![0.0; s.cars.len()];
        for _ in 0..5 {
            last = t.get_rewards(s);
        }
        last
    }

    #[test]
    fn anti_farm_goal_side_idle() {
        let mut s = carry_2v2();
        s.cars[0].vel = DVec3::ZERO;
        assert!(shape_on(&s).iter().all(|x| *x == 0.0));
    }

    #[test]
    fn anti_farm_ball_side() {
        let mut s = carry_2v2();
        s.cars[0].pos.y = -2000.0; // behind the carrier (ball at -3000, blue net at -5120)
        s.cars[0].vel = DVec3::new(0.0, -1500.0, 0.0); // driving at the ball
        assert!(shape_on(&s)[0] == 0.0);
    }

    #[test]
    fn anti_farm_ground_ball() {
        let mut s = carry_2v2();
        s.ball.pos.z = 93.0;
        assert!(shape_on(&s).iter().all(|x| *x == 0.0));
    }

    #[test]
    fn anti_farm_own_teammate_carry() {
        // Blue teammate carries; no blue car should be paid.
        let mut s = carry_2v2();
        s.cars[0].pos = DVec3::new(0.0, -3000.0, 17.0);
        s.cars[0].vel = DVec3::ZERO;
        s.cars[1].pos = DVec3::new(0.0, -4300.0, 17.0);
        s.ball.pos = DVec3::new(0.0, -3000.0, 147.0);
        s.ball.vel = s.cars[0].vel;
        let r = shape_on(&s);
        assert!(r[0] == 0.0 && r[2] == 0.0, "{r:?}");
    }

    #[test]
    fn anti_farm_threat_distance() {
        let mut s = carry_2v2();
        // Orange attacks -y; distance to that wall is BACK_WALL_Y - (-1)*y = 5120+y
        s.cars[1].pos.y = 1000.0;
        s.ball.pos = DVec3::new(0.0, 1000.0, 147.0);
        s.cars[0].pos.y = -300.0;
        s.cars[0].vel = DVec3::new(0.0, 1500.0, 0.0);
        let r = shape_on(&s);
        assert!(r[0] > 0.0, "6120 < 7000 should pay: {r:?}");
        s.cars[1].pos.y = 3000.0;
        s.ball.pos = DVec3::new(0.0, 3000.0, 147.0);
        let r = shape_on(&s);
        assert_eq!(r[0], 0.0, "8120 > 7000 should not pay: {r:?}");
    }

    #[test]
    fn anti_farm_kickoff_pending() {
        let mut s = carry_2v2();
        s.ball.pos = DVec3::new(0.0, 0.0, 93.0);
        s.ball.vel = DVec3::ZERO;
        assert!(shape_on(&s).iter().all(|x| *x == 0.0));
    }

    fn run_steps(t: &mut DefendCarry, states: &[StateView]) -> Vec<(usize, Vec<f64>)> {
        states.iter().enumerate().map(|(i, s)| (i, t.get_rewards(s))).collect()
    }

    #[test]
    fn break_event() {
        let mut s = carry_2v2();
        // (a) carry 5 steps, then blue 0 touches and the ball leaves the roof;
        // Break pays r[0]=1.0 exactly 8 steps later.
        let mut b = brk();
        let mut sh = shape();
        b.reset(&s);
        sh.reset(&s);
        for _ in 0..5 {
            assert_eq!(b.get_rewards(&s)[0], 0.0);
            sh.get_rewards(&s);
        }
        let mut broken = s.clone();
        broken.cars[0].ball_touches = 1;
        broken.ball.pos.z = 93.0; // leaves the roof
        assert_eq!(b.get_rewards(&broken)[0], 0.0);
        assert_ne!(sh.get_rewards(&broken)[0], 1.0);
        let mut after = broken.clone();
        after.cars[0].ball_touches = 0;
        for i in 1..=8 {
            let rb = b.get_rewards(&after);
            let rs = sh.get_rewards(&after);
            if i < 8 {
                assert_eq!(rb[0], 0.0, "break step {i}");
            } else {
                assert_eq!(rb[0], 1.0, "break should pay at step 8");
            }
            assert_ne!(rs[0], 1.0, "shape must never pay the 1.0");
        }

        // (b) carrier back on the roof for 4 steps inside the window -> no pay
        let mut b = brk();
        b.reset(&s);
        for _ in 0..5 { b.get_rewards(&s); }
        b.get_rewards(&broken);
        for _ in 0..4 {
            let r = b.get_rewards(&s); // carrier on roof again
            assert_eq!(r[0], 0.0);
        }
        for _ in 0..8 {
            assert_eq!(b.get_rewards(&after)[0], 0.0);
        }

        // (c) carrier drops with no blue contact -> no pay
        let mut b = brk();
        b.reset(&s);
        for _ in 0..5 { b.get_rewards(&s); }
        let mut drop = s.clone();
        drop.ball.pos.z = 93.0;
        b.get_rewards(&drop);
        for _ in 0..10 {
            assert_eq!(b.get_rewards(&drop)[0], 0.0);
        }

        // (d) contact via bump_victim = Some(carrier) (new this step) and no touch -> pays
        let mut b = brk();
        b.reset(&s);
        for _ in 0..5 { b.get_rewards(&s); }
        let mut bump = s.clone();
        bump.ball.pos.z = 93.0;
        bump.cars[0].bump_victim = Some(1);
        assert_eq!(b.get_rewards(&bump)[0], 0.0);
        let mut after_b = bump.clone();
        after_b.cars[0].bump_victim = Some(1); // still set, but not "new"
        for i in 1..=8 {
            let r = b.get_rewards(&after_b);
            if i == 8 { assert_eq!(r[0], 1.0, "bump break pays at 8"); } else { assert_eq!(r[0], 0.0); }
        }

        // (e) 4 qualifying breaks -> only the first 3 pay
        let mut b = brk();
        b.reset(&s);
        let mut paid = 0u32;
        for _k in 0..4 {
            for _ in 0..5 { b.get_rewards(&s); }
            let mut ev = s.clone();
            ev.cars[0].ball_touches = 1;
            ev.ball.pos.z = 93.0;
            b.get_rewards(&ev);
            ev.cars[0].ball_touches = 0;
            for i in 1..=8 {
                if b.get_rewards(&ev)[0] == 1.0 && i == 8 { paid += 1; }
            }
            // resume a new carry
        }
        assert_eq!(paid, 3);

        // (f) Orange scores (ball.y < -GOAL_THRESHOLD) inside the window -> no pay
        let mut b = brk();
        b.reset(&s);
        for _ in 0..5 { b.get_rewards(&s); }
        b.get_rewards(&broken);
        let mut goal = after.clone();
        goal.ball.pos.y = -GOAL_THRESHOLD - 10.0;
        for _ in 0..10 {
            assert_eq!(b.get_rewards(&goal)[0], 0.0);
        }
    }

    fn ko_state_3v3(pending: bool, ball_y: f64) -> StateView {
        // Blue 0 first (closest), 2 second, 4 last (furthest back).
        let mut s = StateView {
            cars: vec![
                car(0, 0.0, -200.0),
                car(1, 0.0, 200.0),
                car(0, 0.0, -800.0),
                car(1, 0.0, 800.0),
                car(0, 0.0, -2000.0),
                car(1, 0.0, 2000.0),
            ],
            ..StateView::default()
        };
        if pending {
            s.ball.pos = DVec3::new(0.0, 0.0, 93.0);
            s.ball.vel = DVec3::ZERO;
        } else {
            s.ball.pos = DVec3::new(0.0, ball_y, 93.0);
            s.ball.vel = DVec3::new(0.0, 100.0, 0.0);
        }
        s
    }

    fn kick(p: KickoffOutcomeParams) -> KickoffOutcome {
        KickoffOutcome::new(p, roles_fast())
    }

    #[test]
    fn kickoff_outcome() {
        let p = KickoffOutcomeParams::default();
        // (a) 3v3: 2 pending steps, ball moves at ft; ft+21 nothing; ft+22 y=+1500 pays.
        let mut t = kick(p.clone());
        let pend = ko_state_3v3(true, 0.0);
        t.reset(&pend);
        assert!(t.get_rewards(&pend).iter().all(|x| *x == 0.0));
        assert!(t.get_rewards(&pend).iter().all(|x| *x == 0.0));
        let moving = ko_state_3v3(false, 1500.0);
        let r = t.get_rewards(&moving); // ft
        assert!(r.iter().all(|x| *x == 0.0), "{r:?}");
        for i in 1..=21 {
            let r = t.get_rewards(&moving);
            assert!(r.iter().all(|x| *x == 0.0), "ft+{i}: {r:?}");
        }
        let r = t.get_rewards(&moving); // ft+22
        assert_eq!(r[0], 1.0, "blue first {r:?}");
        assert_eq!(r[2], 1.0, "blue second {r:?}");
        assert_eq!(r[4], 0.0, "blue last {r:?}");
        assert_eq!(r[1], -1.0, "orange first {r:?}");
        assert_eq!(r[3], -1.0, "orange second {r:?}");
        assert_eq!(r[5], 0.0, "orange last {r:?}");

        // (b) 2v2: Last gets ±0.5
        let mut t = kick(p.clone());
        let mut pend2 = StateView {
            cars: vec![car(0, 0.0, -200.0), car(1, 0.0, 200.0), car(0, 0.0, -2000.0), car(1, 0.0, 2000.0)],
            ..StateView::default()
        };
        pend2.ball.pos = DVec3::new(0.0, 0.0, 93.0);
        t.reset(&pend2);
        t.get_rewards(&pend2);
        t.get_rewards(&pend2);
        let mut mv = pend2.clone();
        mv.ball.pos = DVec3::new(0.0, 1500.0, 93.0);
        mv.ball.vel = DVec3::new(0.0, 100.0, 0.0);
        t.get_rewards(&mv);
        for _ in 0..21 { t.get_rewards(&mv); }
        let r = t.get_rewards(&mv);
        assert_eq!(r[0], 1.0, "{r:?}");
        assert_eq!(r[2], 0.5, "{r:?}");
        assert_eq!(r[1], -1.0, "{r:?}");
        assert_eq!(r[3], -0.5, "{r:?}");

        // (c) 1v1: always 0
        let mut t = kick(p.clone());
        let mut pend1 = StateView { cars: vec![car(0, 0.0, -200.0), car(1, 0.0, 200.0)], ..StateView::default() };
        pend1.ball.pos = DVec3::new(0.0, 0.0, 93.0);
        t.reset(&pend1);
        t.get_rewards(&pend1);
        let mut mv = pend1.clone();
        mv.ball.pos = DVec3::new(0.0, 1500.0, 93.0);
        mv.ball.vel = DVec3::new(0.0, 100.0, 0.0);
        t.get_rewards(&mv);
        for _ in 0..30 {
            assert!(t.get_rewards(&mv).iter().all(|x| *x == 0.0));
        }

        // (d) Goal at ft+10 -> nothing ever
        let mut t = kick(p.clone());
        t.reset(&pend);
        t.get_rewards(&pend);
        t.get_rewards(&pend);
        t.get_rewards(&moving);
        for _ in 0..9 { t.get_rewards(&moving); }
        let mut g = moving.clone();
        g.ball.pos.y = GOAL_THRESHOLD + 10.0;
        t.get_rewards(&g);
        for _ in 0..20 {
            assert!(t.get_rewards(&g).iter().all(|x| *x == 0.0));
            assert!(t.get_rewards(&moving).iter().all(|x| *x == 0.0));
        }

        // (e) episode not starting on a kickoff -> 0 forever
        let mut t = kick(p.clone());
        t.reset(&moving);
        for _ in 0..40 {
            assert!(t.get_rewards(&moving).iter().all(|x| *x == 0.0));
        }

        // (f) ball y=+600 -> ±0.4
        let mut t = kick(p);
        t.reset(&pend);
        t.get_rewards(&pend);
        t.get_rewards(&pend);
        let mut mv = ko_state_3v3(false, 600.0);
        t.get_rewards(&mv);
        for _ in 0..21 { t.get_rewards(&mv); }
        let r = t.get_rewards(&mv);
        assert!((r[0] - 0.4).abs() < 1e-9, "{r:?}");
        assert!((r[2] - 0.4).abs() < 1e-9, "{r:?}");
        assert!((r[1] + 0.4).abs() < 1e-9, "{r:?}");
    }
}
