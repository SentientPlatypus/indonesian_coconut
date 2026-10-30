//! `ContactQualityReward`: score every car-car contact once, from each side.
//! Contacts are detected by proximity (centres closer than 175 uu, re-armed
//! past 220 uu) and scored one step after they start so the victim's
//! velocity change covers the impact. Shell hits pay
//! `part_scale[part] * hardness`, wheel hits cost `wheel_penalty * hardness`,
//! with `hardness = min(1, dv / hard_target) ^ hard_power`. Only near the
//! ball (`ball_max_dist`).

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;

use super::StepReward;
use super::mech::{ContactPart, bump_contact_part};
use crate::math::safe_norm;
use crate::state::StateView;

const START_DIST: f64 = 175.0;
const END_DIST: f64 = 220.0;

pub struct ContactQuality {
    hard_target: f64,
    hard_power: f64,
    wheel_penalty: f64,
    ball_max_dist: f64,
    touching: BTreeSet<(usize, usize)>,
    /// (agent, victim) -> (part, victim velocity before the contact).
    pending: BTreeMap<(usize, usize), (ContactPart, DVec3)>,
    pending_order: Vec<(usize, usize)>,
    prev_vel: Vec<DVec3>,
}

impl ContactQuality {
    #[must_use]
    pub fn new(hard_target: f64, wheel_penalty: f64) -> Self {
        Self {
            hard_target,
            hard_power: 1.5,
            wheel_penalty,
            ball_max_dist: 2500.0,
            touching: BTreeSet::new(),
            pending: BTreeMap::new(),
            pending_order: Vec::new(),
            prev_vel: Vec::new(),
        }
    }

    fn part_scale(part: ContactPart) -> f64 {
        match part {
            ContactPart::Nose => 1.0,
            ContactPart::Roof | ContactPart::Side => 0.6,
            ContactPart::Back => 0.3,
            ContactPart::Wheels => 0.0,
        }
    }

    /// `ContactTracker.update`: contacts that finished resolving this step.
    fn update(&mut self, state: &StateView) -> Vec<(usize, ContactPart, f64)> {
        let vel: Vec<DVec3> = state.cars.iter().map(|c| c.vel).collect();
        let mut out = Vec::new();
        for key in self.pending_order.drain(..) {
            let (part, v0) = self.pending.remove(&key).expect("pending contact");
            if let Some(v) = vel.get(key.1) {
                out.push((key.0, part, (*v - v0).length()));
            }
        }
        for a in 0..state.cars.len() {
            for b in a + 1..state.cars.len() {
                let (ca, cb) = (&state.cars[a], &state.cars[b]);
                if ca.team == cb.team {
                    continue;
                }
                let d = (cb.pos - ca.pos).length();
                let pair = (a, b);
                if !self.touching.contains(&pair) && d < START_DIST {
                    self.touching.insert(pair);
                    for (x, y) in [(a, b), (b, a)] {
                        let part = bump_contact_part(&state.cars[x], state.cars[y].pos);
                        let before = self.prev_vel.get(y).copied().unwrap_or(vel[y]);
                        if self.pending.insert((x, y), (part, before)).is_none() {
                            self.pending_order.push((x, y));
                        }
                    }
                } else if self.touching.contains(&pair) && d >= END_DIST {
                    self.touching.remove(&pair);
                }
            }
        }
        self.prev_vel = vel;
        out
    }
}

impl StepReward for ContactQuality {
    fn reset(&mut self, initial: &StateView) {
        self.touching.clear();
        self.pending.clear();
        self.pending_order.clear();
        self.prev_vel = initial.cars.iter().map(|c| c.vel).collect();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        for (agent, part, dv) in self.update(state) {
            if safe_norm(state.ball.pos - state.cars[agent].pos) > self.ball_max_dist {
                continue;
            }
            let hardness = (dv / self.hard_target.max(1.0))
                .min(1.0)
                .powf(self.hard_power);
            if part == ContactPart::Wheels {
                rewards[agent] -= self.wheel_penalty * hardness;
            } else {
                rewards[agent] += Self::part_scale(part) * hardness;
            }
        }
        rewards
    }
}
