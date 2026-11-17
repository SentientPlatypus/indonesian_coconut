//! Weighted sum of reward terms, exposed to the environment as an RLGym
//! [`Reward`].

use rlgymppo_utils::AvgTracker;
use rlgymppo_utils::rlgym::{GameState, Reward};
use rlgymppo_utils::shared_info::SharedInfoReport;

use super::StepReward;
use crate::env::TickResets;
use crate::state::StateView;

/// A named, weighted reward.
pub struct Term {
    pub name: &'static str,
    pub weight: f64,
    pub reward: Box<dyn StepReward>,
}

/// TM1 team spirit: per term (unless `individual`), each car's weighted
/// value becomes `(1 - tau) r_i + tau * mean(r over its team)`, with `tau`
/// from the piecewise-linear `points` over the learner's cumulative steps
/// ([`crate::team::global_step`]). Skipped when no team has 2+ cars.
#[derive(Clone, Debug, Default)]
pub struct TeamSpirit {
    pub points: Vec<[f64; 2]>,
    /// Per term (term order): stays individual.
    pub individual: Vec<bool>,
    /// TM5, per term (term order): blended at tau 1.0 (shared outcomes such
    /// as goal / possession) instead of the schedule. Empty = none.
    pub full: Vec<bool>,
}

impl TeamSpirit {
    #[must_use]
    pub fn tau(&self) -> f64 {
        crate::team::piecewise(&self.points, crate::team::global_step() as f64).clamp(0.0, 1.0)
    }
}

/// Blends `r` toward each car's team mean (`team[i]` = car `i`'s team).
pub fn blend_team(r: &mut [f64], team: &[u8], tau: f64) {
    let mut sum = [0.0f64; 2];
    let mut n = [0usize; 2];
    for (x, &t) in r.iter().zip(team) {
        sum[usize::from(t.min(1))] += x;
        n[usize::from(t.min(1))] += 1;
    }
    for (x, &t) in r.iter_mut().zip(team) {
        let k = usize::from(t.min(1));
        if n[k] > 1 {
            let mean = sum[k] / n[k] as f64;
            *x = (1.0 - tau) * *x + tau * mean;
        }
    }
}

/// `CombinedReward`: `sum(weight * reward)` per car, in term order.
#[derive(Default)]
pub struct CombinedReward {
    pub terms: Vec<Term>,
    /// Weighted per-term contribution of the last step, for metrics.
    last: Vec<Vec<f64>>,
    /// Per setter kind, a weight multiplier per term (term order).
    setter_scales: Vec<(String, Vec<f64>)>,
    /// TM1 team spirit (None = off, the 1v1 behaviour).
    spirit: Option<TeamSpirit>,
    /// Last step's tau and mean |total change| per car from the blend.
    last_spirit: Option<(f64, f64)>,
    /// TM6 diagnostics: a role tracker and, per term, whether to report
    /// `RewardFirst/<term>` / `RewardOther/<term>` (mean weighted value per
    /// car-step of the first man / the other cars, teams of 2+ cars only).
    role_report: Option<(crate::team::RoleTracker, Vec<bool>)>,
    last_roles: Vec<crate::team::Role>,
}

impl CombinedReward {
    #[must_use]
    pub fn new(terms: Vec<Term>) -> Self {
        Self {
            terms,
            last: Vec::new(),
            setter_scales: Vec::new(),
            spirit: None,
            last_spirit: None,
            role_report: None,
            last_roles: Vec::new(),
        }
    }

    /// TM6: reports the listed terms split by role (see `role_report`).
    #[must_use]
    pub fn with_role_report(mut self, report: Option<(crate::team::RoleTracker, Vec<bool>)>) -> Self {
        self.role_report = report;
        self
    }

    /// Turns on team spirit (see [`TeamSpirit`]).
    #[must_use]
    pub fn with_team_spirit(mut self, spirit: Option<TeamSpirit>) -> Self {
        self.spirit = spirit;
        self
    }

    /// Adds per-setter term multipliers (see `RewardConfig::setter_scale`).
    #[must_use]
    pub fn with_setter_scales(mut self, scales: Vec<(String, Vec<f64>)>) -> Self {
        self.setter_scales = scales;
        self
    }

    pub fn reset_view(&mut self, initial: &StateView) {
        for t in &mut self.terms {
            t.reward.reset(initial);
        }
        if let Some((r, _)) = &mut self.role_report {
            r.reset();
        }
    }

    /// TM6: per reported term, (sum, count) of the first man's and the other
    /// cars' weighted values in the last step (teams of 2+ cars only).
    #[must_use]
    pub fn last_role_split(&self) -> Vec<(&'static str, [f64; 2], [usize; 2])> {
        let Some((_, flags)) = &self.role_report else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ((t, w), &on) in self.terms.iter().zip(&self.last).zip(flags) {
            if !on {
                continue;
            }
            let (mut sum, mut cnt) = ([0.0; 2], [0usize; 2]);
            for (i, x) in w.iter().enumerate() {
                match self.last_roles.get(i) {
                    Some(crate::team::Role::First) => {
                        sum[0] += x;
                        cnt[0] += 1;
                    }
                    Some(crate::team::Role::Second | crate::team::Role::Last) => {
                        sum[1] += x;
                        cnt[1] += 1;
                    }
                    // Demoed cars and 1-car teams (1v1) are not split.
                    Some(crate::team::Role::Out) | None => {}
                }
            }
            out.push((t.name, sum, cnt));
        }
        out
    }

    /// Total reward per car.
    pub fn step_view(&mut self, state: &StateView) -> Vec<f64> {
        self.step_view_for(state, "")
    }

    /// Total reward per car in an episode started by `setter`.
    pub fn step_view_for(&mut self, state: &StateView, setter: &str) -> Vec<f64> {
        let mut total = vec![0.0; state.cars.len()];
        self.last.clear();
        let scales = self
            .setter_scales
            .iter()
            .find(|(s, _)| s == setter)
            .map(|(_, v)| v.clone());
        let teams: Vec<u8> = state.cars.iter().map(|c| c.team).collect();
        let multi = crate::team::team_counts(state).iter().any(|&n| n > 1);
        let tau = match &self.spirit {
            Some(sp) if multi => Some(sp.tau()),
            _ => None,
        };
        self.last_roles.clear();
        if let Some((tracker, _)) = &mut self.role_report {
            let counts = crate::team::team_counts(state);
            let roles = tracker.update(state);
            // Only cars of 2+ car teams are split (1v1 has no roles to compare).
            if counts.iter().any(|&n| n > 1) {
                self.last_roles = roles.to_vec();
                for (r, c) in self.last_roles.iter_mut().zip(&state.cars) {
                    if counts[usize::from(c.team.min(1))] < 2 {
                        *r = crate::team::Role::Out;
                    }
                }
            }
        }
        let mut shift = 0.0;
        for (i, t) in self.terms.iter_mut().enumerate() {
            let r = t.reward.get_rewards(state);
            let weight = t.weight * scales.as_ref().map_or(1.0, |v| v[i]);
            let mut weighted: Vec<f64> = r.iter().map(|x| x * weight).collect();
            if let (Some(tau), Some(sp)) = (tau, &self.spirit) {
                if tau > 0.0 && !sp.individual.get(i).copied().unwrap_or(false) {
                    let before = weighted.clone();
                    let t = if sp.full.get(i).copied().unwrap_or(false) { 1.0 } else { tau };
                    blend_team(&mut weighted, &teams, t);
                    shift += before.iter().zip(&weighted).map(|(a, b)| (a - b).abs()).sum::<f64>();
                }
            }
            for (acc, w) in total.iter_mut().zip(&weighted) {
                *acc += w;
            }
            self.last.push(weighted);
        }
        self.last_spirit = tau.map(|t| (t, shift / state.cars.len().max(1) as f64));
        total
    }

    /// Weighted contribution of each term in the last step.
    #[must_use]
    pub fn last_terms(&self) -> &[Vec<f64>] {
        &self.last
    }
}

/// Per term, `Reward/<term>` is the mean weighted value per car-step (signed,
/// so zero-sum terms average ~0), `RewardAbs/<term>` the mean of its absolute
/// value and `RewardNonzero/<term>` the share of car-steps where it is non-zero.
impl<SI: SharedInfoReport + TickResets> Reward<SI> for CombinedReward {
    fn reset(&mut self, initial_state: &GameState, _shared_info: &mut SI) {
        self.reset_view(&StateView::from_game_state(initial_state));
    }

    fn get_rewards(&mut self, state: &GameState, shared_info: &mut SI) -> Vec<f32> {
        let view = StateView::from_game_state(state).with_tick_resets(shared_info.tick_resets());
        let total = self.step_view_for(&view, shared_info.setter_kind());
        let report = shared_info.report();
        for (t, w) in self.terms.iter().zip(&self.last) {
            let n = w.len().max(1) as f64;
            let mean = w.iter().sum::<f64>() / n;
            let abs = w.iter().map(|x| x.abs()).sum::<f64>() / n;
            let nonzero = w.iter().filter(|x| **x != 0.0).count() as f64 / n;
            report[format!("Reward/{}", t.name).as_str()] += AvgTracker::from(mean as f32);
            report[format!("RewardAbs/{}", t.name).as_str()] += AvgTracker::from(abs as f32);
            report[format!("RewardNonzero/{}", t.name).as_str()] +=
                AvgTracker::from(nonzero as f32);
        }
        for (name, sum, cnt) in self.last_role_split() {
            if cnt[0] > 0 {
                report[format!("RewardFirst/{name}").as_str()] += AvgTracker::from((sum[0] / cnt[0] as f64) as f32);
            }
            if cnt[1] > 0 {
                report[format!("RewardOther/{name}").as_str()] += AvgTracker::from((sum[1] / cnt[1] as f64) as f32);
            }
        }
        if let Some((tau, shift)) = self.last_spirit {
            report["Team/tau"] += AvgTracker::from(tau as f32);
            report["RewardAbs/team_spirit_shift"] += AvgTracker::from(shift as f32);
        }
        total.into_iter().map(|x| x as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_keeps_team_sums_and_skips_solo_teams() {
        let mut r = vec![3.0, 10.0, 1.0, -4.0];
        let teams = [0, 1, 0, 1];
        blend_team(&mut r, &teams, 0.5);
        assert_eq!(r, vec![2.5, 6.5, 1.5, -0.5]);
        assert!(((r[0] + r[2]) - 4.0).abs() < 1e-12);
        // tau = 1: everyone gets the team mean.
        let mut r = vec![3.0, 10.0, 1.0, -4.0];
        blend_team(&mut r, &teams, 1.0);
        assert_eq!(r, vec![2.0, 3.0, 2.0, 3.0]);
        // 1v1 (one car per team): untouched, bit for bit.
        let mut r = vec![0.1, 0.7];
        blend_team(&mut r, &[0, 1], 0.9);
        assert_eq!(r, vec![0.1, 0.7]);
    }

    struct Fixed(Vec<f64>);
    impl StepReward for Fixed {
        fn reset(&mut self, _initial: &StateView) {}
        fn get_rewards(&mut self, _state: &StateView) -> Vec<f64> {
            self.0.clone()
        }
    }

    /// TM5 `spirit_full`: flagged terms blend at tau 1.0, the others at the
    /// schedule's tau, individual terms not at all.
    #[test]
    fn spirit_full_blends_flagged_terms_at_one() {
        let raw = vec![3.0, 10.0, 1.0, -4.0];
        let state = StateView {
            cars: [0u8, 1, 0, 1]
                .iter()
                .map(|&t| crate::state::CarView { team: t, ..Default::default() })
                .collect(),
            ..Default::default()
        };
        let term = |name: &'static str, weight: f64| Term { name, weight, reward: Box::new(Fixed(raw.clone())) };
        let spirit = TeamSpirit {
            // Flat 0.5 at any learner step.
            points: vec![[0.0, 0.5], [1e18, 0.5]],
            individual: vec![false, false, true],
            full: vec![false, true, false],
        };
        let mut c = CombinedReward::new(vec![term("a", 1.0), term("b", 2.0), term("c", 1.0)]).with_team_spirit(Some(spirit));
        let r = c.step_view(&state);
        // a at 0.5: [2.5, 6.5, 1.5, -0.5]; b (x2) at 1.0: [4, 6, 4, 6]; c as is.
        let want = [2.5 + 4.0 + 3.0, 6.5 + 6.0 + 10.0, 1.5 + 4.0 + 1.0, -0.5 + 6.0 - 4.0];
        for (x, y) in r.iter().zip(want) {
            assert!((x - y).abs() < 1e-9, "{r:?} vs {want:?}");
        }
        // Without full flags b blends at 0.5 too.
        let spirit = TeamSpirit { points: vec![[0.0, 0.5], [1e18, 0.5]], individual: vec![false, false, true], full: Vec::new() };
        let mut c = CombinedReward::new(vec![term("a", 1.0), term("b", 2.0), term("c", 1.0)]).with_team_spirit(Some(spirit));
        let r = c.step_view(&state);
        assert!((r[0] - (2.5 + 5.0 + 3.0)).abs() < 1e-9, "{r:?}");
    }

    /// Blending after zero-sum = blending before: the opposing team's mean
    /// is unchanged by a blend within that team.
    #[test]
    fn blend_commutes_with_zero_sum() {
        let raw = [2.0, 5.0, -1.0, 3.0, 0.5, 7.0];
        let teams = [0u8, 1, 0, 1, 0, 1];
        let zs = |r: &[f64]| -> Vec<f64> {
            (0..r.len())
                .map(|i| {
                    let opp: Vec<f64> = (0..r.len()).filter(|&j| teams[j] != teams[i]).map(|j| r[j]).collect();
                    r[i] - opp.iter().sum::<f64>() / opp.len() as f64
                })
                .collect()
        };
        let mut a = zs(&raw);
        blend_team(&mut a, &teams, 0.6);
        let mut b = raw.to_vec();
        blend_team(&mut b, &teams, 0.6);
        let b = zs(&b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
    }
}
