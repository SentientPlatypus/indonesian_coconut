//! Training-only critic inputs (TM12). The actor, the observation, and the
//! exported `.bpk` stay unchanged. Default is off (`extra_inputs` empty).
//!
//! Supported names: `role_onehot`, `eta_rank`, `mate_roles`. The hysteretic
//! [`RoleTracker`] uses the same [`RoleParams`] as the rewards.

use std::sync::OnceLock;

use crate::state::StateView;
use crate::team::{Role, RoleParams, RoleTracker, eta_to_ball};

const ROLE_ONEHOT: usize = 3;
const ETA_RANK: usize = 1;
const MATE_ROLES: usize = 6; // 2 teammates * 3 one-hot
const MAX_MATES: usize = 2;

#[derive(Clone, Debug, Default)]
pub struct CriticExtraConfig {
    pub role_onehot: bool,
    pub eta_rank: bool,
    pub mate_roles: bool,
    pub roles: RoleParams,
}

impl CriticExtraConfig {
    pub fn from_names(names: &[String], roles: RoleParams) -> Result<Self, String> {
        let mut c = Self { roles, ..Self::default() };
        for n in names {
            match n.as_str() {
                "role_onehot" => c.role_onehot = true,
                "eta_rank" => c.eta_rank = true,
                "mate_roles" => c.mate_roles = true,
                other => {
                    return Err(format!(
                        "[critic].extra_inputs unknown {other:?}; expected role_onehot, eta_rank, mate_roles"
                    ));
                }
            }
        }
        Ok(c)
    }

    pub fn width(&self) -> usize {
        (if self.role_onehot { ROLE_ONEHOT } else { 0 })
            + (if self.eta_rank { ETA_RANK } else { 0 })
            + (if self.mate_roles { MATE_ROLES } else { 0 })
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0
    }
}

static CFG: OnceLock<CriticExtraConfig> = OnceLock::new();

/// Install the process-wide critic-extra config. Call once at trainer start.
/// Subsequent calls with the same width are ignored; a different width panics.
pub fn install(names: &[String], roles: RoleParams) {
    let cfg = CriticExtraConfig::from_names(names, roles).unwrap_or_else(|e| panic!("{e}"));
    match CFG.set(cfg.clone()) {
        Ok(()) => {
            if !cfg.is_empty() {
                println!(
                    "CRITIC extra_inputs {:?} width {}",
                    names, cfg.width()
                );
            }
        }
        Err(prev) => {
            assert_eq!(
                prev.width(),
                cfg.width(),
                "critic extra width changed mid-process ({} -> {})",
                prev.width(),
                cfg.width()
            );
        }
    }
}

pub fn config() -> CriticExtraConfig {
    CFG.get().cloned().unwrap_or_default()
}

pub fn width() -> usize {
    config().width()
}

fn role_onehot(role: Role) -> [f32; 3] {
    match role {
        Role::First => [1.0, 0.0, 0.0],
        Role::Second => [0.0, 1.0, 0.0],
        Role::Last => [0.0, 0.0, 1.0],
        Role::Out => [0.0, 0.0, 0.0],
    }
}

/// Per-car critic extras for `view`, advancing `tracker` once (same call
/// pattern as the rewards). Width is [`CriticExtraConfig::width`].
pub fn compute(tracker: &mut RoleTracker, view: &StateView) -> Vec<Vec<f32>> {
    let cfg = config();
    let n = view.cars.len();
    if cfg.is_empty() {
        return vec![Vec::new(); n];
    }
    let roles = tracker.update(view).to_vec();
    let mut out = vec![Vec::with_capacity(cfg.width()); n];
    let eta = if cfg.eta_rank {
        (0..n).map(|i| eta_to_ball(view, i)).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    for i in 0..n {
        if cfg.role_onehot {
            out[i].extend_from_slice(&role_onehot(roles[i]));
        }
        if cfg.eta_rank {
            // Rank among living teammates by ETA (0 = shortest). Demoed / alone = 0.
            // Normalized by 2 so 3v3 ranks sit in [0, 1].
            let team = view.cars[i].team;
            let mut ranks: Vec<usize> = (0..n)
                .filter(|&j| view.cars[j].team == team && !view.cars[j].is_demoed())
                .collect();
            ranks.sort_by(|&a, &b| eta[a].partial_cmp(&eta[b]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            let v = if view.cars[i].is_demoed() {
                0.0
            } else {
                ranks.iter().position(|&j| j == i).unwrap_or(0) as f32 / 2.0
            };
            out[i].push(v);
        }
        if cfg.mate_roles {
            let team = view.cars[i].team;
            let mates: Vec<usize> = (0..n).filter(|&j| j != i && view.cars[j].team == team).collect();
            for k in 0..MAX_MATES {
                let row = mates.get(k).map(|&j| role_onehot(roles[j])).unwrap_or([0.0; 3]);
                out[i].extend_from_slice(&row);
            }
        }
        debug_assert_eq!(out[i].len(), cfg.width());
    }
    out
}

/// Per-environment cache of the last computed extras, read by the collector.
#[derive(Clone, Debug, Default)]
pub struct ExtraState {
    tracker: Option<RoleTracker>,
    rows: Vec<Vec<f32>>,
}

impl ExtraState {
    pub fn reset(&mut self, view: &StateView) {
        let cfg = config();
        if cfg.is_empty() {
            self.tracker = None;
            self.rows.clear();
            return;
        }
        let mut t = RoleTracker::new(cfg.roles.clone());
        self.rows = compute(&mut t, view);
        self.tracker = Some(t);
    }

    pub fn update(&mut self, view: &StateView) {
        let cfg = config();
        if cfg.is_empty() {
            self.rows.clear();
            return;
        }
        let t = self.tracker.get_or_insert_with(|| RoleTracker::new(cfg.roles.clone()));
        self.rows = compute(t, view);
    }

    pub fn row(&self, car: usize) -> Vec<f32> {
        self.rows.get(car).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{BallView, CarView, StateView};
    use glam::DVec3;

    fn view() -> StateView {
        let car = |team: u8, y: f64| CarView {
            team,
            pos: DVec3::new(0.0, y, 17.0),
            vel: DVec3::new(0.0, 500.0, 0.0),
            ..CarView::default()
        };
        StateView {
            tick_count: 0,
            ball: BallView {
                pos: DVec3::new(10.0, 0.0, 93.0),
                vel: DVec3::ZERO,
                ..BallView::default()
            },
            // Blue: first at -300, second at -700, last at -3000.
            cars: vec![car(0, -300.0), car(0, -700.0), car(0, -3000.0), car(1, 3000.0)],
        }
    }

    #[test]
    fn empty_config_has_zero_width() {
        let c = CriticExtraConfig::from_names(&[], RoleParams::default()).unwrap();
        assert!(c.is_empty());
        assert_eq!(c.width(), 0);
    }

    #[test]
    fn all_three_are_ten_wide() {
        let c = CriticExtraConfig::from_names(
            &["role_onehot".into(), "eta_rank".into(), "mate_roles".into()],
            RoleParams::default(),
        )
        .unwrap();
        assert_eq!(c.width(), 10);
    }

    #[test]
    fn compute_role_onehot_and_mates() {
        // Use a local config without touching the process OnceLock.
        let mut tracker = RoleTracker::new(RoleParams {
            margin: 300.0,
            hold: 8,
            obs_tiebreak: true,
            touch_mode: "hysteresis".into(),
        });
        // Temporarily compute with a hand-rolled layout, independent of install().
        let roles = tracker.update(&view()).to_vec();
        assert_eq!(roles[0], Role::First);
        assert_eq!(roles[1], Role::Second);
        assert_eq!(roles[2], Role::Last);
        assert_eq!(role_onehot(roles[0]), [1.0, 0.0, 0.0]);
        assert_eq!(role_onehot(Role::Out), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn unknown_name_is_an_error() {
        let e = CriticExtraConfig::from_names(&["nope".into()], RoleParams::default()).unwrap_err();
        assert!(e.contains("nope"));
    }
}
