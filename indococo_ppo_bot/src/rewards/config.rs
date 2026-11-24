//! Reward weights and parameters, loaded from TOML (see `config/e8b2_1v1.toml`).
//! Field names follow the keys of the Python experiment JSON.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::aerial_boost_toward_ball::AerialBoostTowardBall;
use super::aerial_distance::{AerialDistance, AerialDistanceParams};
use super::aerial_front_bump_setup::AerialFrontBumpSetup;
use super::air_dribble::{AirDribble, AirDribbleParams};
use super::air_dribble_bump::{AirDribbleBump, AirDribbleBumpParams};
use super::air_dribble_sequence::{AirDribbleSeqParams, AirDribbleSequence};
use super::air_touch_height::{AirTouchHeight, AirTouchHeightParams};
use super::ang_vel::AngVel;
use super::ball_travel::BallTravel;
use super::boost_change::BoostChange;
use super::boost_keep::BoostKeep;
use super::clear_path_finish::ClearPathFinish;
use super::tm8::{CoverParams, KickoffSamePad, KickoffTaker, OpenNetGate, OpenNetParams, PhaseScale, ShadowCover, TermRamp};
use super::contact_quality::ContactQuality;
use super::contest_high_ball::ContestHighBall;
use super::demo::{Demo, DemoParams};
use super::energy::Energy;
use super::face_ball::FaceBall;
use super::flick::Flick;
use super::flip_reset::{FlipReset, FlipResetParams};
use super::flip_reset_gated::{FlipResetGated, FlipResetGatedParams};
use super::flip_reset_use::{FlipResetUse, FlipResetUseParams};
use super::musty::{Musty, MustyParams};
use super::goal::Goal;
use super::goal_dist::GoalDist;
use super::goal_prob::GoalProb;
use super::in_air::InAir;
use super::no_boost_overextend::NoBoostOverextend;
use super::one_v_one_recover::OneVOneRecover;
use super::opponent_possession_space::OpponentPossessionSpace;
use super::possession::Possession;
use super::possession_range_carry::{PossessionRangeCarry, TakeoffParams};
use super::playmaking_demo::{PlayKind, Playmaking, PlaymakingParams};
use super::tm11;
use super::pressure_flick::PressureFlick;
use super::reset_align::{ResetAlign, ResetAlignParams};
use super::shadow_box::{ShadowBox, ShadowBoxParams};
use super::safe_boost_collect::SafeBoostCollect;
use super::speed_toward_ball::SpeedTowardBall;
use super::touch::Touch;
use super::velocity_ball_to_goal::VelocityBallToGoal;
use super::wall_pop_setup::WallPopSetup;
use super::team_rewards::{
    AirDoubleTouch, ConcedeDist, ConcedeRole, ConcedeRoleParams, DoubleCommit, DoubleTap, DoubleTapParams, GateKind, KickoffMask, Pass, PassParams,
    TeamGate, TeammateFar, TeammateProximity, LastManGoalside, SpacingParams, SupportBand, SupportRole, TeammateSpacing,
};
use super::zero_sum::ZeroSum;
use crate::team::{CommitOrder, DoubleCommitParams, RoleParams, RoleTracker, TouchParams};
use super::{CombinedReward, StepReward, Term};

/// Reads `path` as a TOML table with its `extends` chain merged in.
fn load_extended(path: &Path, seen: &mut Vec<PathBuf>) -> Result<toml::Table, String> {
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if seen.contains(&canonical) {
        return Err(format!("extends cycle at {}", path.display()));
    }
    seen.push(canonical);
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut table: toml::Table = text
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let Some(parent) = table.remove("extends") else {
        return Ok(table);
    };
    let parent = parent
        .as_str()
        .ok_or_else(|| format!("{}: extends must be a path", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut base = load_extended(&dir.join(parent), seen)?;
    merge(&mut base, table);
    Ok(base)
}

/// Overlays `top` on `base`: nested tables merge, everything else replaces.
fn merge(base: &mut toml::Table, top: toml::Table) {
    for (key, value) in top {
        match (base.get_mut(&key), value) {
            (Some(toml::Value::Table(b)), toml::Value::Table(t)) => merge(b, t),
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

/// `reward_weights` of the experiment config.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardWeights {
    pub goal: f64,
    pub goal_prob: f64,
    pub ball_travel: f64,
    pub vel_ball_to_goal: f64,
    pub goal_dist: f64,
    pub speed_to_ball: f64,
    pub face_ball: f64,
    pub touch: f64,
    pub possession: f64,
    pub energy: f64,
    pub boost_keep: f64,
    pub boost_change: f64,
    pub aerial_boost: f64,
    pub aerial_distance: f64,
    pub in_air: f64,
    pub airdribble: f64,
    pub airdribble_seq: f64,
    pub wall_pop: f64,
    pub flick: f64,
    pub flip_reset: f64,
    pub recover: f64,
    pub demo: f64,
    pub contact_quality: f64,
    pub front_bump_setup: f64,
    pub overextend: f64,
    pub safe_boost: f64,
    pub shadow_space: f64,
    pub pressure_flick: f64,
    pub high_ball: f64,
    pub range_carry: f64,
    pub clear_path: f64,
    pub ang_vel: f64,
    // Experiment terms (built only when non-zero).
    pub flip_reset_gated: f64,
    pub air_touch_height: f64,
    pub air_dribble_bump: f64,
    pub flip_reset_use: f64,
    pub reset_align: f64,
    pub shadow_box: f64,
    // TM1 team terms (built only when non-zero; see `[team]`).
    pub pass: f64,
    pub double_tap: f64,
    pub air_double_touch: f64,
    pub concede_dist: f64,
    pub double_commit: f64,
    pub teammate_proximity: f64,
    /// TM2: air / post-reset musty flick (experiment term).
    pub musty: f64,
    /// TM5 spacing / role terms (`[team.spacing]`; experiment terms).
    pub teammate_spacing: f64,
    pub last_man_goalside: f64,
    pub support_band: f64,
    /// TM6: goal-side support band for the support role (`[team.roles]`).
    pub support_role: f64,
    /// TM7: too-far penalty for non-first cars (`[team.spacing]` far_*).
    pub teammate_far: f64,
    /// TM7: role-scaled, team-normalized concede penalty (`[team.concede_role]`).
    pub concede_role: f64,
    /// TM8: exactly-one-cover goal-side term for non-first cars (`[team.cover]`).
    pub shadow_cover: f64,
    /// TM8: kickoff-pending penalty, two non-takers at the same big pad.
    pub kickoff_same_pad: f64,
    /// TM8: kickoff-pending time penalty for the taker.
    pub kickoff_taker: f64,
    /// TM10: support-car demo / bump of the opponent in the play while
    /// attacking (`[playmaking]`; individual, not zero-sum).
    pub playmaking_demo: f64,
    pub playmaking_bump: f64,
    /// TM11: KRC challenge of an opponent grounded roof carry / the break event /
    /// 1.5 s kickoff outcome (all individual, not zero-sum).
    pub defend_carry: f64,
    pub carry_break: f64,
    pub kickoff_outcome: f64,
}

/// `[team]`: TM1 team-mode settings. All empty / unused by default, so
/// 1v1 configs build exactly the same reward.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TeamRewardConfig {
    /// Team spirit schedule `[[step, tau], ...]`, steps relative to
    /// `spirit_origin` (the run's start step; the team launcher fills it).
    /// Per car `r_i' = (1 - tau) r_i + tau mean(team r)` for every term not in
    /// `individual`, applied after the zero-sum wrappers (the same as before
    /// them: blending within a team keeps the opposing team's mean). Empty =
    /// off.
    pub spirit: Vec<[f64; 2]>,
    pub spirit_origin: f64,
    /// Terms that stay individual under team spirit.
    pub individual: Vec<String>,
    /// TM5: terms blended at tau 1.0 (when team spirit is on), the rest
    /// follow `spirit`. Must not overlap `individual`.
    pub spirit_full: Vec<String>,
    /// Terms paid only to each team's closest car to the ball (anti-ballchase).
    pub closest_gate: Vec<String>,
    /// Terms zeroed while the car is double committing.
    pub no_double_commit_gate: Vec<String>,
    /// Terms paid only to each team's last-back car (e.g. `recover`: a
    /// forward player upfield is rotating, not overextended).
    pub last_back_gate: Vec<String>,
    pub touch: TouchParams,
    pub pass: PassParams,
    pub double_tap: DoubleTapParams,
    pub double_commit: DoubleCommitParams,
    /// `concede_dist` length scale (Necto 2300).
    pub concede_scale: f64,
    /// `teammate_proximity` radius.
    pub proximity_radius: f64,
    /// TM5: terms zeroed only for the second committer (a double-committing
    /// car that is not its team's closest; see `GateKind::NotSecondCommit`).
    pub no_second_commit_gate: Vec<String>,
    /// TM5: `double_commit` term parameters when set (the gates keep
    /// `double_commit`), and charge only the non-closest committers.
    pub double_commit_term: Option<DoubleCommitParams>,
    pub double_commit_second_only: bool,
    /// TM5 `teammate_spacing` / `last_man_goalside` / `support_band`.
    pub spacing: SpacingParams,
    /// TM5r: when set, `double_commit` (second only) and the
    /// `no_second_commit_gate` use commit order ([`CommitOrder`]) with the
    /// `double_commit_term` test instead of "not the team's closest car"
    /// with the TM2 test. None = TM5 behaviour.
    pub commit_order: Option<CommitOrderParams>,
    /// TM6 `[team.roles]`: role tracking with hysteresis ([`RoleTracker`]).
    /// When set, `teammate_spacing` exempts the first man by role,
    /// `last_man_goalside` keys on the `Last` role, `support_role` uses it,
    /// and the `role_*` lists below apply. None = TM5r behaviour.
    pub roles: Option<RoleParams>,
    /// TM6: terms paid only to the first man (role gate (1, 0)).
    pub role_first_gate: Vec<String>,
    /// TM6: terms paid only to the non-first cars (role gate (0, 1)).
    pub role_not_first_gate: Vec<String>,
    /// TM6: `{term: scale}`: full for the first man, x scale for the others.
    pub role_scale: BTreeMap<String, f64>,
    /// TM6: terms zeroed during the kickoff phase (pending kickoff plus
    /// `kickoff_grace_s` after the first touch).
    pub kickoff_exempt: Vec<String>,
    pub kickoff_grace_s: f64,
    /// TM6 diagnostics: terms reported split by role
    /// (`RewardFirst/<term>`, `RewardOther/<term>`; needs `[team.roles]`).
    pub role_metrics: Vec<String>,
    /// TM7 `concede_role` scales and role lag.
    pub concede_role: ConcedeRoleParams,
    /// TM8 `shadow_cover` parameters.
    pub cover: CoverParams,
    /// TM8 `kickoff_same_pad` radius around a big pad (uu) and how long after
    /// the first kickoff touch it stays on (s).
    pub kickoff_pad_radius: f64,
    pub kickoff_pad_grace_s: f64,
    /// TM9: if set, `kickoff_scale` ([`PhaseScale`]) uses this grace (s)
    /// instead of `kickoff_grace_s`. `kickoff_exempt` still uses
    /// `kickoff_grace_s`. None = `kickoff_grace_s`.
    pub kickoff_scale_grace_s: Option<f64>,
    /// TM9: at most one non-taker may take or head to an own-corner big pad
    /// during the kickoff phase (2v2 is a no-op).
    pub kickoff_one_corner: bool,
    /// TM8: `{term: scale}` multipliers during the kickoff phase (pending +
    /// `kickoff_grace_s`, or `kickoff_scale_grace_s` when set); e.g. energy /
    /// boost_keep 0.25.
    pub kickoff_scale: BTreeMap<String, f64>,
    /// TM8: `{term: [[step, scale], ...]}` per-term schedules relative to
    /// `spirit_origin` (e.g. contact_quality 1.0 -> 0.4 over 200M).
    pub term_ramp: BTreeMap<String, Vec<[f64; 2]>>,
}

/// `[team.commit_order]` (TM5r).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CommitOrderParams {
    /// Steps the gate keeps a second committer gated after its double
    /// commit ends (the touch step's post-contact velocity often no longer
    /// "closes" on the ball). The penalty term has no hold.
    pub hold: u32,
}

impl Default for CommitOrderParams {
    fn default() -> Self {
        Self { hold: 8 }
    }
}

impl Default for TeamRewardConfig {
    fn default() -> Self {
        Self {
            spirit: Vec::new(),
            spirit_origin: 0.0,
            individual: Vec::new(),
            spirit_full: Vec::new(),
            closest_gate: Vec::new(),
            no_double_commit_gate: Vec::new(),
            last_back_gate: Vec::new(),
            touch: TouchParams::default(),
            pass: PassParams::default(),
            double_tap: DoubleTapParams::default(),
            double_commit: DoubleCommitParams::default(),
            concede_scale: 2300.0,
            proximity_radius: 800.0,
            no_second_commit_gate: Vec::new(),
            double_commit_term: None,
            double_commit_second_only: false,
            spacing: SpacingParams::default(),
            commit_order: None,
            roles: None,
            role_first_gate: Vec::new(),
            role_not_first_gate: Vec::new(),
            role_scale: BTreeMap::new(),
            kickoff_exempt: Vec::new(),
            kickoff_grace_s: 1.0,
            role_metrics: Vec::new(),
            concede_role: ConcedeRoleParams::default(),
            cover: CoverParams::default(),
            kickoff_pad_radius: 1500.0,
            kickoff_pad_grace_s: 3.0,
            kickoff_scale_grace_s: None,
            kickoff_one_corner: false,
            kickoff_scale: BTreeMap::new(),
            term_ramp: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ContactQualityParams {
    pub hard_target: f64,
    pub wheel_penalty: f64,
}

impl Default for ContactQualityParams {
    fn default() -> Self {
        Self {
            hard_target: 900.0,
            wheel_penalty: 0.3,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SafeBoostParams {
    pub target: f64,
    pub min_ball_dist: f64,
}

impl Default for SafeBoostParams {
    fn default() -> Self {
        Self {
            target: 80.0,
            min_ball_dist: 1400.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ClearPathParams {
    pub min_lead: f64,
    pub lane_radius: f64,
    /// Minimum goalward ball speed after the touch (E8B2 600; TM8 1500).
    pub min_goalward: f64,
    /// TM8: ball within this distance of the attacked net centre (0 = any).
    pub max_dist: f64,
    /// TM8: pay at most once per team possession (until an opponent touch).
    pub once_per_possession: bool,
    /// TM8: any step with a CarHitBall counts as a touch (see airdribble_seq).
    pub any_touch: bool,
    /// TM8: extra `goal_bonus x payout` if the payer's team scores within
    /// `goal_window_s` of the paid touch.
    pub goal_bonus: f64,
    pub goal_window_s: f64,
}

impl Default for ClearPathParams {
    fn default() -> Self {
        Self {
            min_lead: 900.0,
            lane_radius: 1100.0,
            min_goalward: 600.0,
            max_dist: 0.0,
            once_per_possession: false,
            any_touch: false,
            goal_bonus: 0.0,
            goal_window_s: 2.0,
        }
    }
}

/// Full reward configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardConfig {
    pub weights: RewardWeights,
    /// `{weight_key: opp_scale}`; listed terms are wrapped in [`ZeroSum`].
    pub zero_sum: BTreeMap<String, f64>,
    pub airdribble: AirDribbleParams,
    pub airdribble_seq: AirDribbleSeqParams,
    pub aerial_distance: AerialDistanceParams,
    pub takeoff: TakeoffParams,
    pub flip_reset: FlipResetParams,
    pub demo: DemoParams,
    pub contact_quality: ContactQualityParams,
    pub safe_boost: SafeBoostParams,
    pub clear_path: ClearPathParams,
    pub flip_reset_gated: FlipResetGatedParams,
    pub air_touch_height: AirTouchHeightParams,
    pub air_dribble_bump: AirDribbleBumpParams,
    pub flip_reset_use: FlipResetUseParams,
    pub reset_align: ResetAlignParams,
    pub shadow_box: ShadowBoxParams,
    pub musty: MustyParams,
    /// TM8 open-net gate on air-dribble shaping (`[open_net]`; empty = off).
    pub open_net: OpenNetParams,
    /// TM10 `playmaking_demo` / `playmaking_bump` parameters.
    pub playmaking: PlaymakingParams,
    /// TM11 defend-carry / kickoff-outcome parameters.
    #[serde(default)]
    pub defend_carry: crate::rewards::tm11::DefendCarryParams,
    #[serde(default)]
    pub kickoff_outcome: crate::rewards::tm11::KickoffOutcomeParams,
    /// TM1 team settings (team spirit, gates, pass / double-touch params).
    pub team: TeamRewardConfig,
    /// `{setter kind: {term: scale}}`: term weights multiplied by `scale` in
    /// episodes started by that setter (e.g. no air dribble reward in
    /// `reset_curriculum` episodes). Unknown term names panic in `build`.
    pub setter_scale: BTreeMap<String, BTreeMap<String, f64>>,
}

impl RewardConfig {
    /// Parses a TOML config.
    ///
    /// # Errors
    /// Invalid TOML or unknown keys.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Loads a config file. A file may start from another one with
    /// `extends = "base.toml"` (relative to its own directory) and override
    /// any keys: tables merge key by key, other values replace. An experiment
    /// file then holds only what it changes, e.g. one new weight and its
    /// parameters on top of `e8b2_1v1.toml`.
    ///
    /// # Errors
    /// Unreadable file, invalid TOML, an `extends` cycle, or unknown keys.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        load_extended(path, &mut Vec::new())?
            .try_into()
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Builds the combined reward in `v4_env._reward_fn` order. Terms with
    /// weight 0 are still evaluated, like the Python `CombinedReward`, so
    /// parity holds term by term.
    #[must_use]
    pub fn build(&self) -> CombinedReward {
        let w = &self.weights;
        // TM1 gates go inside the zero-sum wrapper (gate the raw term);
        // terms without zero-sum are gated in the final mapping below.
        let tm = &self.team;
        // TM5: the spacing ramp is relative to the run start (spirit_origin).
        // Only the TM5 double_commit (`double_commit_second_only`) uses it.
        // TM5r: per-term overrides (`spacing_ramp`, `double_commit_ramp`,
        // `role_ramp`) fall back to `ramp`.
        let abs = |r: &[[f64; 2]]| -> Vec<[f64; 2]> { r.iter().map(|&[st, v]| [st + tm.spirit_origin, v]).collect() };
        let pick = |o: &Option<Vec<[f64; 2]>>| abs(o.as_deref().unwrap_or(&tm.spacing.ramp));
        let spacing = SpacingParams { ramp: pick(&tm.spacing.spacing_ramp), ..tm.spacing.clone() };
        let roles = SpacingParams { ramp: pick(&tm.spacing.role_ramp), ..tm.spacing.clone() };
        // TM6: last_man_goalside / support_role ramps fall back to role_ramp.
        let pick_role = |o: &Option<Vec<[f64; 2]>>| -> Vec<[f64; 2]> {
            match o {
                Some(r) => abs(r),
                None => roles.ramp.clone(),
            }
        };
        let last_p = SpacingParams { ramp: pick_role(&tm.spacing.last_ramp), ..tm.spacing.clone() };
        let support_p = SpacingParams { ramp: pick_role(&tm.spacing.support_ramp), ..tm.spacing.clone() };
        let tracker = || tm.roles.clone().map(RoleTracker::new);
        let role_p = tm.roles.clone().unwrap_or_default();
        let spacing_ramp = if tm.double_commit_second_only { pick(&tm.spacing.double_commit_ramp) } else { Vec::new() };
        // TM5r: commit order on the double_commit_term test (fallback TM2's).
        let order_dc = tm.double_commit_term.clone().unwrap_or_else(|| tm.double_commit.clone());
        let order = || tm.commit_order.as_ref().map(|o| CommitOrder::new(order_dc.clone(), o.hold));
        let gate = |key: &str, r: Box<dyn StepReward>| -> Box<dyn StepReward> {
            let mut r = r;
            if tm.closest_gate.iter().any(|k| k == key) {
                r = Box::new(TeamGate::new(r, GateKind::Closest, tm.double_commit.clone()));
            }
            if tm.no_double_commit_gate.iter().any(|k| k == key) {
                r = Box::new(TeamGate::new(r, GateKind::NotDoubleCommit, tm.double_commit.clone()));
            }
            if tm.last_back_gate.iter().any(|k| k == key) {
                r = Box::new(TeamGate::new(r, GateKind::LastBack, tm.double_commit.clone()));
            }
            if tm.no_second_commit_gate.iter().any(|k| k == key) {
                r = Box::new(TeamGate::new(r, GateKind::NotSecondCommit, tm.double_commit.clone()).with_commit_order(order()));
            }
            if tm.role_first_gate.iter().any(|k| k == key) {
                r = Box::new(TeamGate::role(r, role_p.clone(), 1.0, 0.0));
            }
            if tm.role_not_first_gate.iter().any(|k| k == key) {
                r = Box::new(TeamGate::role(r, role_p.clone(), 0.0, 1.0));
            }
            if let Some(&s) = tm.role_scale.get(key) {
                r = Box::new(TeamGate::role(r, role_p.clone(), 1.0, s));
            }
            r
        };
        let wrapped = std::cell::RefCell::new(Vec::<String>::new());
        let zs = |key: &str, r: Box<dyn StepReward>| -> Box<dyn StepReward> {
            match self.zero_sum.get(key) {
                Some(&s) if s > 0.0 => {
                    wrapped.borrow_mut().push(key.to_owned());
                    Box::new(ZeroSum::new(gate(key, r), s))
                }
                _ => r,
            }
        };
        let t = self.takeoff.clone();
        let mut terms: Vec<(&'static str, f64, Box<dyn StepReward>)> = vec![
            ("goal", w.goal, Box::new(Goal)),
            ("goal_prob", w.goal_prob, Box::new(GoalProb::default())),
            (
                "ball_travel",
                w.ball_travel,
                Box::new(BallTravel::default()),
            ),
            (
                "vel_ball_to_goal",
                w.vel_ball_to_goal,
                zs("vel_ball_to_goal", Box::new(VelocityBallToGoal)),
            ),
            (
                "goal_dist",
                w.goal_dist,
                zs("goal_dist", Box::new(GoalDist)),
            ),
            (
                "speed_to_ball",
                w.speed_to_ball,
                zs("speed_to_ball", Box::new(SpeedTowardBall)),
            ),
            (
                "face_ball",
                w.face_ball,
                zs("face_ball", Box::new(FaceBall)),
            ),
            ("touch", w.touch, zs("touch", Box::new(Touch))),
            ("possession", w.possession, Box::new(Possession::default())),
            ("energy", w.energy, zs("energy", Box::new(Energy))),
            ("boost_keep", w.boost_keep, Box::new(BoostKeep::default())),
            (
                "boost_change",
                w.boost_change,
                Box::new(BoostChange::new(1.0, 0.8)),
            ),
            (
                "aerial_boost",
                w.aerial_boost,
                Box::new(AerialBoostTowardBall::new(0.96)),
            ),
            (
                "aerial_distance",
                w.aerial_distance,
                Box::new(AerialDistance::new(&self.aerial_distance)),
            ),
            ("in_air", w.in_air, Box::new(InAir)),
            (
                "airdribble",
                w.airdribble,
                Box::new(AirDribble::new(self.airdribble.clone(), t.clone())),
            ),
            (
                "airdribble_seq",
                w.airdribble_seq,
                Box::new(AirDribbleSequence::with_params(
                    t.clone(),
                    self.airdribble_seq.clone(),
                )),
            ),
            ("wall_pop", w.wall_pop, Box::new(WallPopSetup::default())),
            ("flick", w.flick, Box::new(Flick::default())),
            (
                "flip_reset",
                w.flip_reset,
                zs(
                    "flip_reset",
                    Box::new(FlipReset::new(self.flip_reset.clone())),
                ),
            ),
            ("recover", w.recover, Box::new(OneVOneRecover::default())),
            ("demo", w.demo, Box::new(Demo::new(self.demo.clone()))),
            (
                "contact_quality",
                w.contact_quality,
                Box::new(ContactQuality::new(
                    self.contact_quality.hard_target,
                    self.contact_quality.wheel_penalty,
                )),
            ),
            (
                "front_bump_setup",
                w.front_bump_setup,
                Box::new(AerialFrontBumpSetup::default()),
            ),
            (
                "overextend",
                w.overextend,
                Box::new(NoBoostOverextend::default()),
            ),
            (
                "safe_boost",
                w.safe_boost,
                Box::new(SafeBoostCollect::new(
                    self.safe_boost.target,
                    self.safe_boost.min_ball_dist,
                )),
            ),
            (
                "shadow_space",
                w.shadow_space,
                Box::new(OpponentPossessionSpace::default()),
            ),
            (
                "pressure_flick",
                w.pressure_flick,
                Box::new(PressureFlick::default()),
            ),
            (
                "high_ball",
                w.high_ball,
                Box::new(ContestHighBall::default()),
            ),
            (
                "range_carry",
                w.range_carry,
                Box::new(PossessionRangeCarry::new(t)),
            ),
            (
                "clear_path",
                w.clear_path,
                Box::new(ClearPathFinish::with_params(&self.clear_path)),
            ),
            ("ang_vel", w.ang_vel, Box::new(AngVel::default())),
        ];
        let experiment: [(&'static str, f64, Box<dyn StepReward>); 27] = [
            (
                "flip_reset_gated",
                w.flip_reset_gated,
                zs(
                    "flip_reset_gated",
                    Box::new(FlipResetGated::new(self.flip_reset_gated.clone())),
                ),
            ),
            (
                "air_touch_height",
                w.air_touch_height,
                Box::new(AirTouchHeight::new(self.air_touch_height.clone())),
            ),
            (
                "air_dribble_bump",
                w.air_dribble_bump,
                Box::new(AirDribbleBump::new(self.air_dribble_bump.clone())),
            ),
            (
                "flip_reset_use",
                w.flip_reset_use,
                Box::new(FlipResetUse::new(self.flip_reset_use.clone())),
            ),
            (
                "reset_align",
                w.reset_align,
                Box::new(ResetAlign::new(self.reset_align.clone())),
            ),
            (
                "shadow_box",
                w.shadow_box,
                Box::new(ShadowBox::new(self.shadow_box.clone())),
            ),
            (
                "pass",
                w.pass,
                Box::new(Pass::new(tm.touch.clone(), tm.pass.clone())),
            ),
            (
                "double_tap",
                w.double_tap,
                Box::new(DoubleTap::new(tm.touch.clone(), tm.double_tap.clone())),
            ),
            (
                "air_double_touch",
                w.air_double_touch,
                zs("air_double_touch", Box::new(AirDoubleTouch::new(tm.touch.clone()))),
            ),
            ("concede_dist", w.concede_dist, Box::new(ConcedeDist::new(tm.concede_scale))),
            (
                "double_commit",
                w.double_commit,
                Box::new(
                    DoubleCommit::new(tm.double_commit_term.clone().unwrap_or_else(|| tm.double_commit.clone()))
                        .second_only(tm.double_commit_second_only, spacing_ramp.clone())
                        .with_commit_order(order()),
                ),
            ),
            (
                "teammate_proximity",
                w.teammate_proximity,
                Box::new(TeammateProximity::new(tm.proximity_radius)),
            ),
            ("musty", w.musty, Box::new(Musty::new(self.musty.clone()))),
            ("teammate_spacing", w.teammate_spacing, Box::new(TeammateSpacing::new(spacing.clone()).with_roles(tracker()))),
            ("last_man_goalside", w.last_man_goalside, Box::new(LastManGoalside::new(last_p.clone()).with_roles(tracker()))),
            ("support_band", w.support_band, Box::new(SupportBand::new(roles.clone()))),
            ("support_role", w.support_role, Box::new(SupportRole::new(support_p.clone(), role_p.clone()))),
            ("teammate_far", w.teammate_far, Box::new(TeammateFar::new(tm.spacing.clone(), role_p.clone()))),
            ("concede_role", w.concede_role, Box::new(ConcedeRole::new(tm.concede_role.clone(), role_p.clone()))),
            ("shadow_cover", w.shadow_cover, Box::new(ShadowCover::new(tm.cover.clone(), role_p.clone()))),
            ("kickoff_same_pad", w.kickoff_same_pad, Box::new(KickoffSamePad::new_with(tm.kickoff_pad_radius, role_p.clone(), (tm.kickoff_pad_grace_s * crate::consts::STEPS_PER_SECOND).round().max(0.0) as u32, tm.kickoff_one_corner))),
            ("kickoff_taker", w.kickoff_taker, Box::new(KickoffTaker::new(role_p.clone()))),
            ("playmaking_demo", w.playmaking_demo, Box::new(Playmaking::new(self.playmaking.clone(), role_p.clone(), PlayKind::Demo))),
            ("playmaking_bump", w.playmaking_bump, Box::new(Playmaking::new(self.playmaking.clone(), role_p.clone(), PlayKind::Bump))),
            ("defend_carry", w.defend_carry, Box::new(tm11::DefendCarry::new(self.defend_carry.clone(), role_p.clone(), tm11::CarryPart::Shape))),
            ("carry_break", w.carry_break, Box::new(tm11::DefendCarry::new(self.defend_carry.clone(), role_p.clone(), tm11::CarryPart::Break))),
            ("kickoff_outcome", w.kickoff_outcome, Box::new(tm11::KickoffOutcome::new(self.kickoff_outcome.clone(), role_p.clone()))),
        ];
        terms.extend(
            experiment
                .into_iter()
                .filter(|(_, weight, _)| *weight != 0.0),
        );
        if w.concede_role != 0.0 {
            assert!(tm.individual.iter().any(|k| k == "concede_role"), "concede_role must be individual (post-mix)");
            assert!(!self.zero_sum.contains_key("concede_role"), "concede_role is a penalty, not zero-sum");
        }
        for (name, weight) in [("defend_carry", w.defend_carry), ("carry_break", w.carry_break), ("kickoff_outcome", w.kickoff_outcome)] {
            if weight != 0.0 {
                assert!(tm.individual.iter().any(|k| k == name), "{name} must be in team.individual");
                assert!(!self.zero_sum.contains_key(name), "{name} must not be in any zero-sum list");
            }
        }
        let role_scale_keys: Vec<String> = tm.role_scale.keys().cloned().collect();
        let ko_scale_keys: Vec<String> = tm.kickoff_scale.keys().cloned().collect();
        let ramp_keys: Vec<String> = tm.term_ramp.keys().cloned().collect();
        for list in [
            &tm.individual, &tm.closest_gate, &tm.no_double_commit_gate, &tm.last_back_gate, &tm.no_second_commit_gate, &tm.spirit_full,
            &tm.role_first_gate, &tm.role_not_first_gate, &role_scale_keys, &tm.kickoff_exempt, &tm.role_metrics,
            &ko_scale_keys, &ramp_keys, &self.open_net.terms,
        ] {
            for key in list {
                assert!(
                    terms.iter().any(|(n, _, _)| n == key),
                    "[team]: unknown or unweighted term {key}"
                );
            }
        }
        let grace = (tm.kickoff_grace_s * crate::consts::STEPS_PER_SECOND).round().max(0.0) as u32;
        let scale_grace = (tm.kickoff_scale_grace_s.unwrap_or(tm.kickoff_grace_s) * crate::consts::STEPS_PER_SECOND).round().max(0.0) as u32;
        let terms: Vec<Term> = terms
            .into_iter()
            .map(|(name, weight, reward)| {
                // Gate terms that have no zero-sum wrapper here (zs gated the others).
                let mut reward = if wrapped.borrow().iter().any(|k| k == name) { reward } else { gate(name, reward) };
                if tm.kickoff_exempt.iter().any(|k| k == name) {
                    reward = Box::new(KickoffMask::new(reward, grace));
                }
                // TM8 wrappers (all off unless configured).
                if self.open_net.terms.iter().any(|k| k == name) {
                    reward = Box::new(OpenNetGate::new(reward, self.open_net.clone()));
                }
                if let Some(&s) = tm.kickoff_scale.get(name) {
                    reward = Box::new(PhaseScale::new(reward, scale_grace, s));
                }
                if let Some(pts) = tm.term_ramp.get(name) {
                    reward = Box::new(TermRamp::new(reward, abs(pts)));
                }
                Term { name, weight, reward }
            })
            .collect();
        let scales = self
            .setter_scale
            .iter()
            .map(|(setter, by_term)| {
                for term in by_term.keys() {
                    assert!(
                        terms.iter().any(|t| t.name == term),
                        "setter_scale.{setter}: unknown or unweighted term {term}"
                    );
                }
                let per_term = terms
                    .iter()
                    .map(|t| by_term.get(t.name).copied().unwrap_or(1.0))
                    .collect();
                (setter.clone(), per_term)
            })
            .collect();
        let spirit = (!tm.spirit.is_empty()).then(|| super::combined::TeamSpirit {
            points: tm
                .spirit
                .iter()
                .map(|&[step, tau]| [step + tm.spirit_origin, tau])
                .collect(),
            individual: terms
                .iter()
                .map(|t| tm.individual.iter().any(|k| k == t.name))
                .collect(),
            full: terms
                .iter()
                .map(|t| tm.spirit_full.iter().any(|k| k == t.name))
                .collect(),
        });
        let role_report = (!tm.role_metrics.is_empty())
            .then(|| tm.roles.clone())
            .flatten()
            .map(|p| (RoleTracker::new(p), terms.iter().map(|t| tm.role_metrics.iter().any(|k| k == t.name)).collect()));
        CombinedReward::new(terms)
            .with_setter_scales(scales)
            .with_team_spirit(spirit)
            .with_role_report(role_report)
    }
}

#[cfg(test)]
mod extends_tests {
    use super::*;

    #[test]
    fn fr4_is_aud_plus_the_fr3_reset_setup() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let fr4 = RewardConfig::from_file(&dir.join("fr4.toml")).unwrap();
        let fr3 = RewardConfig::from_file(&dir.join("fr3.toml")).unwrap();
        let aud = RewardConfig::from_file(&dir.join("aud.toml")).unwrap();
        let w = &fr4.weights;
        assert_eq!(
            (
                w.airdribble_seq,
                w.airdribble,
                w.aerial_distance,
                w.air_touch_height
            ),
            (50.0, 50.0, 38.0, 0.0)
        );
        assert_eq!(
            (w.flip_reset, w.flip_reset_use, w.reset_align),
            (0.0, 600.0, 400.0)
        );
        assert_eq!(fr4.airdribble_seq, aud.airdribble_seq);
        assert_eq!(fr4.aerial_distance, aud.aerial_distance);
        let mut want = fr3.setter_scale.clone();
        want.get_mut("reset_curriculum")
            .unwrap()
            .remove("air_touch_height");
        assert_eq!(fr4.setter_scale, want);
        assert_eq!(
            format!("{:?}", fr4.flip_reset_use),
            format!("{:?}", fr3.flip_reset_use)
        );
        assert_eq!(
            format!("{:?}", fr4.reset_align),
            format!("{:?}", fr3.reset_align)
        );
        let _ = fr4.build();
    }

    #[test]
    fn fr7c_renders_and_builds() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let text = std::fs::read_to_string(dir.join("fr7c.toml"))
            .unwrap()
            .replace("@NORMAL@", "2.0")
            .replace("@DRILL@", "3.0");
        let live = dir.join(format!("fr7c_test_{}.toml", std::process::id()));
        std::fs::write(&live, text).unwrap();
        let cfg = RewardConfig::from_file(&live);
        std::fs::remove_file(&live).unwrap();
        let cfg = cfg.unwrap();
        assert_eq!(cfg.weights.air_touch_height, 8.0);
        assert_eq!(cfg.weights.flip_reset_use, 600.0);
        assert_eq!(cfg.air_touch_height.wall_margin, 300.0);
        assert_eq!(cfg.flip_reset_use.goal_bonus, 3.0);
        assert_eq!(cfg.setter_scale["air_reset"]["flip_reset_use"], 3.0);
        assert_eq!(cfg.setter_scale["kickoff"]["flip_reset_use"], 2.0);
        // fr4's drill zeroing survives the extends chain.
        assert_eq!(cfg.setter_scale["reset_curriculum"]["airdribble"], 0.0);
        let _ = cfg.build();
    }

    #[test]
    fn aud_reworks_air_terms_and_adb_keeps_e8b2_behaviour() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let adb = RewardConfig::from_file(&dir.join("adb.toml")).unwrap();
        assert_eq!(adb.airdribble_seq, AirDribbleSeqParams::default());
        assert_eq!(adb.aerial_distance, AerialDistanceParams::default());
        assert_eq!(adb.airdribble.require_touch_s, 0.0);
        assert_eq!(adb.weights.airdribble_seq, 55.0);
        let aud = RewardConfig::from_file(&dir.join("aud.toml")).unwrap();
        let w = &aud.weights;
        assert_eq!(
            (
                w.air_touch_height,
                w.airdribble_seq,
                w.airdribble,
                w.aerial_distance
            ),
            (0.0, 50.0, 50.0, 38.0)
        );
        assert_eq!(w.air_dribble_bump, 40.0);
        assert_eq!(aud.airdribble_seq.max_chain_touches, 6);
        assert!(aud.aerial_distance.goalward_ball_only);
        assert_eq!(aud.airdribble.push_floor, 0.0);
        let _ = aud.build();
    }

    #[test]
    fn experiment_file_overrides_its_base() {
        let dir = std::env::temp_dir().join(format!("indococo_extends_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("base.toml"),
            "[weights]\ngoal = 10.0\ntouch = 2.0\n[flip_reset]\nmin_wheels_cos = 0.5\nuse_weight = 3.0\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("exp.toml"),
            "extends = \"base.toml\"\n[weights]\ntouch = 5.0\nflip_reset = 70.0\n[flip_reset]\nmin_wheels_cos = 0.7\n",
        )
        .unwrap();
        let cfg = RewardConfig::from_file(&dir.join("exp.toml")).unwrap();
        assert_eq!(cfg.weights.goal, 10.0);
        assert_eq!(cfg.weights.touch, 5.0);
        assert_eq!(cfg.weights.flip_reset, 70.0);
        assert_eq!(cfg.flip_reset.min_wheels_cos, 0.7);
        assert_eq!(cfg.flip_reset.use_weight, 3.0);

        std::fs::write(dir.join("a.toml"), "extends = \"b.toml\"\n").unwrap();
        std::fs::write(dir.join("b.toml"), "extends = \"a.toml\"\n").unwrap();
        assert!(
            RewardConfig::from_file(&dir.join("a.toml"))
                .unwrap_err()
                .contains("cycle")
        );
        std::fs::write(
            dir.join("bad.toml"),
            "extends = \"base.toml\"\n[weights]\nnope = 1.0\n",
        )
        .unwrap();
        assert!(RewardConfig::from_file(&dir.join("bad.toml")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn st36_adds_shadow_box_on_st_toml() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let st = RewardConfig::from_file(&dir.join("st.toml")).unwrap();
        assert_eq!(st.weights.shadow_box, 0.0);
        let names: Vec<_> = st.build().terms.iter().map(|t| t.name).collect();
        assert!(!names.contains(&"shadow_box"), "{names:?}");

        let st36 = RewardConfig::from_file(&dir.join("st36.toml")).unwrap();
        assert_eq!(st36.weights.shadow_box, 30.0);
        assert_eq!(st36.weights.face_ball, 1.0);
        assert_eq!(st36.weights.shadow_space, 18.0);
        assert_eq!(st36.weights.recover, 70.0);
        assert!((st36.shadow_box.gamma - 0.998).abs() < 1e-12);
        assert!(st36.zero_sum.get("shadow_box").is_none() || st36.zero_sum["shadow_box"] == 0.0);
        let built = st36.build();
        let term = built
            .terms
            .iter()
            .find(|t| t.name == "shadow_box")
            .expect("shadow_box term");
        assert_eq!(term.weight, 30.0);
        assert_eq!(
            built.terms.iter().find(|t| t.name == "face_ball").unwrap().weight,
            1.0
        );
    }

    #[test]
    fn tm1_adds_team_terms_on_st36() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let st36 = RewardConfig::from_file(&dir.join("st36.toml")).unwrap();
        assert!(st36.team.spirit.is_empty() && st36.team.closest_gate.is_empty());
        assert!(!st36.shadow_box.team_last_back);
        let names: Vec<_> = st36.build().terms.iter().map(|t| t.name).collect();
        for t in ["pass", "double_tap", "air_double_touch", "concede_dist", "double_commit", "teammate_proximity"] {
            assert!(!names.contains(&t), "{t} in st36");
        }
        let tm1 = RewardConfig::from_file(&dir.join("tm1.toml")).unwrap();
        assert_eq!(tm1.weights.flip_reset, 10.0);
        assert_eq!(tm1.weights.pass, 40.0);
        assert_eq!(tm1.weights.double_tap, 400.0);
        assert_eq!(tm1.weights.goal, 1200.0);
        assert!(tm1.shadow_box.team_last_back);
        assert_eq!(tm1.team.spirit.len(), 3);
        let built = tm1.build();
        let names: Vec<_> = built.terms.iter().map(|t| t.name).collect();
        for t in ["pass", "double_tap", "air_double_touch", "concede_dist", "double_commit", "shadow_box"] {
            assert!(names.contains(&t), "{t} missing from tm1");
        }
        assert!(!names.contains(&"teammate_proximity"));
        let fr = RewardConfig::from_file(&dir.join("tm1_fr.toml")).unwrap();
        assert_eq!(fr.weights.flip_reset_use, 30.0);
        assert_eq!(fr.flip_reset_use.goal_bonus, 8.0);
    }

    #[test]
    fn tm5_config_adds_self_only_spacing_terms() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let tm2 = RewardConfig::from_file(&dir.join("tm2.toml")).unwrap();
        let n2: Vec<_> = tm2.build().terms.iter().map(|t| t.name).collect();
        for t in ["teammate_spacing", "last_man_goalside", "support_band"] {
            assert!(!n2.contains(&t), "{t} in tm2");
        }
        assert!(tm2.team.no_second_commit_gate.is_empty() && tm2.team.double_commit_term.is_none());
        assert!(!tm2.team.double_commit_second_only);
        let tm5 = RewardConfig::from_file(&dir.join("tm5.toml")).unwrap();
        assert_eq!(tm5.weights.teammate_spacing, 0.5);
        assert_eq!(tm5.weights.double_commit, 2.0);
        assert_eq!(tm5.weights.concede_dist, 0.0);
        assert_eq!(tm5.weights.goal, 1200.0);
        assert_eq!(tm5.team.double_commit_term.as_ref().unwrap().radius, 1000.0);
        assert_eq!(tm5.team.double_commit.radius, 600.0, "gates keep the TM2 test");
        assert_eq!(tm5.team.spirit, vec![[0.0, 0.9]]);
        assert_eq!(tm5.team.spirit_full, vec!["goal".to_owned(), "possession".to_owned()]);
        for k in &tm5.team.spirit_full {
            assert!(!tm5.team.individual.contains(k), "{k} both individual and full");
        }
        let built = tm5.build();
        let names: Vec<_> = built.terms.iter().map(|t| t.name).collect();
        for t in ["teammate_spacing", "last_man_goalside", "support_band", "double_commit", "pass", "double_tap"] {
            assert!(names.contains(&t), "{t} missing from tm5");
        }
        assert!(!names.contains(&"concede_dist"));
        for t in ["teammate_spacing", "last_man_goalside", "support_band", "double_commit"] {
            assert!(tm5.team.individual.iter().any(|k| k == t), "{t} must be self-only");
        }
        for t in ["pass", "double_tap", "air_double_touch", "flip_reset", "flip_reset_use", "musty"] {
            assert!(!tm5.team.no_second_commit_gate.iter().any(|k| k == t), "{t} gated");
        }
        // 1v1 / st36 unchanged by the new fields.
        let st36 = RewardConfig::from_file(&dir.join("st36.toml")).unwrap();
        let n: Vec<_> = st36.build().terms.iter().map(|t| t.name).collect();
        assert!(!n.contains(&"teammate_spacing"));
    }

    #[test]
    fn tm5r_config_loads_commit_order_and_stronger_spacing() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let tm5 = RewardConfig::from_file(&dir.join("tm5.toml")).unwrap();
        assert!(tm5.team.commit_order.is_none(), "TM5 unchanged");
        let c = RewardConfig::from_file(&dir.join("tm5r.toml")).unwrap();
        assert_eq!(c.team.commit_order.as_ref().unwrap().hold, 8);
        assert_eq!(c.weights.teammate_spacing, 2.0);
        assert_eq!(c.weights.double_commit, 4.0);
        assert_eq!((c.team.spacing.near, c.team.spacing.clear), (600.0, 1500.0));
        assert!(c.team.spacing.exempt_closest);
        // Each new term starts at its TM5 value (0.5 / 2.0), roles stay flat.
        let sr = c.team.spacing.spacing_ramp.as_ref().unwrap();
        assert!((sr[0][1] * c.weights.teammate_spacing - 0.5).abs() < 1e-9 && sr[1] == [100_000_000.0, 1.0]);
        let dr = c.team.spacing.double_commit_ramp.as_ref().unwrap();
        assert!((dr[0][1] * c.weights.double_commit - 2.0).abs() < 1e-9 && dr[1] == [100_000_000.0, 1.0]);
        assert_eq!(c.team.spacing.role_ramp, Some(vec![[0.0, 1.0]]));
        assert!(tm5.team.spacing.spacing_ramp.is_none() && tm5.team.spacing.role_ramp.is_none(), "TM5 unchanged");
        assert_eq!(c.weights.last_man_goalside, 0.2);
        assert_eq!(c.weights.support_band, 0.1);
        assert_eq!(c.team.no_second_commit_gate, tm5.team.no_second_commit_gate);
        assert_eq!(c.team.individual, tm5.team.individual);
        let names: Vec<_> = c.build().terms.iter().map(|t| t.name).collect();
        for t in ["teammate_spacing", "last_man_goalside", "support_band", "double_commit", "touch", "possession"] {
            assert!(names.contains(&t), "{t} missing from tm5r");
        }
    }

    /// TM5r ramps at the run start (learner step 0 = spirit_origin 0 here):
    /// teammate_spacing pays 2.0 x 0.25 = 0.5 (TM5's value), the role terms
    /// their full 0.2 / 0.1 (no dip).
    #[test]
    fn tm5r_ramps_start_at_tm5_values_and_roles_stay_flat() {
        use crate::state::{BallView, CarView, StateView};
        use glam::DVec3;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm5r.toml")).unwrap();
        let car = |team: u8, y: f64| CarView { team, pos: DVec3::new(0.0, y, 17.0), ..CarView::default() };
        // Blue: closest at -300, second at -700 (400 uu from it), last back at -3000.
        let s = StateView {
            tick_count: 0,
            ball: BallView { pos: DVec3::new(10.0, 0.0, 93.0), vel: DVec3::new(0.0, 500.0, 0.0), ..BallView::default() },
            cars: vec![car(0, -300.0), car(0, -700.0), car(0, -3000.0), car(1, 3000.0)],
        };
        let mut b = c.build();
        b.reset_view(&s);
        let _ = b.step_view(&s);
        let names: Vec<&str> = b.terms.iter().map(|t| t.name).collect();
        let get = |n: &str| b.last_terms()[names.iter().position(|x| *x == n).unwrap()].clone();
        let sp = get("teammate_spacing");
        assert!((sp[1] + 0.5).abs() < 1e-9 && sp[0] == 0.0, "{sp:?}");
        let lm = get("last_man_goalside");
        assert!((lm[2] - 0.2).abs() < 1e-9, "{lm:?}");
    }

    /// TM6 builds with the role terms and gates; at the run start
    /// teammate_spacing pays 2.0 (first man by role exempt), support_role 0
    /// (ramp 0), last_man_goalside 0.3 x 0.6667 on the Last role, the
    /// first-man gates zero face_ball for the non-first car, and the
    /// kickoff-exempt terms are 0 during the kickoff.
    #[test]
    fn tm6_builds_role_terms_and_gates() {
        use crate::state::{BallView, CarView, StateView};
        use glam::DVec3;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm6.toml")).unwrap();
        assert_eq!(c.team.roles, Some(RoleParams { margin: 300.0, hold: 8, ..RoleParams::default() }));
        assert_eq!((c.weights.support_role, c.weights.support_band, c.weights.last_man_goalside), (0.2, 0.0, 0.3));
        assert_eq!(c.team.role_scale.get("contact_quality"), Some(&0.25));
        assert_eq!(c.team.closest_gate, vec!["shadow_space".to_string()]);
        let car = |team: u8, y: f64| CarView { team, pos: DVec3::new(0.0, y, 17.0), forward: DVec3::Y, ..CarView::default() };
        // Blue: first at -300, a teammate 400 uu behind it, last back at -3000.
        let s = StateView {
            tick_count: 0,
            ball: BallView { pos: DVec3::new(10.0, 0.0, 93.0), vel: DVec3::new(0.0, 500.0, 0.0), ..BallView::default() },
            cars: vec![car(0, -300.0), car(0, -700.0), car(0, -3000.0), car(1, 3000.0)],
        };
        let mut b = c.build();
        let names: Vec<&str> = b.terms.iter().map(|t| t.name).collect();
        for t in ["teammate_spacing", "last_man_goalside", "support_role", "double_commit", "contact_quality", "shadow_box"] {
            assert!(names.contains(&t), "{t} missing from tm6");
        }
        assert!(!names.contains(&"support_band"));
        b.reset_view(&s);
        let _ = b.step_view(&s);
        let get = |b: &CombinedReward, n: &str| b.last_terms()[names.iter().position(|x| *x == n).unwrap()].clone();
        let sp = get(&b, "teammate_spacing");
        assert!(sp[0] == 0.0 && (sp[1] + 2.0).abs() < 1e-9, "{sp:?}");
        let lm = get(&b, "last_man_goalside");
        assert!((lm[2] - 0.3 * 0.6667).abs() < 1e-6 && lm[1] == 0.0, "{lm:?}");
        assert!(get(&b, "support_role").iter().all(|x| *x == 0.0));
        let fb = get(&b, "face_ball");
        // Zero-sum after the gate: the gated cars keep only the opponents' -mean.
        assert!((fb[1] - fb[2]).abs() < 1e-12 && fb[0] > fb[1], "face_ball first man only: {fb:?}");
        // Role-split diagnostics: blue first man vs its 2 teammates; orange (1 car) not split.
        let split = b.last_role_split();
        let ts = split.iter().find(|(n, _, _)| *n == "teammate_spacing").expect("teammate_spacing split");
        assert_eq!(ts.2, [1, 2]);
        assert!((ts.1[1] + 2.0).abs() < 1e-9 && ts.1[0] == 0.0, "{ts:?}");
        // Kickoff pending (ball resting at center), two blue cars committing together.
        let mut k = s.clone();
        k.ball = BallView { pos: DVec3::new(0.0, 0.0, 93.0), ..BallView::default() };
        k.cars[0].pos = DVec3::new(-100.0, -400.0, 17.0);
        k.cars[1].pos = DVec3::new(100.0, -400.0, 17.0);
        for c in &mut k.cars[..2] {
            c.vel = DVec3::new(0.0, 1500.0, 0.0);
        }
        let mut b = c.build();
        b.reset_view(&k);
        let _ = b.step_view(&k);
        for t in ["teammate_spacing", "double_commit", "support_role"] {
            assert!(get(&b, t).iter().all(|x| *x == 0.0), "{t} not exempt at kickoff");
        }
    }

    #[test]
    fn tm2_config_builds_with_the_tm2_terms() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm2.toml")).unwrap();
        assert_eq!(c.team.spirit, vec![[0.0, 0.9]]);
        assert_eq!(c.weights.musty, 40.0);
        assert_eq!(c.weights.air_double_touch, 20.0);
        assert_eq!((c.weights.airdribble, c.weights.airdribble_seq), (35.0, 35.0));
        assert_eq!(c.weights.flip_reset_use, 30.0);
        assert!(c.flip_reset_use.goal_requires_use);
        assert_eq!(c.flip_reset_use.max_obtains_per_flight, 1);
        assert!((c.weights.flip_reset_use * c.flip_reset_use.goal_bonus - 400.0).abs() < 2.0);
        assert_eq!(c.flip_reset.obtain_decay, 0.0);
        assert_eq!(c.team.double_tap.backboard_helper, 0.05);
        assert_eq!(c.weights.double_tap, 400.0);
        assert!(c.team.individual.iter().any(|k| k == "musty"));
        let names: Vec<_> = c.build().terms.iter().map(|t| t.name).collect();
        for t in ["musty", "flip_reset_use", "double_tap", "air_double_touch", "pass"] {
            assert!(names.contains(&t), "{t} missing from tm2");
        }
        // TM1 configs are unchanged: no musty term, FR goal rule.
        let tm1 = RewardConfig::from_file(&dir.join("tm1_fr.toml")).unwrap();
        assert!(!tm1.flip_reset_use.goal_requires_use);
        assert!(!tm1.build().terms.iter().any(|t| t.name == "musty"));
    }

    #[test]
    #[should_panic(expected = "unknown or unweighted term")]
    fn team_lists_reject_unknown_terms() {
        let mut c = RewardConfig::default();
        c.team.closest_gate = vec!["nope".into()];
        let _ = c.build();
    }

    /// TM7: tm6 plus teammate_spacing 4.0 (2.0 -> 4.0 over 200M, clear 1800),
    /// teammate_far 0.5, pass receiver_min_sep 1200 and concede_role 120
    /// (individual); kickoff-exempt teammate_far; 1v1 builds no extra reward.
    #[test]
    fn tm7_builds_spacing_far_pass_and_concede_role() {
        use crate::state::{BallView, CarView, StateView};
        use glam::DVec3;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm7.toml")).unwrap();
        assert_eq!((c.weights.teammate_spacing, c.weights.teammate_far, c.weights.concede_role), (4.0, 0.5, 120.0));
        assert_eq!((c.team.spacing.near, c.team.spacing.clear), (600.0, 1800.0));
        assert_eq!(c.team.spacing.spacing_ramp, Some(vec![[0.0, 0.5], [200_000_000.0, 1.0]]));
        assert_eq!((c.team.spacing.far_start, c.team.spacing.far_span), (4500.0, 1500.0));
        assert_eq!(c.team.pass.receiver_min_sep, 1200.0);
        assert_eq!(c.team.concede_role, ConcedeRoleParams::default());
        // TM6 kept: role gates, support / last man, double commit, spirit.
        assert_eq!(c.team.roles, Some(RoleParams { margin: 300.0, hold: 8, ..RoleParams::default() }));
        assert_eq!((c.weights.support_role, c.weights.last_man_goalside, c.weights.double_commit), (0.2, 0.3, 4.0));
        assert_eq!(c.team.spirit, vec![[0.0, 0.9]]);
        assert!(c.team.no_second_commit_gate.iter().any(|k| k == "touch"));
        for t in ["teammate_far", "concede_role", "teammate_spacing"] {
            assert!(c.team.individual.iter().any(|k| k == t), "{t} not individual");
        }
        assert!(c.team.kickoff_exempt.iter().any(|k| k == "teammate_far"));
        let tm6 = RewardConfig::from_file(&dir.join("tm6.toml")).unwrap();
        assert_eq!((tm6.weights.teammate_far, tm6.weights.concede_role, tm6.team.pass.receiver_min_sep), (0.0, 0.0, 0.0));
        let car = |team: u8, y: f64| CarView { team, pos: DVec3::new(0.0, y, 17.0), forward: DVec3::Y, ..CarView::default() };
        // Blue: first at -300, a teammate 400 uu behind it, a third 6500 behind.
        let s = StateView {
            tick_count: 0,
            ball: BallView { pos: DVec3::new(10.0, 0.0, 93.0), vel: DVec3::new(0.0, 500.0, 0.0), ..BallView::default() },
            cars: vec![car(0, -300.0), car(0, -700.0), car(0, -7000.0), car(1, 3000.0)],
        };
        let mut b = c.build();
        let names: Vec<&str> = b.terms.iter().map(|t| t.name).collect();
        for t in ["teammate_far", "concede_role", "teammate_spacing", "pass", "support_role"] {
            assert!(names.contains(&t), "{t} missing from tm7");
        }
        b.reset_view(&s);
        let _ = b.step_view(&s);
        let get = |b: &CombinedReward, n: &str| b.last_terms()[names.iter().position(|x| *x == n).unwrap()].clone();
        // Spacing at the run start: 4.0 x 0.5, d = 400 <= near.
        let sp = get(&b, "teammate_spacing");
        assert!(sp[0] == 0.0 && (sp[1] + 2.0).abs() < 1e-9, "{sp:?}");
        // Far: car 2 is 6300 from its nearest teammate -> -1 x 0.5, unblended.
        let far = get(&b, "teammate_far");
        assert!((far[2] + 0.5).abs() < 1e-9 && far[0] == 0.0 && far[1] == 0.0 && far[3] == 0.0, "{far:?}");
        // Orange scores into blue's goal: the role split survives the blend.
        let mut g = s.clone();
        g.ball.pos = DVec3::new(0.0, -5400.0, 93.0);
        let _ = b.step_view(&g);
        let cr = get(&b, "concede_role");
        let blue = cr[0] + cr[1] + cr[2];
        assert!((blue + 3.0 * 120.0).abs() < 1e-6 && cr[3] == 0.0, "{cr:?}");
        assert!(cr[0] > cr[1] && cr[1] > cr[2], "first < second < last penalty: {cr:?}");
        // 1v1: no far / concede_role reward.
        let s1 = StateView { cars: vec![car(0, -300.0), car(1, 3000.0)], ..s.clone() };
        let mut b = c.build();
        b.reset_view(&s1);
        let _ = b.step_view(&s1);
        let mut g1 = s1.clone();
        g1.ball.pos = DVec3::new(0.0, -5400.0, 93.0);
        let _ = b.step_view(&g1);
        assert!(get(&b, "concede_role").iter().all(|x| *x == 0.0));
        assert!(get(&b, "teammate_far").iter().all(|x| *x == 0.0));
    }

    /// TM9: tm8 plus kickoff_same_pad 5.0, kickoff_one_corner, boost-scale
    /// grace 3s, contact_quality baked to 100, term_ramp held at 1.0.
    /// tm8.toml keeps the pre-TM9 defaults.
    #[test]
    fn tm9_holds_tm8_end_ramps_and_one_corner() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm9.toml")).unwrap();
        assert_eq!(c.weights.kickoff_same_pad, 5.0);
        assert_eq!(c.weights.contact_quality, 100.0);
        assert!(c.team.kickoff_one_corner);
        assert_eq!(c.team.kickoff_scale_grace_s, Some(3.0));
        assert_eq!(c.team.kickoff_grace_s, 1.0);
        assert_eq!(c.team.kickoff_pad_grace_s, 3.0);
        assert_eq!(c.team.term_ramp.get("contact_quality"), Some(&vec![[0.0, 1.0]]));
        assert_eq!(c.team.term_ramp.get("shadow_cover"), Some(&vec![[0.0, 1.0]]));
        assert_eq!(c.team.spirit, vec![[0.0, 0.9]]);
        let names: Vec<_> = c.build().terms.iter().map(|t| t.name).collect();
        assert!(names.contains(&"kickoff_same_pad"), "{names:?}");
        let tm8 = RewardConfig::from_file(&dir.join("tm8.toml")).unwrap();
        assert!(!tm8.team.kickoff_one_corner);
        assert_eq!(tm8.team.kickoff_scale_grace_s, None);
        assert_eq!(tm8.weights.kickoff_same_pad, 0.5);
        assert_eq!(tm8.weights.contact_quality, 250.0);
    }

    /// TM10: tm9 plus the support-car playmaking demo / bump terms, both
    /// individual (post team spirit), not zero-sum, ramped 0.25 -> 1 over
    /// 200M; the TM9 terms and ramps are unchanged.
    #[test]
    fn tm10_adds_playmaking_on_top_of_tm9() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm10.toml")).unwrap();
        let tm9 = RewardConfig::from_file(&dir.join("tm9.toml")).unwrap();
        assert_eq!(c.weights.playmaking_demo, 150.0);
        assert_eq!(c.weights.playmaking_bump, 20.0);
        assert_eq!(c.weights.demo, tm9.weights.demo);
        assert_eq!(c.weights.contact_quality, 100.0);
        assert_eq!(c.weights.kickoff_same_pad, 5.0);
        assert_eq!(c.playmaking, PlaymakingParams::default());
        for k in ["playmaking_demo", "playmaking_bump"] {
            assert!(c.team.individual.iter().any(|x| x == k), "{k} individual");
            assert!(c.team.role_metrics.iter().any(|x| x == k), "{k} role metric");
            assert!(!c.zero_sum.contains_key(k), "{k} not zero-sum");
            assert_eq!(c.team.term_ramp.get(k), Some(&vec![[0.0, 0.25], [200_000_000.0, 1.0]]));
        }
        for k in &tm9.team.individual {
            assert!(c.team.individual.contains(k), "tm9 individual {k} kept");
        }
        for k in &tm9.team.role_metrics {
            assert!(c.team.role_metrics.contains(k), "tm9 role metric {k} kept");
        }
        assert_eq!(c.team.term_ramp.get("contact_quality"), Some(&vec![[0.0, 1.0]]));
        assert_eq!(c.team.term_ramp.get("shadow_cover"), Some(&vec![[0.0, 1.0]]));
        assert!(c.team.kickoff_one_corner);
        assert_eq!(c.team.spirit, vec![[0.0, 0.9]]);
        let names: Vec<_> = c.build().terms.iter().map(|t| t.name).collect();
        assert!(names.contains(&"playmaking_demo") && names.contains(&"playmaking_bump"), "{names:?}");
        let n9: Vec<_> = tm9.build().terms.iter().map(|t| t.name).collect();
        assert!(!n9.contains(&"playmaking_demo"));
        assert_eq!(names.len(), n9.len() + 2);
    }

    fn ramp_at(pts: &[[f64; 2]], step: f64) -> f64 {
        if pts.is_empty() {
            return 1.0;
        }
        if step <= pts[0][0] {
            return pts[0][1];
        }
        for w in pts.windows(2) {
            let (t0, v0, t1, v1) = (w[0][0], w[0][1], w[1][0], w[1][1]);
            if step <= t1 {
                if (t1 - t0).abs() < 1e-12 {
                    return v1;
                }
                let a = (step - t0) / (t1 - t0);
                return v0 + a * (v1 - v0);
            }
        }
        pts.last().unwrap()[1]
    }

    /// TM11: tm10 plus defend_carry / carry_break / kickoff_outcome, playmaking
    /// ramps held at 1.0 so a new spirit_origin does not re-ramp them.
    #[test]
    fn tm11_extends_tm10_holds_ramps() {
        use crate::team::RoleParams;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm11.toml")).unwrap();
        let tm10 = RewardConfig::from_file(&dir.join("tm10.toml")).unwrap();
        assert_eq!(c.weights.defend_carry, 40.0);
        assert_eq!(c.weights.carry_break, 600.0);
        assert_eq!(c.weights.kickoff_outcome, 120.0);
        assert_eq!(c.weights.playmaking_demo, 150.0);
        assert_eq!(c.weights.playmaking_bump, 20.0);
        for k in ["defend_carry", "carry_break", "kickoff_outcome"] {
            assert!(c.team.individual.iter().any(|x| x == k), "{k} individual");
            assert!(c.team.role_metrics.iter().any(|x| x == k), "{k} role metric");
            assert!(!c.zero_sum.contains_key(k), "{k} not zero-sum");
        }
        for k in &tm10.team.individual {
            assert!(c.team.individual.contains(k), "tm10 individual {k} kept");
        }
        for k in &tm10.team.role_metrics {
            assert!(c.team.role_metrics.contains(k), "tm10 role metric {k} kept");
        }
        let roles = c.team.roles.clone().unwrap_or_default();
        assert_eq!(
            roles,
            RoleParams { margin: 300.0, hold: 8, obs_tiebreak: true, touch_mode: "hysteresis".into() }
        );
        let new_terms = ["defend_carry", "carry_break", "kickoff_outcome"];
        for (k, pts) in &c.team.term_ramp {
            if new_terms.contains(&k.as_str()) {
                assert!((ramp_at(pts, 0.0) - 0.25).abs() < 1e-9, "{k} at 0: {}", ramp_at(pts, 0.0));
                assert!((ramp_at(pts, 200_000_000.0) - 1.0).abs() < 1e-9, "{k} at 2e8");
            } else {
                assert!((ramp_at(pts, 0.0).abs() - 1.0).abs() < 1e-9, "{k} at 0: {}", ramp_at(pts, 0.0));
            }
        }
        let names: Vec<_> = c.build().terms.iter().map(|t| t.name).collect();
        let n10: Vec<_> = tm10.build().terms.iter().map(|t| t.name).collect();
        assert_eq!(names.len(), n10.len() + 3, "{names:?}");
        for k in new_terms {
            assert!(names.contains(&k), "{k} missing from {names:?}");
        }
    }

    #[test]
    fn tm11_kickoff_terms_kept() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments");
        let c = RewardConfig::from_file(&dir.join("tm11.toml")).unwrap();
        let tm10 = RewardConfig::from_file(&dir.join("tm10.toml")).unwrap();
        assert_eq!(c.weights.kickoff_same_pad, tm10.weights.kickoff_same_pad);
        assert_eq!(c.weights.kickoff_taker, tm10.weights.kickoff_taker);
        assert_eq!(c.team.kickoff_one_corner, tm10.team.kickoff_one_corner);
        assert_eq!(c.team.kickoff_pad_radius, tm10.team.kickoff_pad_radius);
        assert_eq!(c.team.kickoff_pad_grace_s, tm10.team.kickoff_pad_grace_s);
        assert_eq!(c.team.kickoff_scale, tm10.team.kickoff_scale);
        assert_eq!(c.team.kickoff_scale_grace_s, tm10.team.kickoff_scale_grace_s);
    }
}
