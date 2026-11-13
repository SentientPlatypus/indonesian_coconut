//! ST35: prioritized fictitious self-play (PFSP) over the training league.
//!
//! The trainer picks one opponent per rollout (`rlgymppo` `pick_self_play`).
//! With `[pfsp]` in the train config, that pick no longer uses the static
//! `[opponents]` weights. Instead:
//!
//! * `self_share` of rollouts are pure self-play (current policy both sides),
//!   `past_share` play a saved past version (uniform inside the group), and
//!   the rest (`1 - self_share - past_share`) is the adaptive league: every
//!   `[[opponents.fixed]]` and `[[opponents.external]]` entry.
//! * Each league member's learner win rate `x` comes from finished training
//!   episodes (goals), attributed to the rollout's opponent when it started
//!   the episode or drove it for `MIN_INHERITED_STEPS` (episodes run across
//!   rollouts). It is a Beta(1,1) posterior over exponentially decayed
//!   counts (`ema_games` ~ 40 games), shared by every env/worker.
//! * League mass goes in proportion to `f_hard(x)`: `1 - x`, and for very hard
//!   opponents (`x < low_x`, 0.25) the variance form `x(1 - x)` scaled by
//!   `(1 - low_x)/(low_x(1 - low_x))` (= 4 at 0.25) so `f` is continuous at
//!   `low_x` (unscaled it would jump 4x there).
//! * Every member gets at least `floor` of league mass; `pinned` members get
//!   exactly the floor; `clamp` bounds a member's share of ALL rollouts
//!   (ST35: galactus in [0.20, 0.26]). Shares are the exact projection
//!   `s_i = clip(lambda * f_i, lo_i, hi_i)`, `sum s_i = 1`.
//! * Shares move toward that target by `smooth` per rollout (a convex blend,
//!   so the floors and clamps keep holding), recomputed at every pick.
//!
//! Every pick logs one `PFSP ...` line (stdout -> train.log) and, with `log`,
//! one JSON line per rollout.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;

/// `[pfsp]` in the train config.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PfspConfig {
    /// Off keeps the static `[opponents]` weights (and records nothing).
    #[serde(default = "yes")]
    pub enabled: bool,
    /// Share of rollouts that are pure self-play.
    #[serde(default = "d_self")]
    pub self_share: f64,
    /// Share of rollouts against a saved past version. Goes to self-play
    /// while no past version exists yet.
    #[serde(default = "d_past")]
    pub past_share: f64,
    /// Decay window of the win-rate counts, in games.
    #[serde(default = "d_ema")]
    pub ema_games: f64,
    /// Minimum share of league mass per member.
    #[serde(default = "d_floor")]
    pub floor: f64,
    /// Per-rollout step of the shares toward the target (1 = no smoothing).
    #[serde(default = "d_smooth")]
    pub smooth: f64,
    /// Below this win rate `f_hard` switches to the (scaled) variance form.
    #[serde(default = "d_low")]
    pub low_x: f64,
    /// `name = [lo, hi]`: bounds on that member's share of ALL rollouts.
    #[serde(default)]
    pub clamp: BTreeMap<String, [f64; 2]>,
    /// Members held at exactly `floor` of league mass (not adaptive).
    #[serde(default)]
    pub pinned: Vec<String>,
    /// Optional JSONL log, one line per pick.
    #[serde(default)]
    pub log: Option<PathBuf>,
}

fn yes() -> bool {
    true
}
fn d_self() -> f64 {
    0.30
}
fn d_past() -> f64 {
    0.20
}
fn d_ema() -> f64 {
    40.0
}
fn d_floor() -> f64 {
    0.03
}
fn d_smooth() -> f64 {
    0.25
}
fn d_low() -> f64 {
    0.25
}

impl Default for PfspConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            self_share: d_self(),
            past_share: d_past(),
            ema_games: d_ema(),
            floor: d_floor(),
            smooth: d_smooth(),
            low_x: d_low(),
            clamp: BTreeMap::new(),
            pinned: Vec::new(),
            log: None,
        }
    }
}

impl PfspConfig {
    /// League share of all rollouts.
    #[must_use]
    pub fn league_share(&self) -> f64 {
        (1.0 - self.self_share - self.past_share).clamp(0.0, 1.0)
    }

    /// Checks the shares, floors and clamps can all hold for `names`.
    pub fn validate(&self, names: &[&str]) -> Result<(), String> {
        let league = self.league_share();
        if !(0.0..=1.0).contains(&self.self_share) || !(0.0..=1.0).contains(&self.past_share) {
            return Err("self_share and past_share must be in [0, 1]".into());
        }
        if league <= 0.0 {
            return Err("self_share + past_share leaves no league".into());
        }
        if !(self.ema_games >= 1.0) || !(self.smooth > 0.0 && self.smooth <= 1.0) {
            return Err("ema_games >= 1 and 0 < smooth <= 1".into());
        }
        for n in self.clamp.keys().chain(self.pinned.iter()) {
            if !names.contains(&n.as_str()) {
                return Err(format!("pfsp: {n:?} is not a league opponent ({names:?})"));
            }
        }
        let (lo, hi) = bounds(self, names);
        let (sl, sh) = (lo.iter().sum::<f64>(), hi.iter().sum::<f64>());
        if lo.iter().zip(&hi).any(|(l, h)| l > h) || sl > 1.0 + 1e-9 || sh < 1.0 - 1e-9 {
            return Err(format!(
                "pfsp bounds infeasible: sum lo {sl:.3}, sum hi {sh:.3} (league share {league:.3})"
            ));
        }
        Ok(())
    }
}

/// PFSP "hard" weighting with the continuous low-x variance form.
#[must_use]
pub fn f_hard(x: f64, low_x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    if x < low_x && low_x > 0.0 && low_x < 1.0 {
        let scale = (1.0 - low_x) / (low_x * (1.0 - low_x));
        scale * x * (1.0 - x)
    } else {
        1.0 - x
    }
}

/// Per-member bounds on the share of league mass.
fn bounds(cfg: &PfspConfig, names: &[&str]) -> (Vec<f64>, Vec<f64>) {
    let league = cfg.league_share().max(1e-12);
    names
        .iter()
        .map(|n| {
            if cfg.pinned.iter().any(|p| p == n) {
                return (cfg.floor, cfg.floor);
            }
            let (mut lo, mut hi) = (cfg.floor, 1.0_f64);
            if let Some([a, b]) = cfg.clamp.get(*n) {
                lo = lo.max(a / league);
                hi = hi.min(b / league);
            }
            (lo, hi)
        })
        .unzip()
}

/// Target league shares (sum 1): `clip(lambda * f_hard(x_i), lo_i, hi_i)`.
#[must_use]
pub fn target_shares(cfg: &PfspConfig, names: &[&str], x: &[f64]) -> Vec<f64> {
    let (lo, hi) = bounds(cfg, names);
    let f: Vec<f64> = x.iter().map(|&x| f_hard(x, cfg.low_x).max(1e-6)).collect();
    let at = |lam: f64| -> Vec<f64> {
        f.iter()
            .zip(lo.iter().zip(&hi))
            .map(|(fi, (l, h))| (lam * fi).clamp(*l, *h))
            .collect()
    };
    let sum = |v: &[f64]| v.iter().sum::<f64>();
    let (mut a, mut b) = (0.0_f64, 1.0_f64);
    while sum(&at(b)) < 1.0 && b < 1e12 {
        b *= 2.0;
    }
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if sum(&at(m)) < 1.0 {
            a = m;
        } else {
            b = m;
        }
    }
    let s = at(b);
    let t = sum(&s);
    // Infeasible bounds (validated away at startup): fall back to normalizing.
    if (t - 1.0).abs() > 1e-6 {
        return s.iter().map(|v| v / t.max(1e-12)).collect();
    }
    s
}

/// Decayed Beta(1,1) win-rate counts for one opponent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WinStats {
    /// Decayed wins and games.
    pub w: f64,
    pub n: f64,
    /// Raw totals (for the log).
    pub wins: u64,
    pub games: u64,
}

impl WinStats {
    pub fn record(&mut self, won: bool, ema_games: f64) {
        let d = 1.0 - 1.0 / ema_games.max(1.0);
        self.w = self.w * d + f64::from(u8::from(won));
        self.n = self.n * d + 1.0;
        self.wins += u64::from(won);
        self.games += 1;
    }

    /// Posterior mean with the Beta(1,1) prior.
    #[must_use]
    pub fn x(&self) -> f64 {
        (self.w + 1.0) / (self.n + 2.0)
    }
}

/// The rollout mix handed back to `rlgymppo` (probabilities of all rollouts).
#[derive(Clone, Debug, PartialEq)]
pub struct Mix {
    /// Probability that the rollout is not pure self-play.
    pub chance: f64,
    pub past: f64,
    /// Per league member, in `names` order.
    pub league: Vec<f64>,
}

struct State {
    cfg: PfspConfig,
    stats: BTreeMap<String, WinStats>,
    shares: Option<Vec<f64>>,
    picks: u64,
    last_games: BTreeMap<String, (u64, u64)>,
}

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);

fn lock() -> std::sync::MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Turns PFSP on with `cfg` (off if `cfg.enabled` is false). Resets the stats.
pub fn install(cfg: PfspConfig) {
    let on = cfg.enabled;
    *lock() = Some(State {
        cfg,
        stats: BTreeMap::new(),
        shares: None,
        picks: 0,
        last_games: BTreeMap::new(),
    });
    ON.store(on, Ordering::Release);
}

/// Whether PFSP is on.
#[must_use]
pub fn enabled() -> bool {
    ON.load(Ordering::Acquire)
}

/// A decided training episode: `learner_won` against `opponent`.
pub fn record_outcome(opponent: &str, learner_won: bool) {
    if !enabled() || opponent == "past" {
        return;
    }
    if let Some(st) = lock().as_mut() {
        let ema = st.cfg.ema_games;
        st.stats.entry(opponent.to_owned()).or_default().record(learner_won, ema);
    }
}

/// Steps (15 Hz) the rollout's opponent must have driven an episode that
/// began in an earlier rollout before its goal counts (5 s). Episodes keep
/// running across rollouts (~590 steps per game per 300k-step rollout vs
/// ~290-step episodes), so many goals land in an episode inherited from the
/// previous rollout's opponent; those count for the new opponent only if it
/// had at least this long to play from the inherited state.
pub const MIN_INHERITED_STEPS: u32 = 75;

/// Per-env episode bookkeeping for the PFSP win rates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EpisodeTracker {
    seq: u64,
    fresh: bool,
    steps: u32,
    decided: bool,
}

impl EpisodeTracker {
    /// Episode start during rollout `seq`.
    pub fn reset(&mut self, seq: u64) {
        *self = Self { seq, fresh: true, steps: 0, decided: false };
    }

    /// One step during rollout `seq` with the ball at `ball_y`. Returns the
    /// scoring team (0 blue, 1 orange) when this step's goal counts: the
    /// episode's first goal, scored by the rollout's opponent's game that
    /// began in this rollout or was driven by it for `MIN_INHERITED_STEPS`.
    pub fn step(&mut self, seq: u64, ball_y: f64) -> Option<u8> {
        if seq != self.seq {
            *self = Self { seq, fresh: false, steps: 0, decided: self.decided };
        } else {
            self.steps = self.steps.saturating_add(1);
        }
        if self.decided || ball_y.abs() <= crate::consts::GOAL_THRESHOLD {
            return None;
        }
        self.decided = true;
        (self.fresh || self.steps >= MIN_INHERITED_STEPS).then_some(if ball_y > 0.0 {
            crate::consts::BLUE_TEAM
        } else {
            crate::consts::ORANGE_TEAM
        })
    }
}

/// A counted goal (see [`EpisodeTracker::step`]) for the current rollout's
/// opponent. Self-play and past versions are not tracked.
pub fn record_goal(scoring_team: u8) {
    if !enabled() {
        return;
    }
    let hit = crate::setters::with_current_opponent(|c| c.map(|(n, t)| (n.to_owned(), t)));
    if let Some((name, team)) = hit {
        record_outcome(&name, usize::from(scoring_team) != team);
    }
}

/// Pure step: next shares from the previous ones and the stats.
#[must_use]
pub fn next_shares(
    cfg: &PfspConfig,
    names: &[&str],
    x: &[f64],
    prev: Option<&[f64]>,
) -> Vec<f64> {
    let target = target_shares(cfg, names, x);
    match prev {
        Some(p) if p.len() == target.len() => p
            .iter()
            .zip(&target)
            .map(|(p, t)| (1.0 - cfg.smooth) * p + cfg.smooth * t)
            .collect(),
        _ => target,
    }
}

/// Mix for the next rollout from league shares.
#[must_use]
pub fn mix_from_shares(cfg: &PfspConfig, shares: &[f64], num_versions: usize) -> Mix {
    let league = cfg.league_share();
    let past = if num_versions > 0 { cfg.past_share } else { 0.0 };
    Mix {
        chance: past + league,
        past,
        league: shares.iter().map(|s| s * league).collect(),
    }
}

/// The trainer's per-rollout hook: recomputes the shares and logs them.
/// `None` when PFSP is off (the static weights apply).
pub fn pick_mix(names: &[&str], num_versions: usize) -> Option<Mix> {
    if !enabled() {
        return None;
    }
    let mut g = lock();
    let st = g.as_mut()?;
    let x: Vec<f64> = names
        .iter()
        .map(|n| st.stats.get(*n).map_or(0.5, WinStats::x))
        .collect();
    let shares = next_shares(&st.cfg, names, &x, st.shares.as_deref());
    let mix = mix_from_shares(&st.cfg, &shares, num_versions);
    st.picks += 1;
    log_pick(st, names, &x, &shares, &mix, num_versions);
    st.shares = Some(shares);
    Some(mix)
}

fn log_pick(st: &mut State, names: &[&str], x: &[f64], shares: &[f64], mix: &Mix, nv: usize) {
    let mut line = format!(
        "PFSP pick={} self={:.3} past={:.3} (versions {nv}) |",
        st.picks,
        1.0 - mix.chance,
        mix.past
    );
    let mut js = format!(
        "{{\"pick\":{},\"self\":{:.4},\"past\":{:.4},\"versions\":{nv},\"opp\":{{",
        st.picks,
        1.0 - mix.chance,
        mix.past
    );
    for (i, n) in names.iter().enumerate() {
        let s = st.stats.get(*n).copied().unwrap_or_default();
        let (pw, pg) = st.last_games.get(*n).copied().unwrap_or((0, 0));
        let (dw, dg) = (s.wins - pw, s.games - pg);
        line.push_str(&format!(
            " {n} x={:.3} share={:.3} g+{dg}({:.2})",
            x[i],
            mix.league[i],
            if dg > 0 { dw as f64 / dg as f64 } else { f64::NAN }
        ));
        js.push_str(&format!(
            "{}\"{n}\":{{\"x\":{:.4},\"share\":{:.4},\"league_share\":{:.4},\"games\":{},\"wins\":{},\"new_games\":{dg},\"new_wins\":{dw}}}",
            if i > 0 { "," } else { "" },
            x[i],
            mix.league[i],
            shares[i],
            s.games,
            s.wins
        ));
        st.last_games.insert((*n).to_owned(), (s.wins, s.games));
    }
    js.push_str("}}");
    println!("{line}");
    if let Some(p) = &st.cfg.log {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{js}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: [&str; 6] = ["FR4_final", "ST25_final", "HUNT_final", "ripple", "galactus", "megabyte"];

    fn st35() -> PfspConfig {
        let mut c = PfspConfig::default();
        c.clamp.insert("galactus".into(), [0.20, 0.26]);
        c.pinned = vec!["HUNT_final".into(), "FR4_final".into()];
        c
    }

    fn idx(n: &str) -> usize {
        NAMES.iter().position(|m| *m == n).unwrap()
    }

    fn check_invariants(c: &PfspConfig, s: &[f64]) {
        let league = c.league_share();
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-9, "sum {s:?}");
        for (i, n) in NAMES.iter().enumerate() {
            assert!(s[i] >= c.floor - 1e-9, "{n} below floor: {s:?}");
        }
        for p in &c.pinned {
            assert!((s[idx(p)] - c.floor).abs() < 1e-9, "{p} not pinned: {s:?}");
        }
        let g = s[idx("galactus")] * league;
        assert!((0.20 - 1e-9..=0.26 + 1e-9).contains(&g), "galactus share {g}");
    }

    #[test]
    fn episode_tracker_counts_fresh_and_long_inherited_episodes_once() {
        let goal = crate::consts::GOAL_THRESHOLD + 10.0;
        let mut t = EpisodeTracker::default();
        // Fresh episode in rollout 5: its goal counts at once, only once.
        t.reset(5);
        assert_eq!(t.step(5, 0.0), None);
        assert_eq!(t.step(5, goal), Some(crate::consts::BLUE_TEAM));
        assert_eq!(t.step(5, goal), None);
        // Inherited by rollout 6, goal 10 steps later: dropped.
        t.reset(5);
        assert_eq!(t.step(6, 0.0), None);
        for _ in 0..9 {
            assert_eq!(t.step(6, 0.0), None);
        }
        assert_eq!(t.step(6, -goal), None);
        // Inherited, goal after MIN_INHERITED_STEPS under the new opponent.
        t.reset(5);
        for _ in 0..=MIN_INHERITED_STEPS {
            assert_eq!(t.step(7, 0.0), None);
        }
        assert_eq!(t.step(7, -goal), Some(crate::consts::ORANGE_TEAM));
        // A second switch restarts the clock.
        t.reset(1);
        for _ in 0..MIN_INHERITED_STEPS {
            t.step(2, 0.0);
        }
        assert_eq!(t.step(3, goal), None);
    }

    #[test]
    fn f_hard_is_one_minus_x_and_continuous_at_low_x() {
        assert!((f_hard(0.6, 0.25) - 0.4).abs() < 1e-12);
        assert!((f_hard(0.25, 0.25) - 0.75).abs() < 1e-12);
        assert!((f_hard(0.2499999, 0.25) - 0.75).abs() < 1e-5);
        // Variance form below low_x: 4 x (1 - x) at low_x 0.25.
        assert!((f_hard(0.1, 0.25) - 4.0 * 0.1 * 0.9).abs() < 1e-12);
        assert!(f_hard(0.0, 0.25) == 0.0);
        // Hard but not hopeless beats easy; hopeless is down-weighted.
        assert!(f_hard(0.3, 0.25) > f_hard(0.8, 0.25));
        assert!(f_hard(0.02, 0.25) < f_hard(0.3, 0.25));
    }

    #[test]
    fn beta_prior_and_ema_window() {
        let mut w = WinStats::default();
        assert!((w.x() - 0.5).abs() < 1e-12);
        for _ in 0..400 {
            w.record(false, 40.0);
        }
        // ~40-game window: effective n ~ 40, so x ~ 1/42.
        assert!(w.n > 39.0 && w.n < 40.0 + 1e-9, "n {}", w.n);
        assert!(w.x() < 0.03);
        for _ in 0..40 {
            w.record(true, 40.0);
        }
        // 40 recent wins move it most of the way back.
        assert!(w.x() > 0.6, "x {}", w.x());
        assert_eq!((w.wins, w.games), (40, 440));
    }

    #[test]
    fn floors_pins_and_galactus_clamp_hold_for_any_win_rates() {
        let c = st35();
        c.validate(&NAMES).unwrap();
        let grid = [0.01, 0.1, 0.24, 0.26, 0.4, 0.6, 0.9, 0.99];
        for &a in &grid {
            for &b in &grid {
                for &g in &grid {
                    let x = [a, b, 0.5, b, g, a];
                    let s = target_shares(&c, &NAMES, &x);
                    check_invariants(&c, &s);
                }
            }
        }
    }

    #[test]
    fn galactus_clamp_binds_both_ways() {
        let c = st35();
        // Galactus easy, everyone else hard: Galactus held at the 20% floor.
        let s = target_shares(&c, &NAMES, &[0.3, 0.3, 0.3, 0.3, 0.95, 0.3]);
        assert!((s[idx("galactus")] * 0.5 - 0.20).abs() < 1e-9, "{s:?}");
        // Galactus hard, everyone else easy: capped at 26% of all rollouts.
        let s = target_shares(&c, &NAMES, &[0.9, 0.9, 0.5, 0.9, 0.3, 0.9]);
        assert!((s[idx("galactus")] * 0.5 - 0.26).abs() < 1e-9, "{s:?}");
        // The rest stays proportional to f_hard where unclamped.
        let (a, b) = (s[idx("ST25_final")], s[idx("ripple")]);
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn harder_opponents_get_more_mass() {
        let c = st35();
        let s = target_shares(&c, &NAMES, &[0.9, 0.45, 0.9, 0.8, 0.4, 0.6]);
        check_invariants(&c, &s);
        assert!(s[idx("ST25_final")] > s[idx("megabyte")]);
        assert!(s[idx("megabyte")] > s[idx("ripple")]);
        let r = s[idx("ST25_final")] / s[idx("megabyte")];
        assert!((r - 0.55 / 0.4).abs() < 1e-6, "proportional to 1-x: {r}");
    }

    #[test]
    fn priors_give_the_expected_start_mix() {
        let c = st35();
        let s = target_shares(&c, &NAMES, &[0.5; 6]);
        // Equal f: the 0.94 left after two pins over 4 -> 0.235 < 0.40, so
        // galactus sits at its 20% floor (0.40 of league) and the other 3
        // split 0.54.
        assert!((s[idx("galactus")] - 0.40).abs() < 1e-9);
        for n in ["ST25_final", "ripple", "megabyte"] {
            assert!((s[idx(n)] - 0.18).abs() < 1e-9, "{n} {s:?}");
        }
        let m = mix_from_shares(&c, &s, 3);
        assert!((m.chance - 0.70).abs() < 1e-12 && (m.past - 0.20).abs() < 1e-12);
        assert!((m.league.iter().sum::<f64>() - 0.50).abs() < 1e-12);
        // No past versions yet: their 20% goes to self-play, league stays 50%.
        let m0 = mix_from_shares(&c, &s, 0);
        assert!((m0.chance - 0.50).abs() < 1e-12 && m0.past == 0.0);
        assert!((m0.league[idx("galactus")] - 0.20).abs() < 1e-9);
    }

    #[test]
    fn smoothing_keeps_constraints_and_converges() {
        let c = st35();
        let mut prev = target_shares(&c, &NAMES, &[0.5; 6]);
        let x = [0.9, 0.3, 0.9, 0.95, 0.35, 0.7];
        let t = target_shares(&c, &NAMES, &x);
        for _ in 0..60 {
            prev = next_shares(&c, &NAMES, &x, Some(&prev));
            check_invariants(&c, &prev);
        }
        for (p, t) in prev.iter().zip(&t) {
            assert!((p - t).abs() < 1e-6);
        }
    }

    #[test]
    fn validate_rejects_infeasible_or_unknown() {
        let mut c = st35();
        c.clamp.insert("galactus".into(), [0.6, 0.7]);
        assert!(c.validate(&NAMES).is_err());
        let mut c = st35();
        c.pinned.push("nobody".into());
        assert!(c.validate(&NAMES).is_err());
        let mut c = st35();
        c.floor = 0.2;
        assert!(c.validate(&NAMES).is_err());
    }

    #[test]
    fn parses_from_toml() {
        let c: PfspConfig = toml::from_str(
            "self_share = 0.3\npast_share = 0.2\nfloor = 0.03\npinned = [\"HUNT_final\", \"FR4_final\"]\n[clamp]\ngalactus = [0.20, 0.26]\n",
        )
        .unwrap();
        assert_eq!(c, st35());
    }
}
