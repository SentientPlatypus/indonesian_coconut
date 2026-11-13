use std::fs;
use std::path::PathBuf;

use burn::prelude::*;
use rand::Rng;
use rand::rngs::SmallRng;

use super::model::Actic;
use super::skill_tracker::SkillRating;

/// Configuration for saving policy versions and training against old
/// versions (self-play).
///
/// When `train_against_old_versions` is enabled, the collector will
/// occasionally use a randomly-chosen saved policy version as the
/// opponent for one team.  This provides a natural curriculum:  the
/// current policy learns to beat increasingly strong versions of itself.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Whether to periodically snapshot the current policy.
    /// Automatically enabled when `train_against_old_versions` is true.
    pub save_policy_versions: bool,
    /// Number of timesteps between saving a new policy version.
    pub ts_per_version: u64,
    /// Maximum number of old versions to keep in memory.
    pub max_old_versions: usize,
    /// Maximum number of policy versions retained on disk.
    ///
    /// `None` preserves every saved version. `Some(n)` retains the newest
    /// `n` version directories independently of `max_old_versions`.
    pub max_saved_versions: Option<usize>,
    /// Whether to occasionally train against old versions (self-play).
    pub train_against_old_versions: bool,
    /// Probability (0.0 – 1.0) that a given training iteration will
    /// use an old version as the opponent.
    pub train_against_old_chance: f32,
    /// Sampling weight of the saved past versions as a group, relative to
    /// `fixed_opponents`. With no fixed opponents only the ratio to zero
    /// matters, so the historical behavior is unchanged.
    pub past_versions_weight: f32,
    /// Frozen opponents that never rotate out (e.g. an imported reference
    /// policy or a kept-best checkpoint), each with its sampling weight.
    pub fixed_opponents: Vec<FixedOpponent>,
    /// Frozen non-`Actic` opponents (own obs/actions), each with its weight.
    pub external_opponents: Vec<super::external::ExternalOpponent>,
}

/// A frozen opponent policy in the training pool.
#[derive(Clone, Debug)]
pub struct FixedOpponent {
    pub name: String,
    pub model: Actic,
    pub weight: f32,
}

/// The opponent chosen for one training iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentPick {
    /// Index into `VersionManager::versions`.
    PastVersion(usize),
    /// Index into `SelfPlayConfig::fixed_opponents`.
    Fixed(usize),
    /// Index into `SelfPlayConfig::external_opponents`.
    External(usize),
}

impl SelfPlayConfig {
    /// Whether any iteration can be played against a non-current policy.
    pub fn uses_opponents(&self) -> bool {
        self.train_against_old_versions
            || !self.fixed_opponents.is_empty()
            || !self.external_opponents.is_empty()
    }

    /// Picks the opponent for one iteration, or `None` for pure self-play.
    ///
    /// With probability `train_against_old_chance` an opponent is drawn:
    /// the past versions as a group (uniform inside it, only if
    /// `train_against_old_versions` and any exist) against each fixed
    /// opponent, in proportion to their weights.
    pub fn pick_opponent(&self, num_versions: usize, rng: &mut SmallRng) -> Option<OpponentPick> {
        let past = if self.train_against_old_versions && num_versions > 0 {
            self.past_versions_weight
        } else {
            0.0
        };
        let mut fixed: Vec<f32> = self.fixed_opponents.iter().map(|o| o.weight).collect();
        fixed.extend(self.external_opponents.iter().map(|o| o.weight));
        let n_fixed = self.fixed_opponents.len();
        pick_weighted(
            self.train_against_old_chance,
            past,
            &fixed,
            num_versions,
            rng,
        )
        .map(|pick| match pick {
            OpponentPick::Fixed(i) if i >= n_fixed => OpponentPick::External(i - n_fixed),
            other => other,
        })
    }
}

/// An externally computed opponent mix for one rollout (indococo ST35 PFSP):
/// probability `chance` of not playing pure self-play, then past versions vs
/// each fixed-then-external opponent in proportion to `past_weight` and
/// `weights` (same order as [`SelfPlayConfig::opponent_names`]).
#[derive(Clone, Debug, PartialEq)]
pub struct OpponentMix {
    pub chance: f32,
    pub past_weight: f32,
    pub weights: Vec<f32>,
}

impl SelfPlayConfig {
    /// Fixed then external opponent names (the order of [`OpponentMix::weights`]).
    pub fn opponent_names(&self) -> Vec<&str> {
        self.fixed_opponents
            .iter()
            .map(|o| o.name.as_str())
            .chain(self.external_opponents.iter().map(|o| o.name.as_str()))
            .collect()
    }

    /// Past versions usable for a pick (0 when old versions are off).
    pub fn usable_versions(&self, num_versions: usize) -> usize {
        if self.train_against_old_versions { num_versions } else { 0 }
    }

    /// Like [`Self::pick_opponent`] with the weights of `mix`. A mix with the
    /// wrong number of weights falls back to the configured weights.
    pub fn pick_opponent_mix(
        &self,
        mix: &OpponentMix,
        num_versions: usize,
        rng: &mut SmallRng,
    ) -> Option<OpponentPick> {
        let n_fixed = self.fixed_opponents.len();
        if mix.weights.len() != n_fixed + self.external_opponents.len() {
            return self.pick_opponent(num_versions, rng);
        }
        let nv = self.usable_versions(num_versions);
        let past = if nv > 0 { mix.past_weight } else { 0.0 };
        pick_weighted(mix.chance, past, &mix.weights, nv, rng).map(|pick| match pick {
            OpponentPick::Fixed(i) if i >= n_fixed => OpponentPick::External(i - n_fixed),
            other => other,
        })
    }
}

/// [`SelfPlayConfig::pick_opponent`] on plain weights (negative weights count as 0).
fn pick_weighted(
    chance: f32,
    past_weight: f32,
    fixed_weights: &[f32],
    num_versions: usize,
    rng: &mut SmallRng,
) -> Option<OpponentPick> {
    let past = past_weight.max(0.0);
    let total = past + fixed_weights.iter().map(|w| w.max(0.0)).sum::<f32>();
    if total <= 0.0 || unit(rng) >= chance {
        return None;
    }
    let mut x = unit(rng) * total;
    if x < past {
        return Some(OpponentPick::PastVersion(
            rng.next_u32() as usize % num_versions,
        ));
    }
    x -= past;
    for (i, &w) in fixed_weights.iter().enumerate() {
        let w = w.max(0.0);
        if x < w {
            return Some(OpponentPick::Fixed(i));
        }
        x -= w;
    }
    // Float rounding at the top end.
    fixed_weights
        .iter()
        .rposition(|&w| w > 0.0)
        .map(OpponentPick::Fixed)
}

fn unit(rng: &mut SmallRng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            save_policy_versions: false,
            ts_per_version: 25_000_000,
            max_old_versions: 32,
            max_saved_versions: Some(32),
            train_against_old_versions: false,
            train_against_old_chance: 0.15,
            past_versions_weight: 1.0,
            fixed_opponents: Vec::new(),
            external_opponents: Vec::new(),
        }
    }
}

/// A snapshot of the policy network saved at a particular timestep.
/// Uses `Module::clone()` for a deep copy so later mutations of the
/// original network do not affect the stored version.
#[derive(Clone, Debug)]
pub struct PolicyVersion {
    pub timesteps: u64,
    pub model: Actic,
    /// Frozen Elo ratings at the time this version was created.
    pub ratings: SkillRating,
}

impl PolicyVersion {
    /// Deep-copy the model into a frozen version snapshot.
    /// Optionally records the current skill ratings.
    pub fn from_model(model: &Actic, timesteps: u64, ratings: SkillRating) -> Self {
        // Burn's Module derive provides Clone, which does a true
        // deep-copy (independent parameter buffers).
        Self {
            timesteps,
            model: model.clone(),
            ratings,
        }
    }
}

/// Manages a sliding window of saved policy versions for self-play.
pub struct VersionManager {
    pub versions: Vec<PolicyVersion>,
    config: SelfPlayConfig,
    #[allow(dead_code)]
    save_folder: PathBuf,
}

impl VersionManager {
    pub fn new(save_folder: PathBuf, config: SelfPlayConfig) -> Self {
        Self {
            versions: Vec::new(),
            config,
            save_folder,
        }
    }

    /// Called after every training iteration.  Snapshots a new version
    /// when crossing a `ts_per_version` boundary, and prunes the oldest
    /// version when the sliding window is full.
    ///
    /// `cur_ratings`, when provided, is frozen into the new version
    /// (the skill tracker's current Elo ratings).
    pub fn on_iteration(
        &mut self,
        model: &Actic,
        cur_timesteps: u64,
        prev_timesteps: u64,
        cur_ratings: Option<&SkillRating>,
    ) {
        if !self.config.save_policy_versions {
            return;
        }

        let crossed = cur_timesteps / self.config.ts_per_version
            > prev_timesteps / self.config.ts_per_version;
        if crossed || prev_timesteps == 0 {
            #[cfg(not(feature = "tui"))]
            println!(" > Saving policy version at {cur_timesteps} ts ...");

            let version = PolicyVersion::from_model(
                model,
                cur_timesteps,
                cur_ratings.cloned().unwrap_or_default(),
            );
            self.persist_version(&version);
            self.versions.push(version);

            while self.versions.len() > self.config.max_old_versions {
                self.versions.remove(0);
            }
            self.prune_saved_versions();

            #[cfg(not(feature = "tui"))]
            println!(
                " > Saved policy version at {cur_timesteps} ts ({} versions now)",
                self.versions.len()
            );
        }
    }

    // ── Disk persistence ─────────────────────────────────────────

    /// Persist in-memory versions and enforce the independent disk-retention
    /// policy. New versions are also persisted immediately by `on_iteration`.
    pub fn save_versions(&self) {
        for version in &self.versions {
            self.persist_version(version);
        }
        self.prune_saved_versions();
    }

    fn persist_version(&self, version: &PolicyVersion) {
        let path = self.save_folder.join(version.timesteps.to_string());

        if path.exists() {
            // The version directory already exists. Keep the frozen model
            // files untouched, but always refresh `stats.toml` so updated
            // Elo ratings survive autosave and restart.
            let stats_toml = toml::to_string_pretty(&version.ratings).unwrap();
            fs::write(path.join("stats.toml"), stats_toml).unwrap();
            return;
        }

        fs::create_dir_all(&path).unwrap();

        version
            .model
            .actor
            .clone()
            .save_file(path.join("actor.bpk"))
            .unwrap();
        version
            .model
            .critic
            .clone()
            .save_file(path.join("critic.bpk"))
            .unwrap();
        if let Some(ref head) = version.model.shared_head {
            head.clone()
                .save_file(path.join("shared_head.bpk"))
                .unwrap();
        }

        let stats_toml = toml::to_string_pretty(&version.ratings).unwrap();
        fs::write(path.join("stats.toml"), stats_toml).unwrap();

        #[cfg(not(feature = "tui"))]
        println!(" > Persisted policy version {} to disk", version.timesteps);
    }

    fn prune_saved_versions(&self) {
        let Some(max_saved_versions) = self.config.max_saved_versions else {
            return;
        };

        let Ok(entries) = fs::read_dir(&self.save_folder) else {
            return;
        };
        let mut versions: Vec<(u64, PathBuf)> = entries
            .flatten()
            .filter_map(|entry| {
                entry
                    .file_type()
                    .ok()
                    .filter(|file_type| file_type.is_dir())?;
                let timestep = entry.file_name().to_str()?.parse::<u64>().ok()?;
                Some((timestep, entry.path()))
            })
            .collect();
        versions.sort_unstable_by_key(|(timestep, _)| *timestep);

        let remove_count = versions.len().saturating_sub(max_saved_versions);
        for (_, path) in versions.into_iter().take(remove_count) {
            let _ = fs::remove_dir_all(&path);
            #[cfg(not(feature = "tui"))]
            println!(" > Removed old policy version {}", path.display());
        }
    }

    /// Load previously-saved versions from `save_folder`.  A template
    /// model is required to create new network instances (the weights
    /// are replaced from disk).  Versions whose timestep exceeds
    /// `cur_timesteps` are skipped to avoid loading checkpoints that
    /// are newer than the current model.
    pub fn load_versions(&mut self, template: &Actic, _device: &Device, cur_timesteps: u64) {
        self.versions.clear();

        if !self.save_folder.exists() {
            return;
        }

        // Discover numbered directories.
        let mut timesteps: Vec<u64> = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.save_folder) {
            for entry in entries.flatten() {
                if !entry.file_type().is_ok_and(|t| t.is_dir()) {
                    continue;
                }
                if let Some(name) = entry.file_name().to_str()
                    && let Ok(ts) = name.parse::<u64>()
                {
                    if ts <= cur_timesteps {
                        timesteps.push(ts);
                    } else {
                        #[cfg(not(feature = "tui"))]
                        eprintln!(
                            " > Warning: policy version {ts} is newer than current model \
                             ({cur_timesteps}) – skipping"
                        );
                    }
                }
            }
        }

        timesteps.sort();

        for ts in timesteps {
            let path = self.save_folder.join(ts.to_string());
            for component in ["actor", "critic", "shared_head"] {
                let legacy = path.join(format!("{component}.mpk.gz"));
                if legacy.is_file() && !path.join(format!("{component}.bpk")).is_file() {
                    panic!(
                        "Found legacy policy-version checkpoint {}. Burn 0.22 cannot load .mpk.gz; convert it with Burn 0.21 before loading versions.",
                        legacy.display()
                    );
                }
            }

            // Destructure the template so we can move individual fields.
            let Actic {
                actor: t_actor,
                critic: t_critic,
                shared_head: t_head,
            } = template.clone();

            let actor = t_actor
                .clone()
                .try_load_file(path.join("actor.bpk"))
                .unwrap_or_else(|error| {
                    panic!("Failed to load actor policy version {}: {error}", ts)
                });
            let critic = t_critic
                .clone()
                .try_load_file(path.join("critic.bpk"))
                .unwrap_or_else(|error| {
                    panic!("Failed to load critic policy version {}: {error}", ts)
                });
            let shared_head = t_head.map(|head| {
                head.clone()
                    .try_load_file(path.join("shared_head.bpk"))
                    .unwrap_or_else(|error| {
                        panic!("Failed to load shared-head policy version {}: {error}", ts)
                    })
            });

            // Attempt to load frozen skill ratings.
            let ratings = fs::read_to_string(path.join("stats.toml"))
                .ok()
                .and_then(|s| toml::from_str::<SkillRating>(&s).ok())
                .unwrap_or_default();

            self.versions.push(PolicyVersion {
                timesteps: ts,
                model: Actic {
                    actor,
                    critic,
                    shared_head,
                },
                ratings,
            });
        }

        // Enforce the sliding window.
        while self.versions.len() > self.config.max_old_versions {
            self.versions.remove(0);
        }

        if !self.versions.is_empty() {
            #[cfg(not(feature = "tui"))]
            println!(
                " > Loaded {} policy version(s) from disk (range {} – {})",
                self.versions.len(),
                self.versions.first().unwrap().timesteps,
                self.versions.last().unwrap().timesteps
            );
        }
    }
}

#[cfg(test)]
mod opponent_pick_tests {
    use rand::SeedableRng;

    use super::*;

    /// (self-play, past-version, per-fixed) counts over 100k draws.
    fn counts(chance: f32, past: f32, fixed: &[f32], versions: usize) -> (f64, f64, Vec<f64>) {
        let mut rng = SmallRng::seed_from_u64(7);
        let (mut none, mut old, mut fix) = (0.0, 0.0, vec![0.0; fixed.len()]);
        for _ in 0..100_000 {
            match pick_weighted(chance, past, fixed, versions, &mut rng) {
                None => none += 1e-5,
                Some(OpponentPick::PastVersion(i)) => {
                    assert!(i < versions);
                    old += 1e-5;
                }
                Some(OpponentPick::Fixed(i)) => fix[i] += 1e-5,
                Some(OpponentPick::External(_)) => unreachable!("pick_weighted never returns External"),
            }
        }
        (none, old, fix)
    }

    #[test]
    fn past_versions_only_matches_historical_chance() {
        let (none, old, _) = counts(0.15, 1.0, &[], 3);
        assert!((old - 0.15).abs() < 0.01);
        assert!((none + old - 1.0).abs() < 1e-6);
        assert!(
            (counts(0.15, 0.0, &[], 0).0 - 1.0).abs() < 1e-6,
            "no versions -> self-play"
        );
    }

    #[test]
    fn mix_follows_weights() {
        let (none, old, fixed) = counts(0.4, 2.0, &[1.0, 1.0], 5);
        assert!((none - 0.6).abs() < 0.01);
        assert!((old - 0.2).abs() < 0.01);
        assert!(fixed.iter().all(|f| (f - 0.1).abs() < 0.01));
    }

    #[test]
    fn fixed_opponents_work_without_past_versions() {
        let (none, old, fixed) = counts(0.4, 0.0, &[3.0, 0.0], 5);
        assert_eq!(old, 0.0);
        assert_eq!(fixed[1], 0.0);
        assert!((fixed[0] - 0.4).abs() < 0.01);
        assert!((none + fixed[0] - 1.0).abs() < 1e-6);
    }
}
