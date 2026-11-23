//! Named layout of the 59-slot `h2h` counter array in `src/bin/train.rs`.
//!
//! The train binary still writes and reads raw indices (`tot[9 + k]`, …).
//! Wiring [`H2hCounters`] into `h2h` / `h2h_worker` is deferred until after
//! TM10: it needs a rebuilt eval binary, and the live TM10 binary must not be
//! replaced. The constants and tests here document the current mapping so a
//! later mechanical swap can be checked.

/// Length of the `h2h_worker` output array (`train.rs` `N_H2H`).
pub const N_H2H: usize = 59;

/// One named field of the `h2h` counter array.
///
/// Shared counters occupy a single slot (`index_b` is `None`). Per-side
/// counters occupy two consecutive slots: `a` at `index`, `b` at `index + 1`
/// (`index_b = Some(index + 1)`).
#[derive(Clone, Copy, Debug)]
pub struct H2hField {
    pub name: &'static str,
    pub index: usize,
    pub index_b: Option<usize>,
}

impl H2hField {
    const fn shared(name: &'static str, index: usize) -> Self {
        Self {
            name,
            index,
            index_b: None,
        }
    }

    const fn sides(name: &'static str, index: usize) -> Self {
        Self {
            name,
            index,
            index_b: Some(index + 1),
        }
    }

    /// Every slot this field occupies.
    #[must_use]
    pub fn slots(self) -> impl Iterator<Item = usize> {
        let b = self.index_b;
        std::iter::once(self.index).chain(b)
    }
}

/// The 59-slot `h2h` layout, one entry per named field.
///
/// Index map (matches `h2h` JSON / `h2h_worker` writes in `train.rs`):
/// 0 wins, 1 losses, 2 draws, 3–4 resets a/b, 5–6 uses a/b, 7–8 demos_against
/// a/b, 9–10 passes, 11–12 aerial_passes, 13–14 double_taps, 15–16
/// air_double_touches, 17–18 double_commit, 19 steps (shared), 20–21
/// kickoff_duplicate, 22 kickoffs (shared), 23–24 last_man_back, 25–26
/// pair_lt800, 27–28 two_near_ball1500, 29–30 double_commit_wide, 31–32
/// deep_last_man, 33–34 spacing_steps, 35–36 role_swaps, 37–38 touch_first,
/// 39–40 touch_second, 41–42 touch_last, 43–44 no_challenger, 45–46
/// opp_ball_own_half, 47–48 touches, 49–50 car_steps, 51–52 depth_count,
/// 53–54 deepest_car_depth (sum), 55–56 bumps, 57–58 demos.
pub const H2H_FIELDS: &[H2hField] = &[
    H2hField::shared("wins", 0),
    H2hField::shared("losses", 1),
    H2hField::shared("draws", 2),
    H2hField::sides("resets", 3),
    H2hField::sides("uses", 5),
    H2hField::sides("demos_against", 7),
    H2hField::sides("passes", 9),
    H2hField::sides("aerial_passes", 11),
    H2hField::sides("double_taps", 13),
    H2hField::sides("air_double_touches", 15),
    H2hField::sides("double_commit", 17),
    H2hField::shared("steps", 19),
    H2hField::sides("kickoff_duplicate", 20),
    H2hField::shared("kickoffs", 22),
    H2hField::sides("last_man_back", 23),
    H2hField::sides("pair_lt800", 25),
    H2hField::sides("two_near_ball1500", 27),
    H2hField::sides("double_commit_wide", 29),
    H2hField::sides("deep_last_man", 31),
    H2hField::sides("spacing_steps", 33),
    H2hField::sides("role_swaps", 35),
    H2hField::sides("touch_first", 37),
    H2hField::sides("touch_second", 39),
    H2hField::sides("touch_last", 41),
    H2hField::sides("no_challenger", 43),
    H2hField::sides("opp_ball_own_half", 45),
    H2hField::sides("touches", 47),
    H2hField::sides("car_steps", 49),
    H2hField::sides("depth_count", 51),
    H2hField::sides("deepest_car_depth", 53),
    H2hField::sides("bumps", 55),
    H2hField::sides("demos", 57),
];

/// Named `h2h` counters, one field per metric. Convertible to and from the
/// raw `[u64; N_H2H]` array the train binary uses today.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct H2hCounters {
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub resets: [u64; 2],
    pub uses: [u64; 2],
    pub demos_against: [u64; 2],
    pub passes: [u64; 2],
    pub aerial_passes: [u64; 2],
    pub double_taps: [u64; 2],
    pub air_double_touches: [u64; 2],
    pub double_commit: [u64; 2],
    pub steps: u64,
    pub kickoff_duplicate: [u64; 2],
    pub kickoffs: u64,
    pub last_man_back: [u64; 2],
    pub pair_lt800: [u64; 2],
    pub two_near_ball1500: [u64; 2],
    pub double_commit_wide: [u64; 2],
    pub deep_last_man: [u64; 2],
    pub spacing_steps: [u64; 2],
    pub role_swaps: [u64; 2],
    pub touch_first: [u64; 2],
    pub touch_second: [u64; 2],
    pub touch_last: [u64; 2],
    pub no_challenger: [u64; 2],
    pub opp_ball_own_half: [u64; 2],
    pub touches: [u64; 2],
    pub car_steps: [u64; 2],
    pub depth_count: [u64; 2],
    pub deepest_car_depth: [u64; 2],
    pub bumps: [u64; 2],
    pub demos: [u64; 2],
}

impl H2hCounters {
    /// Copies a raw worker array into named fields.
    #[must_use]
    pub fn from_array(tot: [u64; N_H2H]) -> Self {
        let side = |i: usize| [tot[i], tot[i + 1]];
        Self {
            wins: tot[0],
            losses: tot[1],
            draws: tot[2],
            resets: side(3),
            uses: side(5),
            demos_against: side(7),
            passes: side(9),
            aerial_passes: side(11),
            double_taps: side(13),
            air_double_touches: side(15),
            double_commit: side(17),
            steps: tot[19],
            kickoff_duplicate: side(20),
            kickoffs: tot[22],
            last_man_back: side(23),
            pair_lt800: side(25),
            two_near_ball1500: side(27),
            double_commit_wide: side(29),
            deep_last_man: side(31),
            spacing_steps: side(33),
            role_swaps: side(35),
            touch_first: side(37),
            touch_second: side(39),
            touch_last: side(41),
            no_challenger: side(43),
            opp_ball_own_half: side(45),
            touches: side(47),
            car_steps: side(49),
            depth_count: side(51),
            deepest_car_depth: side(53),
            bumps: side(55),
            demos: side(57),
        }
    }

    /// The raw `[u64; N_H2H]` layout `h2h_worker` returns.
    #[must_use]
    pub fn to_array(&self) -> [u64; N_H2H] {
        let mut tot = [0u64; N_H2H];
        tot[0] = self.wins;
        tot[1] = self.losses;
        tot[2] = self.draws;
        tot[3] = self.resets[0];
        tot[4] = self.resets[1];
        tot[5] = self.uses[0];
        tot[6] = self.uses[1];
        tot[7] = self.demos_against[0];
        tot[8] = self.demos_against[1];
        tot[9] = self.passes[0];
        tot[10] = self.passes[1];
        tot[11] = self.aerial_passes[0];
        tot[12] = self.aerial_passes[1];
        tot[13] = self.double_taps[0];
        tot[14] = self.double_taps[1];
        tot[15] = self.air_double_touches[0];
        tot[16] = self.air_double_touches[1];
        tot[17] = self.double_commit[0];
        tot[18] = self.double_commit[1];
        tot[19] = self.steps;
        tot[20] = self.kickoff_duplicate[0];
        tot[21] = self.kickoff_duplicate[1];
        tot[22] = self.kickoffs;
        tot[23] = self.last_man_back[0];
        tot[24] = self.last_man_back[1];
        tot[25] = self.pair_lt800[0];
        tot[26] = self.pair_lt800[1];
        tot[27] = self.two_near_ball1500[0];
        tot[28] = self.two_near_ball1500[1];
        tot[29] = self.double_commit_wide[0];
        tot[30] = self.double_commit_wide[1];
        tot[31] = self.deep_last_man[0];
        tot[32] = self.deep_last_man[1];
        tot[33] = self.spacing_steps[0];
        tot[34] = self.spacing_steps[1];
        tot[35] = self.role_swaps[0];
        tot[36] = self.role_swaps[1];
        tot[37] = self.touch_first[0];
        tot[38] = self.touch_first[1];
        tot[39] = self.touch_second[0];
        tot[40] = self.touch_second[1];
        tot[41] = self.touch_last[0];
        tot[42] = self.touch_last[1];
        tot[43] = self.no_challenger[0];
        tot[44] = self.no_challenger[1];
        tot[45] = self.opp_ball_own_half[0];
        tot[46] = self.opp_ball_own_half[1];
        tot[47] = self.touches[0];
        tot[48] = self.touches[1];
        tot[49] = self.car_steps[0];
        tot[50] = self.car_steps[1];
        tot[51] = self.depth_count[0];
        tot[52] = self.depth_count[1];
        tot[53] = self.deepest_car_depth[0];
        tot[54] = self.deepest_car_depth[1];
        tot[55] = self.bumps[0];
        tot[56] = self.bumps[1];
        tot[57] = self.demos[0];
        tot[58] = self.demos[1];
        tot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every slot 0..N_H2H is claimed by exactly one field (no overlap, no
    /// hole). This is the "each field is written once" check for the current
    /// `h2h_worker` index arithmetic.
    #[test]
    fn each_field_is_written_once() {
        let mut owner = vec![None; N_H2H];
        for field in H2H_FIELDS {
            for slot in field.slots() {
                assert!(
                    slot < N_H2H,
                    "{} index {slot} is outside 0..{N_H2H}",
                    field.name
                );
                assert!(
                    owner[slot].is_none(),
                    "slot {slot} claimed by both {} and {}",
                    owner[slot].unwrap(),
                    field.name
                );
                owner[slot] = Some(field.name);
            }
        }
        for (i, name) in owner.iter().enumerate() {
            assert!(name.is_some(), "slot {i} is not written by any field");
        }
        assert_eq!(
            H2H_FIELDS.len() + H2H_FIELDS.iter().filter(|f| f.index_b.is_some()).count(),
            N_H2H
        );
    }

    #[test]
    fn from_array_round_trips() {
        let mut raw = [0u64; N_H2H];
        for (i, slot) in raw.iter_mut().enumerate() {
            *slot = (i as u64 + 1) * 3;
        }
        let named = H2hCounters::from_array(raw);
        assert_eq!(named.to_array(), raw);
        assert_eq!(named.wins, 3);
        assert_eq!(named.steps, 20 * 3);
        assert_eq!(named.passes, [10 * 3, 11 * 3]);
        assert_eq!(named.deepest_car_depth, [54 * 3, 55 * 3]);
        assert_eq!(named.bumps, [56 * 3, 57 * 3]);
        assert_eq!(named.demos, [58 * 3, 59 * 3]);
    }
}
