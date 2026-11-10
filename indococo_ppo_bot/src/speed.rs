//! Overlap of a collection and the PPO update that run at the same time.
//!
//! `parallel_wall_s` is the wall clock from the start of the update to the
//! moment the collection running beside it has been joined. The hidden part
//! of the sum is the overlap. A negative slack (the wall longer than the
//! sum, which is timer noise) is zero.

/// Seconds of `collect_s` and `update_s` that ran concurrently.
pub fn overlap_seconds(collect_s: f64, update_s: f64, parallel_wall_s: f64) -> f64 {
    (collect_s + update_s - parallel_wall_s).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_overlap_hides_the_shorter_side() {
        let overlap = overlap_seconds(1.4, 3.07, 3.07);
        assert!((overlap - 1.4).abs() < 1e-9);
        let wall = 1.4 + 3.07 - overlap;
        assert!((wall - 3.07).abs() < 1e-9);
    }

    #[test]
    fn partial_overlap_is_the_slack_under_the_sum() {
        let overlap = overlap_seconds(1.4, 3.07, 3.5);
        assert!((overlap - 0.97).abs() < 1e-9);
    }

    #[test]
    fn no_overlap_when_the_wall_is_the_sum() {
        assert_eq!(overlap_seconds(1.4, 3.07, 4.47), 0.0);
    }

    #[test]
    fn negative_slack_clamps_to_zero() {
        assert_eq!(overlap_seconds(1.0, 1.0, 3.0), 0.0);
    }
}
