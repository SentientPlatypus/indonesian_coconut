//! The clipped PPO surrogate objective and its diagnostics.

use burn::prelude::*;

/// Policy-side terms of one PPO mini-batch. Every field is a one-element
/// tensor that still carries the autodiff graph where applicable.
pub(crate) struct PolicyLoss {
    /// Negated clipped surrogate, `-mean(min(r * A, clip(r) * A))`.
    pub actor_loss: Tensor<1>,
    /// Mean policy entropy (divided by `ln(n_actions)` when normalized).
    pub entropy: Tensor<1>,
    /// Low-variance KL estimate `mean((r - 1) - ln r)`.
    pub kl: Tensor<1>,
    /// Fraction of samples whose ratio was clipped.
    pub clip_fraction: Tensor<1>,
}

/// Computes the clipped PPO policy loss.
///
/// * `log_probs`: `[N, n_actions]` log-probabilities under the current policy.
/// * `actions`: `[N, 1]` indices of the actions taken during the rollout.
/// * `old_log_probs`, `advantages`, `clip_eps`: `[N, 1]` per-sample values;
///   `clip_eps` is the fixed `clip_range` unless adaptive clipping is on.
pub(crate) fn clipped_policy_loss(
    log_probs: Tensor<2>,
    actions: Tensor<2, Int>,
    old_log_probs: Tensor<2>,
    advantages: Tensor<2>,
    clip_eps: Tensor<2>,
    normalize_entropy: bool,
) -> PolicyLoss {
    let num_actions = log_probs.shape().dims::<2>()[1];
    let entropy_sum = -(log_probs.clone() * log_probs.clone().exp()).sum_dim(1);
    let entropy = if normalize_entropy {
        entropy_sum / (num_actions as f32).ln()
    } else {
        entropy_sum
    }
    .mean();

    let log_ratio = log_probs.gather(1, actions) - old_log_probs;
    let ratios = log_ratio.clone().exp();
    let lower = clip_eps.clone().mul_scalar(-1.0).add_scalar(1.0);
    let upper = clip_eps.add_scalar(1.0);
    let clipped_ratios = ratios.clone().min_pair(upper).max_pair(lower);

    let kl = ((ratios.clone() - 1.0) - log_ratio).mean();
    let clip_fraction = clipped_ratios
        .clone()
        .sub(ratios.clone())
        .abs()
        .greater_elem(0.0)
        .float()
        .mean();
    let actor_loss = -(ratios * advantages.clone())
        .min_pair(clipped_ratios * advantages)
        .mean();

    PolicyLoss {
        actor_loss,
        entropy,
        kl,
        clip_fraction,
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;

    fn column(values: &[f32], device: &Device) -> Tensor<2> {
        Tensor::<1>::from_floats(values, device).unsqueeze_dim(1)
    }

    fn scalar(tensor: Tensor<1>) -> f32 {
        tensor.into_data().try_into_vec::<f32>().unwrap()[0]
    }

    /// Two-action policy with probabilities `[p, 1 - p]` per row.
    fn two_action_log_probs(p: &[f32], device: &Device) -> Tensor<2> {
        let rows: Vec<f32> = p.iter().flat_map(|&p| [p.ln(), (1.0 - p).ln()]).collect();
        Tensor::<1>::from_floats(rows.as_slice(), device).reshape([p.len(), 2])
    }

    #[test]
    fn unchanged_policy_has_zero_kl_no_clipping_and_minus_mean_advantage() {
        let device = Device::flex();
        let probs = [0.3, 0.6, 0.5];
        let actions = Tensor::<1, Int>::from_ints([0, 1, 0], &device).unsqueeze_dim(1);
        let old = column(&[0.3f32.ln(), 0.4f32.ln(), 0.5f32.ln()], &device);
        let advantages = column(&[1.0, -2.0, 4.0], &device);

        let loss = clipped_policy_loss(
            two_action_log_probs(&probs, &device),
            actions,
            old,
            advantages,
            column(&[0.2; 3], &device),
            false,
        );

        assert!(scalar(loss.kl).abs() < 1e-6);
        assert_eq!(scalar(loss.clip_fraction), 0.0);
        assert!((scalar(loss.actor_loss) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn ratios_outside_the_trust_region_are_clipped_pessimistically() {
        let device = Device::flex();
        // Action 0 doubled its probability (ratio 2); action 1 halved (ratio 0.5).
        let actions = Tensor::<1, Int>::from_ints([0, 0], &device).unsqueeze_dim(1);
        let current = two_action_log_probs(&[0.8, 0.2], &device);
        let old = column(&[0.4f32.ln(), 0.4f32.ln()], &device);
        // Positive advantage caps the gain at 1.2; negative keeps the full 0.5 ratio penalty.
        let advantages = column(&[1.0, -1.0], &device);

        let loss = clipped_policy_loss(
            current,
            actions,
            old,
            advantages,
            column(&[0.2; 2], &device),
            false,
        );

        // min(2 * 1, 1.2 * 1) = 1.2 and min(0.5 * -1, 0.8 * -1) = -0.8.
        let expected = -(1.2 - 0.8) / 2.0;
        assert!((scalar(loss.actor_loss) - expected).abs() < 1e-5);
        assert_eq!(scalar(loss.clip_fraction), 1.0);
        assert!(scalar(loss.kl) > 0.0);
    }

    #[test]
    fn normalized_entropy_of_a_uniform_policy_is_one() {
        let device = Device::flex();
        let uniform = Tensor::<2>::full([4, 6], -(6f32.ln()), &device);
        let zeros = Tensor::<2, Int>::zeros([4, 1], &device);
        let loss = clipped_policy_loss(
            uniform.clone(),
            zeros,
            Tensor::zeros([4, 1], &device),
            Tensor::zeros([4, 1], &device),
            Tensor::full([4, 1], 0.2, &device),
            true,
        );
        assert!((scalar(loss.entropy) - 1.0).abs() < 1e-5);
    }
}
