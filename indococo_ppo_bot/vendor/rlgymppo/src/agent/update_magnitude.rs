//! Device-resident parameter snapshots for update-magnitude metrics.
//!
//! The learner reports the L2 norm of each network's parameter change per
//! iteration. Upstream computed it by downloading every weight to the CPU
//! before and after the update (~60 ms per iteration for a 15M-parameter
//! actor-critic). Burn tensors are immutable and reference counted, so a
//! snapshot here is just a set of cheap handle clones; the difference and its
//! norm are reduced on the device and only a scalar crosses the bus.

use burn::nn::modules::norm::Normalization;
use burn::prelude::*;

use crate::agent::model::Net;

/// Pre-update parameter handles of one [`Net`], flattened to rank 1.
pub(crate) struct ParamSnapshot {
    params: Vec<Tensor<1>>,
}

impl ParamSnapshot {
    /// Captures the current parameters of `net` without copying them.
    pub(crate) fn capture(net: &Net) -> Self {
        Self {
            params: flat_params(net),
        }
    }

    /// Squared L2 distance between this snapshot and `net`, as a lazily
    /// evaluated one-element tensor so callers can batch the readback.
    ///
    /// # Panics
    ///
    /// Panics if `net` has a different parameter layout than the snapshot.
    pub(crate) fn squared_distance(&self, net: &Net) -> Tensor<1> {
        let after = flat_params(net);
        assert_eq!(
            self.params.len(),
            after.len(),
            "parameter layout changed between snapshot and update"
        );
        self.params
            .iter()
            .zip(after)
            .map(|(before, after)| (after - before.clone()).powi_scalar(2).sum())
            .reduce(|acc, x| acc + x)
            .expect("networks always have at least one parameter")
    }
}

/// All trainable parameters of `net`, flattened, in a stable order.
fn flat_params(net: &Net) -> Vec<Tensor<1>> {
    let mut params = Vec::new();
    for layer in net.linear_layers() {
        params.push(layer.weight.val().flatten(0, 1));
        if let Some(bias) = &layer.bias {
            params.push(bias.val());
        }
    }
    for norm in net.layer_norms() {
        match norm {
            Normalization::Layer(ln) => {
                params.push(ln.gamma.val());
                if let Some(beta) = &ln.beta {
                    params.push(beta.val());
                }
            }
            Normalization::Rms(rms) => params.push(rms.gamma.val()),
            // Batch, group and instance norms are never built by `Net`.
            _ => {}
        }
    }
    params.into_iter().map(Tensor::detach).collect()
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use burn::module::Module;

    use super::*;
    use crate::agent::{flatten_net, l2_diff};

    #[test]
    fn squared_distance_matches_host_side_l2_diff() {
        let device = Device::flex();
        let before = Net::new(5, 3, vec![8, 4], &device, None, true);
        let after = Net::new(5, 3, vec![8, 4], &device, None, true);

        let snapshot = ParamSnapshot::capture(&before);
        let device_norm = f64::from(
            snapshot
                .squared_distance(&after)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()[0],
        )
        .sqrt();
        let host_norm = l2_diff(&flatten_net(&before), &flatten_net(&after));

        assert!(
            (device_norm - host_norm).abs() <= 1e-5 * host_norm.max(1.0),
            "device {device_norm} vs host {host_norm}"
        );
    }

    #[test]
    fn unchanged_network_has_zero_distance() {
        let net = Net::new(5, 3, vec![8], &Device::flex(), None, true);
        let snapshot = ParamSnapshot::capture(&net);
        let distance = snapshot.squared_distance(&net.clone().no_grad());
        assert_eq!(distance.into_data().try_to_vec::<f32>().unwrap()[0], 0.0);
    }
}
