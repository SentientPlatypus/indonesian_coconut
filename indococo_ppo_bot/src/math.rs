//! Small vector helpers that reproduce the NumPy idioms of the Python rewards.

use glam::DVec3;

/// `_safe_norm`: the length, floored at `1e-6`.
#[must_use]
pub fn safe_norm(v: DVec3) -> f64 {
    let n = v.length();
    if n > 1e-6 { n } else { 1e-6 }
}

/// `_unit`: `v / _safe_norm(v)`.
#[must_use]
pub fn unit(v: DVec3) -> DVec3 {
    v / safe_norm(v)
}

/// The point `(0, goal_y, 0)` at the back of the net a car attacks.
#[must_use]
pub fn opp_net(attack_sign: f64) -> DVec3 {
    DVec3::new(0.0, attack_sign * crate::consts::BACK_NET_Y, 0.0)
}
