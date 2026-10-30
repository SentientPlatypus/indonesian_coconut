//! `EnergyReward`: normalized "mechanical energy" of a car: height, speed,
//! boost, an unused jump and an unused flip. Zero while demoed.

use super::StepReward;
use crate::consts::{CAR_MASS, CAR_MAX_SPEED, CEILING_Z, GRAVITY};
use crate::state::StateView;

const PER_BOOST_POTENTIAL: f64 = (0.5 * CAR_MASS * 3000.0 * 3000.0) / 100.0;
const JUMP_VEL: f64 = 292.0;
const JUMP_POTENTIAL: f64 = 0.5 * CAR_MASS * (JUMP_VEL * 2.0) * (JUMP_VEL * 2.0);
const MAX_ENERGY: f64 = 100.0 * PER_BOOST_POTENTIAL
    + JUMP_POTENTIAL
    + CAR_MASS * GRAVITY * (CEILING_Z - 17.0)
    + 0.5 * CAR_MASS * CAR_MAX_SPEED * CAR_MAX_SPEED;

#[derive(Default)]
pub struct Energy;

impl StepReward for Energy {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .map(|c| {
                if c.is_demoed() {
                    return 0.0;
                }
                let speed = c.vel.length();
                let mut energy = 1.1 * CAR_MASS * GRAVITY * c.pos.z;
                energy += 0.5 * CAR_MASS * speed * speed;
                energy += PER_BOOST_POTENTIAL * c.boost;
                if !c.has_jumped {
                    energy += JUMP_POTENTIAL;
                }
                if c.has_flip() {
                    let impulse = if speed <= 1700.0 {
                        500.0 + speed / 17.0
                    } else {
                        600.0 - (speed - 1700.0)
                    };
                    let impulse = (impulse - 25.0).max(0.0);
                    energy += 0.9 * 0.5 * CAR_MASS * impulse * impulse;
                }
                energy / MAX_ENERGY
            })
            .collect()
    }
}
