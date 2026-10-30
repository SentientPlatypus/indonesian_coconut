//! Action parser: the repo's 90-entry `LookupTableAction`, repeated for 8
//! ticks with no action delay and no masking (the Python policy was trained
//! without masks).

use rlgymppo_utils::actions::DefaultAction;
use rlgymppo_utils::rlgym::rocketsim::CarControls;
use rlgymppo_utils::rlgym::{Action, GameState};

/// Number of discrete actions.
pub const N_ACTIONS: usize = 90;

type Inner = DefaultAction<2, 8, 0>;

/// `LookupTableAction` + `RepeatAction(repeats=8)`, all actions always valid.
#[derive(Default)]
pub struct LookupNoMask(Inner);

impl LookupNoMask {
    /// Controls for one action index.
    #[must_use]
    pub fn controls(&self, action: usize) -> CarControls {
        self.0.get_action(action)
    }
}

impl<SI> Action<SI> for LookupNoMask
where
    Inner: Action<SI>,
{
    type Input = <Inner as Action<SI>>::Input;

    fn get_tick_skip() -> u8 {
        crate::consts::TICK_SKIP
    }

    fn get_action_delay() -> u8 {
        0
    }

    fn get_action_space(&self, si: &SI) -> usize {
        self.0.get_action_space(si)
    }

    fn reset(&mut self, state: &GameState, si: &mut SI) {
        self.0.reset(state, si);
    }

    fn parse_actions<'a>(
        &'a mut self,
        actions: &[Self::Input],
        state: &GameState,
        si: &'a mut SI,
    ) -> &'a [(usize, CarControls)] {
        self.0.parse_actions(actions, state, si)
    }

    fn get_action_masks(&mut self, state: &GameState, _si: &mut SI) -> Vec<Vec<bool>> {
        vec![vec![true; N_ACTIONS]; state.cars.len()]
    }
}
