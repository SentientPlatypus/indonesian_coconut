//! Arena and game constants, copied from `rlgym.rocket_league.common_values`
//! so every reward uses exactly the numbers the Python rewards were tuned with.

pub const SIDE_WALL_X: f64 = 4096.0;
pub const BACK_WALL_Y: f64 = 5120.0;
pub const CEILING_Z: f64 = 2044.0;
pub const BACK_NET_Y: f64 = 6000.0;
pub const GOAL_HEIGHT: f64 = 642.775;
pub const GOAL_CENTER_TO_POST: f64 = 892.755;
/// `|ball.y|` above which the ball is inside a goal (`GameState.goal_scored`).
pub const GOAL_THRESHOLD: f64 = 5215.5;
pub const BALL_RADIUS: f64 = 91.25;
pub const BALL_MAX_SPEED: f64 = 6000.0;
pub const CAR_MAX_SPEED: f64 = 2300.0;
pub const CAR_MAX_ANG_VEL: f64 = 5.5;
pub const GRAVITY: f64 = 650.0;
pub const CAR_MASS: f64 = 180.0;
pub const DOUBLEJUMP_MAX_DELAY: f64 = 1.25;
pub const FLIP_TORQUE_TIME: f64 = 0.65;

/// Physics ticks per second.
pub const TICKS_PER_SECOND: f64 = 120.0;
/// Physics ticks per environment step (`RepeatAction(repeats=8)`).
pub const TICK_SKIP: u8 = 8;
/// Environment steps per second. Rewards are evaluated once per step, so a
/// reward that divides by [`TICKS_PER_SECOND`] pays 1/8 of its nominal
/// per-second rate. That dilution is what the policy was trained on and is
/// kept on purpose; rewards that divide by this constant were fixed upstream.
pub const STEPS_PER_SECOND: f64 = TICKS_PER_SECOND / TICK_SKIP as f64;

pub const BLUE_TEAM: u8 = 0;
pub const ORANGE_TEAM: u8 = 1;
