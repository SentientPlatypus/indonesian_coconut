"""
Curriculum state setters for V4 (air dribbles + flip resets).

WHY THIS EXISTS:
  The default reset is KickoffMutator() — every episode starts as a standard
  kickoff. Air dribbles and (especially) flip resets require very specific
  mid-air situations the bot almost never reaches on its own from a kickoff, so
  AirdribbleReward / FlipResetReward almost never fire and therefore can't teach
  the mechanic. This mutator RESETS some episodes directly into those situations
  so the rewards fire constantly and PPO can reinforce them.

THE MIX (curriculum):
  - kickoff   : real-game play, keeps scoring / fundamentals sharp (don't forget!)
  - wall_pop  : ball rolling toward a side wall, car chasing it grounded — the
                ENTRY to an air dribble from a state the bot reaches in real
                games (drive up wall -> pop -> get under). Without this bridge
                the mid-air setups never transfer to kickoff play.
  - air_dribble: ball popped low-to-mid, drifting goalward, car under/behind it
  - flip_reset : ball high, car spawned below it airborne with boost, no flip

SAFETY NOTES:
  - Kept kickoff-heavy by default so the resumed 47-3 policy isn't destabilized.
  - All spawned states are PHYSICALLY VALID values that occur in normal play, so
    the worst case for a flip_reset spawn is that RocketSim doesn't grant the
    reset and the episode is just a high-ball aerial drill (still useful) — not a
    crash or a wasted episode.
  - EXPERIMENTAL: whether RocketSim grants the flip reset from a hand-set state
    (has_flip made False via the air_time_since_jump window) is unverified
    locally (no sim on WSL). Watch the FlipReset reward channel on EC2; if it
    stays flat, the reset isn't being granted and only the air-dribble setups are
    doing work — lower flip_reset_w or set it to 0.

To disable entirely: in freestyler_v4.py set USE_CURRICULUM = False (kickoff-only).
"""
import random
from typing import Dict, Any

import numpy as np

from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import KickoffMutator
from rlgym.rocket_league.common_values import (
    BLUE_TEAM,
    BACK_NET_Y,
    DOUBLEJUMP_MAX_DELAY,
)


def _f32(*xyz) -> np.ndarray:
    return np.array(xyz, dtype=np.float32)


class CurriculumStateMutator(StateMutator[GameState]):
    """
    Picks a reset type per episode by weight. Designed for 1v1 (one attacker, one
    defender) but degrades gracefully: the first car on the attacking team gets
    the setup, everyone else is parked defensively near their own net.

    Weights are normalized, so they don't have to sum to 1.
    """

    def __init__(self, kickoff_w: float = 0.50, air_dribble_w: float = 0.30,
                 flip_reset_w: float = 0.20, wall_pop_w: float = 0.0,
                 ground_dribble_w: float = 0.0):
        total = kickoff_w + air_dribble_w + flip_reset_w + wall_pop_w + ground_dribble_w
        assert total > 0, "curriculum weights must sum to > 0"
        self.kickoff_w = kickoff_w / total
        self.air_dribble_w = air_dribble_w / total
        self.flip_reset_w = flip_reset_w / total
        self.wall_pop_w = wall_pop_w / total
        self.ground_dribble_w = ground_dribble_w / total
        self._kickoff = KickoffMutator()

    # -- helpers --------------------------------------------------------------
    def _attack_dir(self, team_num: int) -> float:
        """+1 if this team attacks +Y (blue), -1 if it attacks -Y (orange)."""
        return 1.0 if team_num == BLUE_TEAM else -1.0

    def _park_defender(self, car) -> None:
        """Put a car back near its OWN net, grounded, facing upfield."""
        attack = self._attack_dir(car.team_num)
        own_goal_y = -attack * BACK_NET_Y * 0.82  # own net is opposite the attack dir
        car.physics.position = _f32(random.uniform(-900, 900), own_goal_y, 17.0)
        car.physics.linear_velocity = _f32(0, 0, 0)
        car.physics.angular_velocity = _f32(0, 0, 0)
        # face upfield (toward attack dir): yaw +pi/2 faces +Y, -pi/2 faces -Y
        car.physics.euler_angles = _f32(0.0, attack * np.pi / 2.0, 0.0)
        car.boost_amount = random.uniform(34.0, 60.0)
        car.on_ground = True

    def _split_cars(self, state: GameState):
        """Pick one attacker (random team) and return (attacker, [defenders])."""
        cars = list(state.cars.values())
        attack_team = random.choice([BLUE_TEAM, 1 - BLUE_TEAM])
        attackers = [c for c in cars if c.team_num == attack_team]
        if not attackers:                       # team not present; fall back
            attackers = cars
        attacker = random.choice(attackers)
        defenders = [c for c in cars if c is not attacker]
        return attacker, defenders

    # -- setups ---------------------------------------------------------------
    def _air_dribble_setup(self, state: GameState) -> None:
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)

        # Ball: popped to mid height, drifting up and toward the attacking net.
        bx = random.uniform(-1400, 1400)
        by = random.uniform(-2200, 2200)
        bz = random.uniform(320, 680)
        up_v = random.uniform(280, 560)
        fwd_v = attack * random.uniform(250, 650)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(random.uniform(-120, 120), fwd_v, up_v)
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Car: just under and slightly behind the ball, moving with it (low rel speed).
        cx = bx + random.uniform(-80, 80)
        cy = by - attack * random.uniform(60, 170)
        cz = max(80.0, bz - random.uniform(120, 220))
        attacker.physics.position = _f32(cx, cy, cz)
        attacker.physics.linear_velocity = _f32(
            random.uniform(-80, 80),
            fwd_v * random.uniform(0.7, 1.0),
            up_v * random.uniform(0.5, 0.9),
        )
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        # face the attacking net, nose slightly up
        attacker.physics.euler_angles = _f32(random.uniform(0.10, 0.40), attack * np.pi / 2.0, 0.0)
        attacker.boost_amount = random.uniform(55, 100)
        attacker.on_ground = False
        attacker.has_jumped = True            # in the air after a jump
        attacker.has_flipped = False
        attacker.has_double_jumped = False
        attacker.air_time_since_jump = 0.10   # flip still available for the dribble flick

        for d in defenders:
            self._park_defender(d)

    def _wall_pop_setup(self, state: GameState) -> None:
        """The air-dribble ENTRY: ball rolling toward a side wall, attacker
        chasing it grounded with boost. Unlike the mid-air setups, this is a
        state the bot reaches constantly in kickoff play, so the learned
        behavior (carry up wall -> pop infield -> get under) transfers."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)

        # Ball: on the ground in the attacking half-ish, rolling toward a side wall.
        wall_sign = random.choice([-1.0, 1.0])
        bx = wall_sign * random.uniform(1800, 3100)          # partway to the wall
        by = attack * random.uniform(-800, 2600)             # mostly attacking half
        state.ball.position = _f32(bx, by, 93.15)
        state.ball.linear_velocity = _f32(
            wall_sign * random.uniform(700, 1400),           # toward the wall
            attack * random.uniform(100, 700),               # drifting downfield
            0.0,
        )
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Car: grounded behind the ball, chasing it with speed + boost.
        chase_dir = _f32(wall_sign * random.uniform(0.55, 0.9),
                         attack * random.uniform(0.1, 0.5), 0.0)
        chase_dir = chase_dir / np.linalg.norm(chase_dir)
        back = random.uniform(700, 1400)
        attacker.physics.position = _f32(
            float(np.clip(bx - chase_dir[0] * back, -3900, 3900)),
            float(np.clip(by - chase_dir[1] * back, -4900, 4900)),
            17.0,
        )
        speed = random.uniform(900, 1700)
        attacker.physics.linear_velocity = _f32(chase_dir[0] * speed, chase_dir[1] * speed, 0.0)
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        # euler order is (pitch, yaw, roll); yaw 0 faces +X, pi/2 faces +Y
        yaw = float(np.arctan2(chase_dir[1], chase_dir[0]))
        attacker.physics.euler_angles = _f32(0.0, yaw, 0.0)
        attacker.boost_amount = random.uniform(65, 100)
        attacker.on_ground = True

        for d in defenders:
            self._park_defender(d)

    def _flip_reset_setup(self, state: GameState) -> None:
        attacker, defenders = self._split_cars(state)

        # Ball: high and roughly hovering / slowly falling.
        bx = random.uniform(-1300, 1300)
        by = random.uniform(-2400, 2400)
        bz = random.uniform(1100, 1500)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(random.uniform(-90, 90), random.uniform(-90, 90), random.uniform(-120, 60))
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Car: below the ball, rising toward it, full boost, oriented wheels-up
        # (roll ~ pi) so the underside faces the ball -> a reset is reachable.
        cx = bx + random.uniform(-110, 110)
        cy = by + random.uniform(-110, 110)
        cz = max(120.0, bz - random.uniform(360, 640))
        attacker.physics.position = _f32(cx, cy, cz)
        attacker.physics.linear_velocity = _f32(random.uniform(-90, 90), random.uniform(-90, 90), random.uniform(420, 780))
        attacker.physics.angular_velocity = _f32(random.uniform(-0.6, 0.6), random.uniform(-0.6, 0.6), random.uniform(-0.6, 0.6))
        attacker.physics.euler_angles = _f32(
            random.uniform(-0.35, 0.35),
            random.uniform(-np.pi, np.pi),
            np.pi + random.uniform(-0.35, 0.35),   # upside-down: wheels point up at the ball
        )
        attacker.boost_amount = 100.0
        attacker.on_ground = False
        # Make has_flip == False so wheel-on-ball contact can GRANT a reset.
        # has_flip == (not has_double_jumped and not has_flipped and air_time_since_jump < DOUBLEJUMP_MAX_DELAY)
        attacker.has_jumped = True
        attacker.has_flipped = False
        attacker.has_double_jumped = False
        attacker.air_time_since_jump = DOUBLEJUMP_MAX_DELAY + 0.25   # past the window -> no flip available

        for d in defenders:
            self._park_defender(d)

    def _active_defender(self, car, bx: float, by: float, attack: float) -> None:
        """Position a car as an ACTIVE defender (not parked at the net): goal-side
        of the ball — between the ball and the net the attacker is attacking — at a
        challenging gap, grounded, facing back toward the oncoming attacker. `attack`
        is the ATTACKER's attack dir; the defended net is at attack*BACK_NET_Y."""
        # goal-side of the ball, ahead toward the defended net but in front of it
        def_y = by + attack * random.uniform(1300, 3300)
        def_y = float(np.clip(def_y, -BACK_NET_Y + 400.0, BACK_NET_Y - 400.0))
        def_x = float(np.clip(bx + random.uniform(-1000, 1000), -3500, 3500))
        car.physics.position = _f32(def_x, def_y, 17.0)
        # a little closing speed toward the ball sometimes (challenge vs. contain)
        close = random.uniform(0.0, 700.0)
        car.physics.linear_velocity = _f32(0.0, -attack * close, 0.0)
        car.physics.angular_velocity = _f32(0, 0, 0)
        # face the oncoming attacker (toward -attack in Y): yaw -attack*pi/2
        car.physics.euler_angles = _f32(0.0, -attack * np.pi / 2.0, 0.0)
        car.boost_amount = random.uniform(30.0, 70.0)
        car.on_ground = True

    def _ground_dribble_setup(self, state: GameState) -> None:
        """Attacker ground-dribbling the ball toward the attacking net, with the
        other car placed as an ACTIVE defender goal-side of the ball. Because the
        attacker is chosen from a random team, our policy trains BOTH dribble
        offense (carry the ball to net past a defender) and dribble defense
        (challenge/contain an incoming ball-carrier)."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)

        # Attacker in own half / midfield, carrying the ball toward the net.
        bx = random.uniform(-1500, 1500)
        by = attack * random.uniform(-2400, 300)        # own half to just past midfield
        carry_speed = random.uniform(650, 1300)         # rolling toward the net

        # Ball resting on the hood/roof, moving with the car (low rel speed), gentle bounce.
        state.ball.position = _f32(
            bx, by + attack * random.uniform(40, 120), random.uniform(150, 240),
        )
        state.ball.linear_velocity = _f32(
            random.uniform(-90, 90), attack * carry_speed, random.uniform(-40, 130),
        )
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Attacker just under/behind the ball, grounded, moving with it toward net.
        attacker.physics.position = _f32(
            bx + random.uniform(-70, 70), by - attack * random.uniform(20, 130), 17.0,
        )
        attacker.physics.linear_velocity = _f32(
            random.uniform(-70, 70), attack * carry_speed * random.uniform(0.9, 1.05), 0.0,
        )
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        attacker.physics.euler_angles = _f32(0.0, attack * np.pi / 2.0, 0.0)   # face the net
        attacker.boost_amount = random.uniform(30, 80)
        attacker.on_ground = True

        for d in defenders:
            self._active_defender(d, bx, by, attack)

    # -- entry point ----------------------------------------------------------
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        r = random.random()
        if r < self.kickoff_w:
            self._kickoff.apply(state, shared_info)
        elif r < self.kickoff_w + self.wall_pop_w:
            self._wall_pop_setup(state)
        elif r < self.kickoff_w + self.wall_pop_w + self.air_dribble_w:
            self._air_dribble_setup(state)
        elif r < self.kickoff_w + self.wall_pop_w + self.air_dribble_w + self.ground_dribble_w:
            self._ground_dribble_setup(state)
        else:
            self._flip_reset_setup(state)
