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
                 (half of these draws are mid-carry FR: air-dribble geometry with
                 no flip left + slight wheels-up roll — teaches reset-then-power
                 during a carry, not just isolated hover resets)
  - double_tap : ball heading into the attacking backboard. Half from an
                 air-dribble carry (car under the ball airborne), half from the
                 ground (car reading the bounce). Finish is the rebound tap.
  - wall_leak  : last man rotating back to net; attacker is BEHIND them on the
                 side wall with the ball, ready to flick from the outside. The
                 Nexto leak — peeling to the corner pad and leaving the far
                 post open.
  - awkward_ball: loose high balls we lose in-game — above the crossbar right
                 over us (finish or challenge), 50/50 both on the ground with
                 the ball above, and wheels-not-down recoveries.

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
    BACK_WALL_Y,
    DOUBLEJUMP_MAX_DELAY,
    GOAL_HEIGHT,
    SIDE_WALL_X,
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
                 ground_dribble_w: float = 0.0, ground_to_air_w: float = 0.0,
                 aerial_front_bump_w: float = 0.0,
                 double_tap_w: float = 0.0,
                 wall_leak_w: float = 0.0,
                 awkward_ball_w: float = 0.0,
                 # v10: how the flip_reset mass splits across the three FR stages.
                 # easy   = static hover, wheels already up (pure mechanic drill)
                 # mid    = mid-carry, half-rolled, no flip left
                 # natural= genuine air-dribble carry vs an ACTIVE defender; the
                 #          bot must create the reset itself. Shift mass toward
                 #          `natural` as the mechanic lands.
                 fr_easy_frac: float = 0.25,
                 fr_mid_frac: float = 0.35):
        total = (kickoff_w + air_dribble_w + flip_reset_w + wall_pop_w
                 + ground_dribble_w + ground_to_air_w + aerial_front_bump_w
                 + double_tap_w + wall_leak_w + awkward_ball_w)
        assert total > 0, "curriculum weights must sum to > 0"
        self.kickoff_w = kickoff_w / total
        self.air_dribble_w = air_dribble_w / total
        self.flip_reset_w = flip_reset_w / total
        self.wall_pop_w = wall_pop_w / total
        self.ground_dribble_w = ground_dribble_w / total
        self.ground_to_air_w = ground_to_air_w / total
        self.aerial_front_bump_w = aerial_front_bump_w / total
        self.double_tap_w = double_tap_w / total
        self.wall_leak_w = wall_leak_w / total
        self.awkward_ball_w = awkward_ball_w / total
        self.fr_easy_frac = max(0.0, min(1.0, fr_easy_frac))
        self.fr_mid_frac = max(0.0, min(1.0 - self.fr_easy_frac, fr_mid_frac))
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

    def _air_dribble_flip_reset_setup(self, state: GameState) -> None:
        """Mid-carry flip-reset drill: air-dribble geometry (ball drifting
        goalward, car under/behind it) but the attacker has NO flip left and a
        slight wheels-up roll so a reset-on-ball is the natural next skill —
        then a powered follow-up hit. Transfers better than the static hover FR
        spawn because the bot already reaches this state from air dribbles."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)

        # Ball: mid-to-high air-dribble height, drifting goalward (carry in progress).
        bx = random.uniform(-1400, 1400)
        by = random.uniform(-2000, 2200)
        bz = random.uniform(520, 980)
        up_v = random.uniform(180, 420)
        fwd_v = attack * random.uniform(280, 700)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(random.uniform(-100, 100), fwd_v, up_v)
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Car: under/behind the ball (carry pose) but rolled toward wheels-up and
        # past the double-jump window so has_flip is False — must reset to flick.
        cx = bx + random.uniform(-90, 90)
        cy = by - attack * random.uniform(40, 140)
        cz = max(100.0, bz - random.uniform(140, 280))
        attacker.physics.position = _f32(cx, cy, cz)
        attacker.physics.linear_velocity = _f32(
            random.uniform(-80, 80),
            fwd_v * random.uniform(0.65, 1.0),
            up_v * random.uniform(0.4, 0.85),
        )
        attacker.physics.angular_velocity = _f32(
            random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5),
        )
        # Face net, pitched up a bit, rolled ~halfway toward inverted so wheels
        # can find the ball without being a pure hover drill.
        attacker.physics.euler_angles = _f32(
            random.uniform(0.05, 0.35),
            attack * np.pi / 2.0,
            random.uniform(1.2, 2.0) * random.choice([-1.0, 1.0]),  # ~70–115° roll
        )
        attacker.boost_amount = random.uniform(60, 100)
        attacker.on_ground = False
        attacker.has_jumped = True
        attacker.has_flipped = False
        attacker.has_double_jumped = False
        attacker.air_time_since_jump = DOUBLEJUMP_MAX_DELAY + 0.25  # no flip

        for d in defenders:
            self._park_defender(d)

    def _natural_flip_reset_setup(self, state: GameState) -> None:
        """Hardest FR stage: a REAL air-dribble carry where a reset is available.

        v10 (user): the end goal is not an isolated trick from an artificial pose —
        it's the bot recognising "I already have control of this aerial, I can get
        underneath here, take my flip, and use it". So this spawns proper
        air-dribble geometry (upright-ish, ball on the roof, drifting goalward),
        with the flip ALREADY SPENT — which is the real in-game precondition, since
        a car that double-jumped for height has no flip left. The car is NOT
        pre-rolled toward wheels-up: it must rotate under the ball itself. A live
        defender is goal-side, so the reset has a purpose.
        """
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)

        # Ball: solid air-dribble height, carrying goalward with a little lift.
        bx = random.uniform(-1500, 1500)
        by = attack * random.uniform(-1800, 1200)
        bz = random.uniform(700, 1250)
        up_v = random.uniform(120, 380)
        fwd_v = attack * random.uniform(350, 800)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(random.uniform(-90, 90), fwd_v, up_v)
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Car: under the ball in a normal carry pose, matching its motion.
        cx = bx + random.uniform(-70, 70)
        cy = by - attack * random.uniform(30, 120)
        cz = max(120.0, bz - random.uniform(180, 320))
        attacker.physics.position = _f32(cx, cy, cz)
        attacker.physics.linear_velocity = _f32(
            random.uniform(-70, 70),
            fwd_v * random.uniform(0.8, 1.0),
            up_v * random.uniform(0.6, 1.0),
        )
        attacker.physics.angular_velocity = _f32(
            random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),
        )
        # Upright carry pose (small roll only) — the bot must invert itself.
        attacker.physics.euler_angles = _f32(
            random.uniform(0.05, 0.35),
            attack * np.pi / 2.0,
            random.uniform(-0.45, 0.45),
        )
        attacker.boost_amount = random.uniform(45, 100)
        attacker.on_ground = False
        # Flip already spent (the double-jump-for-height case) -> reset is live.
        attacker.has_jumped = True
        attacker.has_flipped = False
        attacker.has_double_jumped = True
        attacker.air_time_since_jump = DOUBLEJUMP_MAX_DELAY + 0.25

        for d in defenders:
            self._active_defender(d, bx, by, attack)

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

    def _ground_to_air_setup(self, state: GameState) -> None:
        """v7 (user): BOTH cars grounded with a contestable ball on the ground and
        the defender set goal-side — the exact 'we're both on the ground vs a
        defender' spot where just being fast into a flat ground shot loses. The
        good play is to POP the ball up and follow it into an AERIAL play toward
        goal. Attacker is grounded, approaching with boost (enough to go up); the
        aerial follow-up is rewarded by aerial_boost / AirdribbleReward / GoalProb."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)

        # Ball on the ground, midfield to attacking half, slow (a 50-50 / loose ball).
        bx = random.uniform(-1500, 1500)
        by = attack * random.uniform(-300, 2400)
        state.ball.position = _f32(bx, by, 93.15)
        state.ball.linear_velocity = _f32(
            random.uniform(-150, 150), attack * random.uniform(-150, 300), 0.0,
        )
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Attacker grounded, behind the ball, closing with boost so an aerial pop
        # is the natural next move (rather than a flat ground poke into the wall).
        back = random.uniform(500, 1100)
        attacker.physics.position = _f32(
            float(np.clip(bx + random.uniform(-300, 300), -3500, 3500)),
            by - attack * back, 17.0,
        )
        speed = random.uniform(500, 1150)
        attacker.physics.linear_velocity = _f32(random.uniform(-80, 80), attack * speed, 0.0)
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        attacker.physics.euler_angles = _f32(0.0, attack * np.pi / 2.0, 0.0)   # face the net
        attacker.boost_amount = random.uniform(45, 100)                        # boost to go up
        attacker.on_ground = True

        for d in defenders:
            self._active_defender(d, bx, by, attack)

    # -- entry point ----------------------------------------------------------
    def _aerial_front_bump_setup(self, state: GameState) -> None:
        """Air-dribble carry meeting a close challenger: attacker is BEHIND the
        ball with boost; defender is already goal-side of it. The correct play
        is leave the ball, boost in FRONT, and knock the defender away — not
        bump them while still carrying (which dumps the ball off course)."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)

        bx = random.uniform(-1200, 1200)
        by = attack * random.uniform(-400, 2800)
        bz = random.uniform(380, 780)
        fwd_v = attack * random.uniform(450, 950)
        up_v = random.uniform(80, 360)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(random.uniform(-80, 80), fwd_v, up_v)
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Attacker: under/behind the ball, matching its motion, lots of boost.
        cx = bx + random.uniform(-60, 60)
        cy = by - attack * random.uniform(50, 160)
        cz = max(100.0, bz - random.uniform(140, 260))
        attacker.physics.position = _f32(cx, cy, cz)
        attacker.physics.linear_velocity = _f32(
            random.uniform(-70, 70),
            fwd_v * random.uniform(0.75, 1.0),
            up_v * random.uniform(0.5, 0.9),
        )
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        attacker.physics.euler_angles = _f32(
            random.uniform(0.08, 0.35), attack * np.pi / 2.0, 0.0)
        attacker.boost_amount = random.uniform(70, 100)
        attacker.on_ground = False
        attacker.has_jumped = True
        attacker.has_flipped = False
        attacker.has_double_jumped = False
        attacker.air_time_since_jump = 0.12

        for d in defenders:
            # Close challenger already in FRONT of the ball (between ball and net).
            def_y = by + attack * random.uniform(180, 520)
            def_y = float(np.clip(def_y, -BACK_NET_Y + 400.0, BACK_NET_Y - 400.0))
            def_x = float(np.clip(bx + random.uniform(-280, 280), -3500, 3500))
            def_z = random.uniform(80, 420)
            d.physics.position = _f32(def_x, def_y, def_z)
            d.physics.linear_velocity = _f32(
                0.0, -attack * random.uniform(80, 500), random.uniform(0, 250))
            d.physics.angular_velocity = _f32(0, 0, 0)
            d.physics.euler_angles = _f32(0.0, -attack * np.pi / 2.0, 0.0)
            d.boost_amount = random.uniform(20.0, 60.0)
            d.on_ground = def_z < 30.0
            d.has_jumped = True
            d.has_flipped = False
            d.has_double_jumped = False

    def _double_tap_aerial_setup(self, state: GameState) -> None:
        """Air-dribble carry into the attacking backboard: ball is about to
        hit the wall, car is under/behind it airborne. Finish is a rebound
        tap into the net, not another carry across the field."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)
        wall_y = attack * BACK_WALL_Y

        bx = random.uniform(-1000, 1000)
        # Short of the backboard so the first touch is the wall, not the net.
        by = wall_y - attack * random.uniform(700, 1900)
        bz = random.uniform(380, 880)
        fwd_v = attack * random.uniform(750, 1450)
        up_v = random.uniform(-60, 380)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(random.uniform(-180, 180), fwd_v, up_v)
        state.ball.angular_velocity = _f32(0, 0, 0)

        cx = bx + random.uniform(-70, 70)
        cy = by - attack * random.uniform(50, 160)
        cz = max(110.0, bz - random.uniform(130, 250))
        attacker.physics.position = _f32(cx, cy, cz)
        attacker.physics.linear_velocity = _f32(
            random.uniform(-70, 70),
            fwd_v * random.uniform(0.75, 1.0),
            up_v * random.uniform(0.45, 0.95),
        )
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        attacker.physics.euler_angles = _f32(
            random.uniform(0.08, 0.38), attack * np.pi / 2.0, 0.0)
        attacker.boost_amount = random.uniform(45, 100)
        attacker.on_ground = False
        attacker.has_jumped = True
        attacker.has_flipped = False
        attacker.has_double_jumped = False
        attacker.air_time_since_jump = 0.12

        for d in defenders:
            self._active_defender(d, bx, by, attack)

    def _double_tap_ground_setup(self, state: GameState) -> None:
        """Grounded read on a ball already heading into the attacking
        backboard. Jump the rebound in — not a slow ground dribble to the
        corner, not a full-field air dribble."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)
        wall_y = attack * BACK_WALL_Y

        bx = random.uniform(-1200, 1200)
        by = wall_y - attack * random.uniform(900, 2300)
        bz = random.uniform(220, 640)
        fwd_v = attack * random.uniform(900, 1700)
        up_v = random.uniform(160, 560)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(random.uniform(-220, 220), fwd_v, up_v)
        state.ball.angular_velocity = _f32(0, 0, 0)

        back = random.uniform(550, 1500)
        attacker.physics.position = _f32(
            float(np.clip(bx + random.uniform(-450, 450), -3500, 3500)),
            float(np.clip(by - attack * back, -BACK_WALL_Y + 400.0, BACK_WALL_Y - 400.0)),
            17.0,
        )
        speed = random.uniform(700, 1600)
        attacker.physics.linear_velocity = _f32(
            random.uniform(-120, 120), attack * speed, 0.0)
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        attacker.physics.euler_angles = _f32(0.0, attack * np.pi / 2.0, 0.0)
        attacker.boost_amount = random.uniform(40, 90)
        attacker.on_ground = True

        for d in defenders:
            self._active_defender(d, bx, by, attack)

    def _double_tap_setup(self, state: GameState) -> None:
        if random.random() < 0.5:
            self._double_tap_aerial_setup(state)
        else:
            self._double_tap_ground_setup(state)

    def _wall_leak_setup(self, state: GameState) -> None:
        """Last man rotating back; attacker is BEHIND them on the side wall
        with the ball, ready to flick from the outside into the open net.

        In-game (user vs Nexto): we peel to the corner pad and they flick
        across the face. Defender is spawned already going back — often
        committed toward the pad, low boost — so staying on the goal-ball
        line is the play, not collecting."""
        attacker, defenders = self._split_cars(state)
        attack = self._attack_dir(attacker.team_num)
        wall_sign = random.choice([-1.0, 1.0])

        # Ball: wide, flick height, cutting in toward the far/center of the net.
        bx = wall_sign * random.uniform(2400.0, 3550.0)
        by = attack * random.uniform(-600.0, 2100.0)
        bz = random.uniform(130.0, 280.0)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(
            -wall_sign * random.uniform(60.0, 420.0),
            attack * random.uniform(700.0, 1450.0),
            random.uniform(40.0, 280.0),
        )
        state.ball.angular_velocity = _f32(0, 0, 0)

        # Attacker: under/behind the ball, even closer to the wall than it is.
        attacker.physics.position = _f32(
            float(np.clip(bx + wall_sign * random.uniform(40.0, 220.0),
                          -SIDE_WALL_X + 250.0, SIDE_WALL_X - 250.0)),
            float(np.clip(by - attack * random.uniform(30.0, 160.0),
                          -BACK_WALL_Y + 400.0, BACK_WALL_Y - 400.0)),
            17.0,
        )
        carry = random.uniform(750.0, 1400.0)
        attacker.physics.linear_velocity = _f32(
            -wall_sign * random.uniform(20.0, 280.0),
            attack * carry,
            0.0,
        )
        attacker.physics.angular_velocity = _f32(0, 0, 0)
        attacker.physics.euler_angles = _f32(0.0, attack * np.pi / 2.0, 0.0)
        attacker.boost_amount = random.uniform(35.0, 80.0)
        attacker.on_ground = True

        # Corner pad on the defended end (same side as the wall carry).
        pad_x = wall_sign * 3072.0
        pad_y = attack * 4096.0

        for d in defenders:
            # Goal-side of the ball, already retreating toward own net.
            def_y = float(np.clip(
                by + attack * random.uniform(1400.0, 2800.0),
                -BACK_WALL_Y + 450.0, BACK_WALL_Y - 450.0,
            ))
            if random.random() < 0.65:
                # Already peeling toward the corner pad — the failure pose.
                def_x = float(np.clip(
                    wall_sign * random.uniform(2300.0, 3400.0),
                    -SIDE_WALL_X + 250.0, SIDE_WALL_X - 250.0,
                ))
                to_pad = _f32(pad_x - def_x, pad_y - def_y, 0.0)
                to_pad = to_pad / max(float(np.linalg.norm(to_pad)), 1e-6)
            else:
                # More central, still going back — stay on the shot line.
                def_x = float(np.clip(
                    wall_sign * random.uniform(-200.0, 1100.0),
                    -SIDE_WALL_X + 250.0, SIDE_WALL_X - 250.0,
                ))
                to_pad = _f32(0.0, attack, 0.0)
            speed = random.uniform(900.0, 1700.0)
            d.physics.position = _f32(def_x, def_y, 17.0)
            d.physics.linear_velocity = _f32(
                to_pad[0] * speed, to_pad[1] * speed, 0.0)
            d.physics.angular_velocity = _f32(0, 0, 0)
            d.physics.euler_angles = _f32(
                0.0, float(np.arctan2(to_pad[1], to_pad[0])), 0.0)
            d.boost_amount = random.uniform(10.0, 36.0)
            d.on_ground = True

    def _awkward_crossbar_setup(self, state: GameState, *, own_net: bool) -> None:
        """Ball above the crossbar, right over the hero. own_net=True is a
        last-man challenge/save; False is a dunk / finish on their bar."""
        hero, others = self._split_cars(state)
        attack = self._attack_dir(hero.team_num)
        # Toward the net the ball is hanging over.
        net_sign = -attack if own_net else attack
        bx = random.uniform(-700.0, 700.0)
        by = net_sign * (BACK_WALL_Y - random.uniform(80.0, 720.0))
        bz = random.uniform(GOAL_HEIGHT + 40.0, GOAL_HEIGHT + 480.0)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(
            random.uniform(-180.0, 180.0),
            net_sign * random.uniform(-80.0, 220.0),
            random.uniform(-320.0, 80.0),
        )
        state.ball.angular_velocity = _f32(0, 0, 0)

        hero.physics.position = _f32(
            bx + random.uniform(-180.0, 180.0),
            float(np.clip(by - net_sign * random.uniform(40.0, 280.0),
                          -BACK_WALL_Y + 350.0, BACK_WALL_Y - 350.0)),
            17.0,
        )
        hero.physics.linear_velocity = _f32(
            random.uniform(-200.0, 200.0),
            net_sign * random.uniform(-80.0, 400.0),
            0.0,
        )
        hero.physics.angular_velocity = _f32(0, 0, 0)
        hero.physics.euler_angles = _f32(0.0, net_sign * np.pi / 2.0, 0.0)
        hero.boost_amount = random.uniform(35.0, 90.0)
        hero.on_ground = True
        hero.has_jumped = False
        hero.has_flipped = False
        hero.has_double_jumped = False

        for o in others:
            # Late / off to the side — hero has to go up, not wait.
            o.physics.position = _f32(
                float(np.clip(bx + random.uniform(500.0, 1600.0) * random.choice([-1.0, 1.0]),
                              -3500.0, 3500.0)),
                float(np.clip(by - net_sign * random.uniform(700.0, 2200.0),
                              -BACK_WALL_Y + 400.0, BACK_WALL_Y - 400.0)),
                17.0,
            )
            o.physics.linear_velocity = _f32(
                random.uniform(-250.0, 250.0),
                net_sign * random.uniform(200.0, 900.0),
                0.0,
            )
            o.physics.angular_velocity = _f32(0, 0, 0)
            o.physics.euler_angles = _f32(0.0, net_sign * np.pi / 2.0, 0.0)
            o.boost_amount = random.uniform(20.0, 70.0)
            o.on_ground = True

    def _awkward_fifty_setup(self, state: GameState) -> None:
        """Both cars on the ground, ball hanging above — first one up owns it."""
        hero, others = self._split_cars(state)
        bx = random.uniform(-2200.0, 2200.0)
        by = random.uniform(-2800.0, 2800.0)
        bz = random.uniform(380.0, 920.0)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(
            random.uniform(-220.0, 220.0),
            random.uniform(-220.0, 220.0),
            random.uniform(-380.0, 120.0),
        )
        state.ball.angular_velocity = _f32(0, 0, 0)

        cars = [hero] + list(others)
        base_ang = random.uniform(0.0, 2.0 * np.pi)
        for i, car in enumerate(cars):
            # Opposite sides of the ball so both can go up; not a free ball.
            ang = base_ang if i == 0 else (base_ang + np.pi + random.uniform(-0.45, 0.45))
            # Keep both in a similar ring so it's a real 50/50, not a free ball.
            dist = random.uniform(350.0, 1100.0)
            cx = float(np.clip(bx + dist * np.cos(ang), -SIDE_WALL_X + 300.0, SIDE_WALL_X - 300.0))
            cy = float(np.clip(by + dist * np.sin(ang), -BACK_WALL_Y + 400.0, BACK_WALL_Y - 400.0))
            car.physics.position = _f32(cx, cy, 17.0)
            # Face the ball, some closing speed, jump still available.
            to_ball = _f32(bx - cx, by - cy, 0.0)
            to_ball = to_ball / max(float(np.linalg.norm(to_ball)), 1e-6)
            speed = random.uniform(200.0, 1100.0)
            car.physics.linear_velocity = _f32(to_ball[0] * speed, to_ball[1] * speed, 0.0)
            car.physics.angular_velocity = _f32(0, 0, 0)
            car.physics.euler_angles = _f32(
                0.0, float(np.arctan2(to_ball[1], to_ball[0])), 0.0)
            car.boost_amount = random.uniform(25.0, 80.0)
            car.on_ground = True
            car.has_jumped = False
            car.has_flipped = False
            car.has_double_jumped = False

    def _awkward_recovery_setup(self, state: GameState) -> None:
        """Hero landing on the side/back with a loose ball above — get wheels
        down (or go up from the recovery) and take possession."""
        hero, others = self._split_cars(state)
        bx = random.uniform(-2000.0, 2000.0)
        by = random.uniform(-2400.0, 2400.0)
        bz = random.uniform(320.0, 880.0)
        state.ball.position = _f32(bx, by, bz)
        state.ball.linear_velocity = _f32(
            random.uniform(-280.0, 280.0),
            random.uniform(-280.0, 280.0),
            random.uniform(-240.0, 160.0),
        )
        state.ball.angular_velocity = _f32(0, 0, 0)

        hero.physics.position = _f32(
            float(np.clip(bx + random.uniform(-500.0, 500.0),
                          -SIDE_WALL_X + 300.0, SIDE_WALL_X - 300.0)),
            float(np.clip(by + random.uniform(-500.0, 500.0),
                          -BACK_WALL_Y + 400.0, BACK_WALL_Y - 400.0)),
            random.uniform(40.0, 120.0),
        )
        hero.physics.linear_velocity = _f32(
            random.uniform(-400.0, 400.0),
            random.uniform(-400.0, 400.0),
            random.uniform(-250.0, 80.0),
        )
        hero.physics.angular_velocity = _f32(
            random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2),
        )
        # Wheels not down: side or back landing.
        hero.physics.euler_angles = _f32(
            random.uniform(-0.4, 0.4),
            random.uniform(-np.pi, np.pi),
            random.choice([-1.0, 1.0]) * random.uniform(1.1, np.pi),
        )
        hero.boost_amount = random.uniform(20.0, 70.0)
        hero.on_ground = False
        hero.has_jumped = True
        hero.has_flipped = False
        hero.has_double_jumped = False
        hero.air_time_since_jump = random.uniform(0.15, 0.55)

        for o in others:
            self._active_defender(o, bx, by, self._attack_dir(hero.team_num))

    def _awkward_ball_setup(self, state: GameState) -> None:
        r = random.random()
        if r < 0.35:
            self._awkward_crossbar_setup(state, own_net=False)
        elif r < 0.55:
            self._awkward_crossbar_setup(state, own_net=True)
        elif r < 0.80:
            self._awkward_fifty_setup(state)
        else:
            self._awkward_recovery_setup(state)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        r = random.random()
        acc = 0.0
        for weight, fn in (
            (self.kickoff_w, lambda: self._kickoff.apply(state, shared_info)),
            (self.wall_pop_w, lambda: self._wall_pop_setup(state)),
            (self.air_dribble_w, lambda: self._air_dribble_setup(state)),
            (self.ground_dribble_w, lambda: self._ground_dribble_setup(state)),
            (self.ground_to_air_w, lambda: self._ground_to_air_setup(state)),
            (self.aerial_front_bump_w, lambda: self._aerial_front_bump_setup(state)),
            (self.double_tap_w, lambda: self._double_tap_setup(state)),
            (self.wall_leak_w, lambda: self._wall_leak_setup(state)),
            (self.awkward_ball_w, lambda: self._awkward_ball_setup(state)),
        ):
            acc += weight
            if r < acc:
                fn()
                return
        # Remainder is flip-reset: easy -> mid -> natural (see __init__).
        q = random.random()
        if q < self.fr_easy_frac:
            self._flip_reset_setup(state)
        elif q < self.fr_easy_frac + self.fr_mid_frac:
            self._air_dribble_flip_reset_setup(state)
        else:
            self._natural_flip_reset_setup(state)
