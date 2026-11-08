"""
Minimal rlgym compatibility stubs for Ripple.
Just enough to hold state data - no game logic.
"""
import numpy as np

BLUE_TEAM = 0
ORANGE_TEAM = 1

class V1PhysicsObject:
    def __init__(self):
        self.position = np.zeros(3)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.rotation_mtx = lambda: np.eye(3)
        self.euler_angles = [0, 0, 0]

class V1PlayerData:
    def __init__(self):
        self.car_id = 0
        self.team_num = BLUE_TEAM
        self.car_data = V1PhysicsObject()
        self.on_ground = False
        self.has_jump = True
        self.has_flip = True
        self.boost_amount = 0.33
        self.is_demoed = False

class V1GameState:
    def __init__(self):
        self.ball = V1PhysicsObject()
        self.players = []
        self.boost_pads = np.ones(34)  # All active by default
