from PyFlyt.core import Aviary
import numpy as np

# Starting positions [x, y, z]
start_pos = np.array([
    [ 0.0,  0.0, 3.0],   # Fixedwing (scout)
    [ 3.0,  0.0, 2.5],   # Quad 1 (leader)
    [ 1.0, -1.5, 2.5],   # Quad 2 (follower)
    [-1.0, -1.5, 2.5],   # Quad 3 (follower)
])
start_orn = np.zeros((4, 3))

# Mixed drone types
drone_type = ["fixedwing", "quadx", "quadx", "quadx"]

env = Aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    render=True,
    drone_type=drone_type
)

# Mode 7 = position hold for all drones
env.set_mode(7)

# Setpoints: [x, y, z, yaw] for each drone
env.set_setpoint(0, np.array([10.0, 0.0, 4.0, 0.0]))   # Fixedwing flies forward
env.set_setpoint(1, np.array([ 3.0, 0.0, 2.5, 0.0]))   # Quads hold position
env.set_setpoint(2, np.array([ 1.0,-1.5, 2.5, 0.0]))
env.set_setpoint(3, np.array([-1.0,-1.5, 2.5, 0.0]))

for _ in range(1000):
    env.step()

env.close()

import pybullet as p
from desert_utils import create_desert_terrain, spawn_desert_obstacles

# Get the PyBullet client ID from PyFlyt
client_id = env.CLIENT

# Create your desert terrain (same as before)
create_desert_terrain(
    heightmap_path="assets/terrain_desert_dunes.png",
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.15, 0.15, 1.5)
)

spawn_desert_obstacles(
    num_rocks=12,
    num_vegetation=6,
    area_size=35.0
)
