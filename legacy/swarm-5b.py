import time
import math
import argparse
import os
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool

# ================== CONFIG ==================
NUM_DRONES = 3
SIM_FREQ = 240
CTRL_FREQ = 30

TAKEOFF_Z = 2.0
WORLD_XY_LIMIT = 15.0

BATTERY_MAX = 100.0
BATTERY_DRAIN_RATE = 0.02  # drain per meter moved

# ================== FORMATION ==================
W_OFFSETS = np.array([
    [ 0.0,  0.0],   # Leader
    [-1.5, -1.0],   # Follower 1
    [ 1.5, -1.0],   # Follower 2
], dtype=float)

# ================== BATTERY STATE ==================
battery_levels = np.ones(NUM_DRONES) * BATTERY_MAX
last_positions = None
battery_text_ids = [None] * NUM_DRONES

# ================== HELPERS ==================
def clamp_xy(xy):
    return np.clip(xy, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)

# ================== BATTERY HUD ==================
def update_battery_hud(drone_states):
    global battery_levels, last_positions, battery_text_ids

    if last_positions is None:
        last_positions = [np.array(s[0:3]) for s in drone_states]
        return

    for i in range(NUM_DRONES):
        pos = np.array(drone_states[i][0:3])
        dist = np.linalg.norm(pos - last_positions[i])
        last_positions[i] = pos

        battery_levels[i] -= dist * BATTERY_DRAIN_RATE
        battery_levels[i] = max(0.0, battery_levels[i])

        # Remove old HUD text
        if battery_text_ids[i] is not None:
            p.removeUserDebugItem(battery_text_ids[i])

        # Battery bar text (screen-left illusion)
        bar_len = int((battery_levels[i] / BATTERY_MAX) * 10)
        bar = "█" * bar_len + " " * (10 - bar_len)

        text = f"Drone {i} [{bar}] {battery_levels[i]:.0f}%"

        battery_text_ids[i] = p.addUserDebugText(
            text,
            textPosition=[-1, 1.2 - i * 0.15, 0],  # fixed relative position
            textColorRGB=[1, 1, 1],               # white border/text
            textSize=1.4,
            lifeTime=0
        )

# ================== MAIN ==================
def run(gui=True):
    init_xyzs = np.array([
        [ 0.0,  0.0, 0.1],
        [-1.5, -1.0, 0.1],
        [ 1.5, -1.0, 0.1],
    ])

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=init_xyzs,
        initial_rpys=np.zeros((NUM_DRONES, 3)),
        physics=Physics.PYB,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui
    )

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(NUM_DRONES)]
    action = np.zeros((NUM_DRONES, 4))

    leader_pos = init_xyzs[0].copy()
    dt = 1.0 / CTRL_FREQ
    step = 0
    start = time.time()

    try:
        while True:
            leader_pos[0] += 0.5 * dt
            leader_pos[:2] = clamp_xy(leader_pos[:2])

            targets = np.array([
                leader_pos + np.array([*W_OFFSETS[i], 0])
                for i in range(NUM_DRONES)
            ])

            obs, _, _, _, _ = env.step(action)

            for i in range(NUM_DRONES):
                action[i], _, _ = ctrls[i].computeControlFromState(
                    env.CTRL_TIMESTEP,
                    obs[i],
                    targets[i],
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(3)
                )

            update_battery_hud(obs)

            env.render()
            sync(step, start, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Closed cleanly")

# ================== ENTRY ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", default=True, type=str2bool)
    args = parser.parse_args()
    run(**vars(args))
