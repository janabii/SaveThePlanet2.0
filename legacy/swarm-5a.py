"""
3-Drone Coverage Swarm with Battery System
- 3 drones performing sweeping coverage patterns in sectors
- Added visualization: Debug lines show planned coverage paths
- Battery system with on-screen display for each drone
- Improved visibility: Camera starts top-down, debug visualizer enabled
"""

import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool

# ========= Defaults =========
DEFAULT_DRONES = DroneModel.CF2X
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_SIM_FREQ = 500
DEFAULT_CTRL_FREQ = 100
DEFAULT_DURATION = 3600
DEFAULT_NUM_DRONES = 3

# ========= World =========
WORLD_XY_LIMIT = 5.0
COVERAGE_STEP = 0.5
COVERAGE_HEIGHT = 1.0

# ========= Camera =========
CAM_DIST = 4.0
CAM_YAW = 0.0
CAM_PITCH = -80.0
top_down = True

# ========= Battery System =========
BATTERY_CAPACITY = 100.0  # 100% battery
BATTERY_DRAIN_BASE = 0.002  # Base drain per control step
BATTERY_DRAIN_MOVEMENT = 0.01  # Additional drain when moving


def update_camera(target):
    global CAM_DIST, CAM_YAW, CAM_PITCH, top_down
    keys = p.getKeyboardEvents()

    if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
        top_down = not top_down

    if top_down:
        p.resetDebugVisualizerCamera(15, 0, -90, [0, 0, 0])
    else:
        p.resetDebugVisualizerCamera(CAM_DIST, CAM_YAW, CAM_PITCH, target.tolist())


def generate_coverage(num, limit, step, z):
    paths = []
    width = 2 * limit / num
    for i in range(num):
        x_min = -limit + i * width
        x_max = x_min + width
        y = -limit
        direction = 1
        path = []
        while y <= limit:
            path.append([x_min if direction > 0 else x_max, y, z])
            path.append([x_max if direction > 0 else x_min, y, z])
            y += step
            direction *= -1
        paths.append(path)
    return paths


def draw_paths(paths):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue for 3 drones
    for i, path in enumerate(paths):
        for j in range(len(path) - 1):
            p.addUserDebugLine(path[j], path[j + 1], colors[i % len(colors)], 3)


def update_battery_levels(battery_levels, velocities):
    """Update battery levels based on drone activity"""
    for i in range(len(battery_levels)):
        # Base drain
        battery_levels[i] -= BATTERY_DRAIN_BASE

        # Movement drain based on velocity
        speed = np.linalg.norm(velocities[i])
        battery_levels[i] -= speed * BATTERY_DRAIN_MOVEMENT

        # Ensure battery doesn't go below 0
        battery_levels[i] = max(0.0, battery_levels[i])

    return battery_levels


def draw_battery_display(battery_levels, positions):
    """Draw battery level indicators on screen for each drone"""
    for i, (battery, pos) in enumerate(zip(battery_levels, positions)):
        # Calculate color based on battery level
        if battery > 60:
            color = [0, 1, 0]  # Green
        elif battery > 30:
            color = [1, 1, 0]  # Yellow
        else:
            color = [1, 0, 0]  # Red

        # Create battery bar text
        bar_length = 20
        filled = int((battery / BATTERY_CAPACITY) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Display battery info
        p.addUserDebugText(f"Drone {i+1}: {bar} {battery:.1f}%",
                          [pos[0] - 2.0, pos[1], pos[2] + 1.0],
                          textColorRGB=color,
                          textSize=1.0,
                          lifeTime=0.1)


def run(gui=DEFAULT_GUI):
    init_xyzs = np.array([
        [0.0, 0.0, 0.1],    # Drone 1 - left sector
        [0.0, -2.0, 0.1],   # Drone 2 - center sector
        [0.0, 2.0, 0.1]     # Drone 3 - right sector
    ])

    env = CtrlAviary(
        drone_model=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        initial_xyzs=init_xyzs,
        initial_rpys=np.zeros((DEFAULT_NUM_DRONES, 3)),
        physics=DEFAULT_PHYSICS,
        pyb_freq=DEFAULT_SIM_FREQ,
        ctrl_freq=DEFAULT_CTRL_FREQ,
        gui=gui
    )

    # ===== Apply desert texture to EXISTING plane =====
    plane_id = 0
    tex_id = p.loadTexture("textures/sand.png")
    p.changeVisualShape(plane_id, -1, textureUniqueId=tex_id, textureScaling=10)

    paths = generate_coverage(DEFAULT_NUM_DRONES, WORLD_XY_LIMIT, COVERAGE_STEP, COVERAGE_HEIGHT)
    draw_paths(paths)

    ctrls = [DSLPIDControl(drone_model=DEFAULT_DRONES) for _ in range(DEFAULT_NUM_DRONES)]
    action = np.zeros((DEFAULT_NUM_DRONES, 4))
    targets = np.copy(init_xyzs)
    indices = [0] * DEFAULT_NUM_DRONES

    # Initialize battery system
    battery_levels = np.full(DEFAULT_NUM_DRONES, BATTERY_CAPACITY, dtype=float)
    drone_velocities = np.zeros((DEFAULT_NUM_DRONES, 3), dtype=float)

    start = time.time()
    step = 0

    try:
        while True:
            obs, _, _, _, _ = env.step(action)

            for i in range(DEFAULT_NUM_DRONES):
                if indices[i] < len(paths[i]):
                    targets[i] = paths[i][indices[i]]
                    if np.linalg.norm(obs[i][:2] - targets[i][:2]) < 0.1:
                        indices[i] += 1

                action[i], _, _ = ctrls[i].computeControlFromState(
                    env.CTRL_TIMESTEP,
                    obs[i],
                    targets[i],
                    np.zeros(3)
                )

            # Update drone velocities for battery calculation
            for i in range(DEFAULT_NUM_DRONES):
                if step > 0:  # Calculate velocity based on position change
                    prev_pos = obs[i][:3] if step == 1 else drone_velocities[i]
                    drone_velocities[i] = (obs[i][:3] - prev_pos) / env.CTRL_TIMESTEP
                else:
                    drone_velocities[i] = np.zeros(3)

            # Update battery levels
            battery_levels = update_battery_levels(battery_levels, drone_velocities)

            # Draw battery display
            drone_positions = [obs[i][:3] for i in range(DEFAULT_NUM_DRONES)]
            draw_battery_display(battery_levels, drone_positions)

            update_camera(obs[0][:3])
            env.render()
            sync(step, start, env.CTRL_TIMESTEP)
            step += 1

            # Check if any drone is out of battery
            if np.any(battery_levels <= 0):
                print("Warning: One or more drones have run out of battery!")
                break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-Drone Coverage Swarm with Battery System")
    parser.add_argument("--gui", default=True, type=str2bool)
    args = parser.parse_args()
    run(**vars(args))
