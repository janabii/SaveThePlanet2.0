"""
Keyboard leader with fixed diamond followers + FollowCam (keyboard only, no mouse)
- Formation is world-aligned, leader in front, two followers behind
- Camera follows the leader in third person and can be orbited with arrow keys

Controls:
  Drone:  W/A/S/D move, Q up, E down, B boost, Ctrl+C to exit
  Camera: Left/Right rotate, Up/Down tilt, Z zoom in, X zoom out, R reset, F re-center on leader
"""

import time
import math
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# ========= Drone Defaults =========
DEFAULT_DRONES = DroneModel.CF2X
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 500
DEFAULT_CONTROL_FREQ_HZ = 100
DEFAULT_DURATION_SEC = 3600
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_NUM_DRONES = 3

# ========= Flight Parameters =========
TAKEOFF_Z = 1.0
TAKEOFF_TIME = 3.0

SPEED_XY = 3.6
SPEED_Z  = 2.5
BOOST    = 1.8

WORLD_XY_LIMIT = 5.0
Z_MIN, Z_MAX = 0.2, 2.0
MAX_LEADER_SPEED = 1.2

# fixed diamond offsets relative to leader, world aligned
DIAMOND_OFFSETS = np.array([
    [ 0.0,  0.0],   # leader
    [-1.2, -0.6],   # follower 1
    [-1.2,  0.6],   # follower 2
], dtype=float)

# collision avoidance
MIN_SEP = 0.6
AVOID_GAIN = 0.5
MAX_PUSH = 0.7

# ========= FollowCam state (keyboard only) =========
CAM_DIST_DEFAULT  = 4.0
CAM_YAW_DEFAULT   = 0.0
CAM_PITCH_DEFAULT = -80.0

cam_dist  = CAM_DIST_DEFAULT
cam_yaw   = CAM_YAW_DEFAULT
cam_pitch = CAM_PITCH_DEFAULT

# camera tuning
CAM_YAW_STEP   = 2.0     # deg per tick when arrow held
CAM_PITCH_STEP = 1.5
CAM_DIST_STEP  = 0.15
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0
CAM_DIST_MIN, CAM_DIST_MAX   = 1.5, 12.0

def read_keyboard_vel():
    keys = p.getKeyboardEvents()
    vx = vy = vz = 0.0
    boost = False
    if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
        vx += SPEED_XY
    if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
        vx -= SPEED_XY
    if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
        vy += SPEED_XY
    if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
        vy -= SPEED_XY
    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
        vz += SPEED_Z
    if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
        vz -= SPEED_Z
    if ord('b') in keys and keys[ord('b')] & p.KEY_IS_DOWN:
        boost = True
    mul = BOOST if boost else 1.0
    return np.array([vx*mul, vy*mul, vz*mul], dtype=float)

def clamp_xy(xy):
    return np.clip(xy, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)

def update_follow_cam(target_pos_3d):
    """Keyboard only, stable on Windows. Orbits around leader and looks at it."""
    global cam_dist, cam_yaw, cam_pitch

    keys = p.getKeyboardEvents()

    # rotate yaw with Left/Right arrows
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        cam_yaw -= CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        cam_yaw += CAM_YAW_STEP

    # tilt pitch with Up/Down arrows
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch + CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch - CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)

    # zoom with Z and X
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
        cam_dist = np.clip(cam_dist - CAM_DIST_STEP, CAM_DIST_MIN, CAM_DIST_MAX)
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
        cam_dist = np.clip(cam_dist + CAM_DIST_STEP, CAM_DIST_MIN, CAM_DIST_MAX)

    # reset with R
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist  = CAM_DIST_DEFAULT
        cam_yaw   = CAM_YAW_DEFAULT
        cam_pitch = CAM_PITCH_DEFAULT

    # re center on leader with F
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        # keep current yaw, pitch, distance, only reset the view to the leader target
        pass

    # apply
    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=target_pos_3d.tolist()
    )

def run(
    drone=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER
):
    assert num_drones == 3, "This file is configured for 3 drones"

    # safer staggered start to avoid early bumps
    init_xyzs = np.array([
        [ 0.0,  0.0, 0.10],
        [-0.7, -0.5, 0.20],
        [-0.7,  0.5, 0.30],
    ], dtype=float)
    init_rpys = np.zeros((num_drones, 3), dtype=float)

    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=physics,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        user_debug_gui=user_debug_gui
    )

    ctrls = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
    action = np.zeros((num_drones, 4))

    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones, output_folder=output_folder)

    dt = 1.0 / control_freq_hz
    start = time.time()

    # leader state
    leader_pos = np.copy(init_xyzs[0])
    leader_vel = np.zeros(3, dtype=float)

    # targets
    target_pos = np.copy(init_xyzs)

    takeoff_steps = int(TAKEOFF_TIME * env.CTRL_FREQ)
    step = 0

    try:
        while True:
            # leader
            if step < takeoff_steps:
                alpha = (step + 1) / max(1, takeoff_steps)
                leader_pos[2] = 0.10*(1 - alpha) + TAKEOFF_Z*alpha
                leader_vel[:] = 0.0
            else:
                kv = read_keyboard_vel()
                sp = np.linalg.norm(kv[:2])
                if sp > MAX_LEADER_SPEED:
                    kv[:2] = kv[:2] * (MAX_LEADER_SPEED / max(sp, 1e-6))
                leader_vel = kv
                leader_pos[:2] = clamp_xy(leader_pos[:2] + leader_vel[:2] * dt)
                leader_pos[2]  = np.clip(leader_pos[2] + leader_vel[2] * dt, Z_MIN, Z_MAX)

            # fixed diamond, world aligned
            raw_targets = np.zeros((num_drones, 3), dtype=float)
            for j in range(num_drones):
                off_xy = DIAMOND_OFFSETS[j]
                tgt_xy = leader_pos[:2] + off_xy
                tgt_z  = leader_pos[2]
                raw_targets[j] = np.array([tgt_xy[0], tgt_xy[1], tgt_z], dtype=float)

            # step physics
            obs, _, _, _, _ = env.step(action)

            # collision avoidance and bounds
            target_pos[:] = raw_targets
            for j in range(num_drones):
                pos_j = obs[j][0:3]
                sep_push = np.zeros(2, dtype=float)
                for k in range(num_drones):
                    if k == j:
                        continue
                    pos_k = obs[k][0:3]
                    dx = pos_j[0] - pos_k[0]
                    dy = pos_j[1] - pos_k[1]
                    dist = math.hypot(dx, dy)
                    if dist < MIN_SEP and dist > 1e-6:
                        push = AVOID_GAIN * (MIN_SEP - dist) / dist * np.array([dx, dy])
                        sep_push += push
                n = np.linalg.norm(sep_push)
                if n > MAX_PUSH and n > 0:
                    sep_push = sep_push * (MAX_PUSH / n)
                target_pos[j, 0:2] += sep_push
                target_pos[j, 0:2] = clamp_xy(target_pos[j, 0:2])
                target_pos[j, 2]   = np.clip(target_pos[j, 2], Z_MIN, Z_MAX)

            # controls
            for j in range(num_drones):
                raw = ctrls[j].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[j],
                    target_pos=target_pos[j],
                    target_rpy=np.zeros(3),
                    target_vel=np.zeros(3),
                    target_rpy_rates=np.zeros(3)
                )
                action[j, :] = raw[0]
                logger.log(drone=j, timestamp=step * dt, state=obs[j])

            # follow camera
            # camera target is slightly above leader to keep drones in frame
            cam_target = leader_pos.copy()
            update_follow_cam(cam_target)

            env.render()
            if gui:
                sync(step, start, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        logger.save()
        logger.save_as_csv("leader_fixed_diamond_followcam")
        if plot:
            logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyboard leader with fixed diamond followers + FollowCam")
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
