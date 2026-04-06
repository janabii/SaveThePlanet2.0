"""
Autonomous search: 5 drones in W formation fly over all waste objects.

- Leader follows waypoints above all waste positions automatically (no WASD).
- 5 drones keep a fixed W formation around the leader.
- The world has multiple "waste" cubes, widely spaced.
- Some cubes are ORANGE, others have different colors.
- Each drone has a virtual onboard camera.
- When a drone's camera sees ORANGE region in the image:
    -> It automatically saves a screenshot
    -> A GREEN rectangle is drawn around the orange area (detection box).

Controls:
  Movement:        AUTOPILOT (no keyboard needed).
  Manual save:     Press C to save camera image of every drone (PNG files).
  Camera control:  Arrow keys (yaw/pitch), Z/X (zoom), R (reset).
  Exit:            Ctrl + C in terminal / closing window.
"""

import time
import math
import argparse
import os
import numpy as np
import pybullet as p

# ===== Optional: OpenCV for saving camera images and color detection =====
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

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

DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_DURATION_SEC = 3600
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_NUM_DRONES = 5

# ========= Flight Parameters =========
TAKEOFF_Z = 2.0
TAKEOFF_TIME = 3.0

AUTO_SPEED_XY = 1.8
AUTO_SPEED_Z  = 1.0

WORLD_XY_LIMIT = 15.0
Z_MIN, Z_MAX = 0.5, 4.0

# ========= Desert Ground (NEW) =========
# Put your texture here: project/assets/desert_sand.png
DESERT_TEXTURE_PATH = os.path.join("assets", "desert_sand.png")

# W formation offsets relative to leader (top view)
W_OFFSETS = np.array([
    [ 0.0,  0.0],   # Leader (drone 0)
    [-1.5, -1.0],   # Drone 1
    [ 1.5, -1.0],   # Drone 2
    [-3.0,  1.0],   # Drone 3
    [ 3.0,  1.0],   # Drone 4
], dtype=float)

# collision avoidance
MIN_SEP = 0.8
AVOID_GAIN = 0.5
MAX_PUSH = 0.7

# FollowCam (third-person)
CAM_DIST_DEFAULT  = 10.0
CAM_YAW_DEFAULT   = 0.0
CAM_PITCH_DEFAULT = -70.0

cam_dist  = CAM_DIST_DEFAULT
cam_yaw   = CAM_YAW_DEFAULT
cam_pitch = CAM_PITCH_DEFAULT

CAM_YAW_STEP   = 2.0
CAM_PITCH_STEP = 1.5
CAM_DIST_STEP  = 0.2
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0
CAM_DIST_MIN, CAM_DIST_MAX   = 2.0, 25.0

# Onboard camera
CAM_WIDTH  = 200
CAM_HEIGHT = 150
CAM_FOV    = 45.0
CAM_NEAR   = 0.1
CAM_FAR    = 100.0

CAMERA_OUTPUT_DIR = "camera_frames"

# ========= Waste positions (very spread out) =========
WASTE_POSITIONS = [
    (-12.0, -10.0, 0.0),
    (-12.0,   10.0, 0.0),
    (   0.0,  -12.0, 0.0),
    (   0.0,   12.0, 0.0),
    (  12.0,  -10.0, 0.0),
    (  12.0,   10.0, 0.0),
]

# ========= Colors for each cube =========
WASTE_COLORS = [
    "orange",   # 0
    "blue",     # 1
    "orange",   # 2
    "gray",     # 3
    "orange",   # 4
    "green",    # 5
]

# Color mapping RGBA
COLOR_MAP = {
    "orange": [1.0, 0.5, 0.0, 1.0],
    "blue":   [0.1, 0.4, 1.0, 1.0],
    "gray":   [0.7, 0.7, 0.7, 1.0],
    "green":  [0.2, 0.8, 0.2, 1.0],
}

# PyBullet body IDs
WASTE_BODY_IDS = []

# Cooldown between auto-screenshots per drone (in simulation steps)
MIN_STEPS_BETWEEN_CAPTURES = 40


def clamp_xy(xy):
    return np.clip(xy, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)


def apply_desert_ground(texture_path=DESERT_TEXTURE_PATH):
    """
    Apply a sand/desert texture to the ground plane.
    Works even if the plane body id is not 0, by scanning bodies for a name containing "plane".
    """
    if not os.path.exists(texture_path):
        print(f"[World] Desert texture not found: {texture_path}")
        print("[World] Create folder 'assets' and place 'desert_sand.png' inside it.")
        return

    try:
        tex_id = p.loadTexture(texture_path)
    except Exception as e:
        print(f"[World] Failed to load texture: {texture_path}")
        print(f"[World] Error: {e}")
        return

    applied = False
    num_bodies = p.getNumBodies()
    for bid in range(num_bodies):
        try:
            info = p.getBodyInfo(bid)
            base_name = info[1].decode("utf-8", errors="ignore").lower() if info and len(info) > 1 else ""
        except Exception:
            base_name = ""

        # The default ground in many setups is "plane"
        if "plane" in base_name:
            try:
                p.changeVisualShape(objectUniqueId=bid, linkIndex=-1, textureUniqueId=tex_id)
                applied = True
            except Exception:
                pass

    # Fallback: if we didn't find a named plane, try applying to body 0
    if not applied and num_bodies > 0:
        try:
            p.changeVisualShape(objectUniqueId=0, linkIndex=-1, textureUniqueId=tex_id)
            applied = True
        except Exception:
            applied = False

    if applied:
        print(f"[World] Desert ground texture applied: {texture_path}")
    else:
        print("[World] Could not apply texture to ground. No plane body found.")


def update_follow_cam(target_pos_3d, keys):
    """Update third-person camera using arrow/Z/X/R keys."""
    global cam_dist, cam_yaw, cam_pitch

    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        cam_yaw -= CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        cam_yaw += CAM_YAW_STEP

    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch + CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch - CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)

    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
        cam_dist = np.clip(cam_dist - CAM_DIST_STEP, CAM_DIST_MIN, CAM_DIST_MAX)
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
        cam_dist = np.clip(cam_dist + CAM_DIST_STEP, CAM_DIST_MIN, CAM_DIST_MAX)

    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist  = CAM_DIST_DEFAULT
        cam_yaw   = CAM_YAW_DEFAULT
        cam_pitch = CAM_PITCH_DEFAULT

    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=target_pos_3d.tolist()
    )


def get_drone_camera_rgb(obs_drone):
    """Onboard camera RGB from drone state."""
    if not _HAS_CV2:
        return None

    pos = np.array(obs_drone[0:3], dtype=float)
    quat = np.array(obs_drone[3:7], dtype=float)

    rm = p.getMatrixFromQuaternion(quat)
    forward = np.array([rm[0], rm[3], rm[6]], dtype=float)
    up      = np.array([rm[2], rm[5], rm[8]], dtype=float)

    cam_pos = pos + 0.05 * up
    cam_target = cam_pos + forward

    view = p.computeViewMatrix(
        cameraEyePosition=cam_pos.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=up.tolist()
    )
    proj = p.computeProjectionMatrixFOV(
        fov=CAM_FOV,
        aspect=float(CAM_WIDTH) / float(CAM_HEIGHT),
        nearVal=CAM_NEAR,
        farVal=CAM_FAR
    )

    img = p.getCameraImage(
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER
    )

    rgba = np.reshape(img[2], (CAM_HEIGHT, CAM_WIDTH, 4))
    rgb = rgba[:, :, :3].astype(np.uint8)
    return rgb


def save_all_drone_cameras(obs, frame_idx):
    """Manual C key: save all drones' camera images (no detection logic)."""
    if not _HAS_CV2:
        print("[Camera] OpenCV not installed, can't save images.")
        return

    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)

    for j, state in enumerate(obs):
        rgb = get_drone_camera_rgb(state)
        if rgb is None:
            continue
        filename = os.path.join(
            CAMERA_OUTPUT_DIR,
            f"manual_drone{j}_{frame_idx:05d}.png"
        )
        cv2.imwrite(filename, rgb)
        print(f"[Camera] [Manual] Saved {filename}")


def detect_orange_and_save(obs, frame_idx, last_saved_steps):
    """
    AUTO:
    - لكل درون: ناخذ الصورة من الكاميرا
    - نبحث عن اللون البرتقالي باستخدام HSV
    - لو وجدناه بشكل واضح → نرسم مستطيل أخضر حوله ونحفظ الصورة
    """
    if not _HAS_CV2:
        return

    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)

    lower_orange = np.array([5, 150, 100])
    upper_orange = np.array([25, 255, 255])

    for j, state in enumerate(obs):
        if frame_idx - last_saved_steps[j] < MIN_STEPS_BETWEEN_CAPTURES:
            continue

        rgb = get_drone_camera_rgb(state)
        if rgb is None:
            continue

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < 80:
            continue

        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        filename = os.path.join(
            CAMERA_OUTPUT_DIR,
            f"orange_detect_drone{j}_{frame_idx:05d}.png"
        )
        cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        last_saved_steps[j] = frame_idx
        print(f"[Camera] [AUTO] Drone {j} detected ORANGE → Saved {filename}")


def spawn_waste_objects():
    """
    Spawn waste cubes at fixed positions, color some of them orange.
    """
    global WASTE_BODY_IDS

    try:
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf_name = "cube.urdf"
    except ImportError:
        urdf_name = "cube.urdf"

    ids = []
    for pos, color_name in zip(WASTE_POSITIONS, WASTE_COLORS):
        body_id = p.loadURDF(
            urdf_name,
            basePosition=pos,
            globalScaling=0.5
        )
        ids.append(body_id)

        color_rgba = COLOR_MAP.get(color_name, [1, 1, 1, 1])
        p.changeVisualShape(body_id, -1, rgbaColor=color_rgba)

    WASTE_BODY_IDS = ids
    print("[World] Spawned waste cubes at positions:")
    for pos, col in zip(WASTE_POSITIONS, WASTE_COLORS):
        print(f"   {pos}  -> {col}")


def build_search_waypoints():
    """
    Leader waypoints: fly above ALL waste positions (x,y) at altitude TAKEOFF_Z.
    """
    waypoints = []
    for (x, y, z_ground) in WASTE_POSITIONS:
        waypoints.append(np.array([x, y, TAKEOFF_Z], dtype=float))
    return waypoints


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
    assert num_drones == 5, "This file is configured for 5 drones"

    # Initial W-ish positions near origin
    init_xyzs = np.array([
        [ 0.0,  0.0, 0.10],
        [-1.5, -1.0, 0.10],
        [ 1.5, -1.0, 0.10],
        [-3.0,  1.0, 0.10],
        [ 3.0,  1.0, 0.10],
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

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # ===== Desert ground (NEW) =====
    apply_desert_ground(DESERT_TEXTURE_PATH)

    # World setup
    spawn_waste_objects()
    waypoints = build_search_waypoints()
    current_wp_idx = 0

    ctrls = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
    action = np.zeros((num_drones, 4))

    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones, output_folder=output_folder)

    dt = 1.0 / control_freq_hz
    start = time.time()

    leader_pos = np.copy(init_xyzs[0])
    leader_vel = np.zeros(3, dtype=float)
    target_pos = np.copy(init_xyzs)

    takeoff_steps = int(TAKEOFF_TIME * env.CTRL_FREQ)
    step = 0

    last_saved_steps = [-999999] * num_drones

    try:
        while True:
            keys = p.getKeyboardEvents()

            # === Leader motion (takeoff + autopilot) ===
            if step < takeoff_steps:
                alpha = (step + 1) / max(1, takeoff_steps)
                leader_pos[2] = 0.10*(1 - alpha) + TAKEOFF_Z*alpha
                leader_vel[:] = 0.0
            else:
                if current_wp_idx < len(waypoints):
                    wp = waypoints[current_wp_idx]
                    diff = wp - leader_pos
                    dist_xy = math.hypot(diff[0], diff[1])

                    if dist_xy < 0.5 and abs(diff[2]) < 0.3:
                        print(f"[AUTO] Reached waypoint {current_wp_idx}: {wp}")
                        current_wp_idx += 1
                        leader_vel[:] = 0.0
                    else:
                        if dist_xy > 1e-6:
                            dir_xy = diff[:2] / dist_xy
                        else:
                            dir_xy = np.zeros(2, dtype=float)

                        leader_vel[0] = dir_xy[0] * AUTO_SPEED_XY
                        leader_vel[1] = dir_xy[1] * AUTO_SPEED_XY
                        leader_vel[2] = np.clip(diff[2], -AUTO_SPEED_Z, AUTO_SPEED_Z)

                        leader_pos[:2] = clamp_xy(leader_pos[:2] + leader_vel[:2] * dt)
                        leader_pos[2]  = np.clip(leader_pos[2] + leader_vel[2] * dt, Z_MIN, Z_MAX)
                else:
                    leader_vel[:] = 0.0

            # === Formation target positions for all drones ===
            raw_targets = np.zeros((num_drones, 3), dtype=float)
            for j in range(num_drones):
                off_xy = W_OFFSETS[j]
                tgt_xy = leader_pos[:2] + off_xy
                tgt_z  = leader_pos[2]
                raw_targets[j] = np.array([tgt_xy[0], tgt_xy[1], tgt_z], dtype=float)

            # Step physics
            obs, _, _, _, _ = env.step(action)

            # Manual capture
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                save_all_drone_cameras(obs, step)

            # AUTO: detect orange and save with green box
            detect_orange_and_save(obs, step, last_saved_steps)

            # Collision avoidance and clamp
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

            # Controllers
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

            # Follow camera on leader
            update_follow_cam(leader_pos.copy(), keys)

            env.render()
            if gui:
                sync(step, start, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        logger.save()
        logger.save_as_csv("auto_orange_detect_wide")
        if plot:
            logger.plot()
        print("Closed. Logs saved.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous search with orange color detection and green bounding box")
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
