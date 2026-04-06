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
import json
import secrets
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
from swarm_crypto_comm import SecureSwarmComm, CyberMetrics

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

AUTO_SPEED_XY = 1.8  # Incremental speed increase with smooth transitions
AUTO_SPEED_Z  = 1.0  # Match original vertical speed with smooth control

WORLD_XY_LIMIT = 15.0
Z_MIN, Z_MAX = 1.0, 4.0  # Increased minimum to 1.0m for safety

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
AVOID_GAIN = 0.3  # Reduced from 0.5 for gentler collision avoidance
MAX_PUSH = 0.7

# ========= Secure Communication & Consensus Parameters =========
COMM_RADIUS = 8.0  # meters
NETWORK_LATENCY_MS = 10.0  # simulated propagation delay
CONSENSUS_GAIN = 0.1  # Reduced from 0.3 for stability (weight of consensus velocity correction)
CRYPTO_POWER_DRAW_MW = 7.5  # mW per crypto operation
ENABLE_ENCRYPTION = True  # Set to False for baseline comparison

# ========= Battery & Propulsion Model (Research-Based) =========
# Based on Zeng et al. (2018) UAV energy model: P = P_hover + k_v*v^2 + k_a*|a|
# CF2X (Crazyflie 2.X) specifications
BATTERY_CAPACITY_WH = 4.44  # Watt-hours (simulated larger battery for long missions)
BATTERY_CAPACITY_J = BATTERY_CAPACITY_WH * 3600  # Convert to Joules (3196.8 J)

# Power consumption parameters (Watts)
P_HOVER = 4.0        # Hover power for CF2X quadrotor
K_VELOCITY = 0.05    # Velocity drag coefficient W/(m/s)^2
K_ACCEL = 0.02       # Acceleration maneuver coefficient W/(m/s^2)
P_CRYPTO = 0.0075    # Crypto operation power (7.5 mW)

# Low battery threshold
LOW_BATTERY_THRESHOLD = 0.10  # 10% triggers RTB
CRITICAL_BATTERY_THRESHOLD = 0.05  # 5% emergency landing

# Fleet configuration
NUM_DRONES = 5  # All quadrotors

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

    for j_str in obs.keys():
        j = int(j_str)
        state = obs[j_str]["state"]
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

    for j_str in obs.keys():
        j = int(j_str)
        if frame_idx - last_saved_steps[j] < MIN_STEPS_BETWEEN_CAPTURES:
            continue

        state = obs[j_str]["state"]
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


def compute_consensus_adjustment(drone_id, drone_vel, neighbor_states):
    """
    Compute consensus-based velocity adjustment (Olfati-Saber framework).
    
    Parameters
    ----------
    drone_id : int
        ID of the current drone
    drone_vel : np.ndarray
        Current velocity of the drone [vx, vy, vz]
    neighbor_states : list
        List of neighbor state dictionaries
        
    Returns
    -------
    np.ndarray
        Consensus velocity adjustment [vx, vy, vz]
    """
    if not neighbor_states:
        return np.zeros(3)
    
    # Consensus: u_i = sum_{j in N_i} (v_j - v_i)
    consensus_sum = np.zeros(3)
    
    for neighbor_state in neighbor_states:
        neighbor_vel = np.array(neighbor_state['vel'])
        consensus_sum += (neighbor_vel - drone_vel)
    
    consensus_adjustment = CONSENSUS_GAIN * consensus_sum
    
    return consensus_adjustment


def update_battery_hud(drone_positions, battery_levels, text_ids, rtb_active=None):
    """
    Display battery percentage above each drone using PyBullet debug text.
    
    Parameters
    ----------
    drone_positions : np.ndarray
        Current positions of all drones [num_drones x 3]
    battery_levels : np.ndarray
        Battery levels for all drones (0.0 to 1.0)
    text_ids : list
        List of existing text IDs to update/replace
    rtb_active : list, optional
        RTB active status for all drones
        
    Returns
    -------
    list
        Updated text IDs
    """
    for j in range(len(battery_levels)):
        battery_pct = battery_levels[j] * 100.0
        
        # Show RTB status in text
        if rtb_active is not None and j < len(rtb_active) and rtb_active[j]:
            if battery_pct < 5:
                text = f"Drone {j}: {battery_pct:.1f}% EMERGENCY"
            else:
                text = f"Drone {j}: {battery_pct:.1f}% RTB"
        else:
            text = f"Drone {j}: {battery_pct:.1f}%"
        
        # Choose color based on battery level
        if battery_pct > 60:
            color = [0, 1, 0]  # Green
        elif battery_pct > 30:
            color = [1, 0.8, 0]  # Yellow/Orange
        else:
            color = [1, 0, 0]  # Red
        
        # Position text above the drone
        text_position = drone_positions[j] + np.array([0, 0, 0.5])
        
        # Remove old text if exists
        if text_ids[j] != -1:
            p.removeUserDebugItem(text_ids[j])
        
        # Add new text
        text_ids[j] = p.addUserDebugText(
            text=text,
            textPosition=text_position.tolist(),
            textColorRGB=color,
            textSize=1.2,
            lifeTime=0  # Permanent until manually removed
        )
    
    return text_ids


def calculate_drone_power(velocity, acceleration, crypto_ops_per_sec=0):
    """
    Calculate instantaneous power consumption based on UAV propulsion model.
    
    Based on Zeng et al. (2018): P = P_hover + k_v*v^2 + k_a*|a| + P_crypto
    
    Parameters
    ----------
    velocity : np.ndarray
        Velocity vector [vx, vy, vz] in m/s
    acceleration : np.ndarray
        Acceleration vector [ax, ay, az] in m/s^2
    crypto_ops_per_sec : float
        Number of crypto operations per second
        
    Returns
    -------
    float
        Power consumption in Watts
    """
    # Hover baseline (always needed for quadrotor)
    power = P_HOVER
    
    # Velocity drag term (proportional to speed squared)
    speed = np.linalg.norm(velocity)
    power += K_VELOCITY * (speed ** 2)
    
    # Acceleration maneuver term (proportional to acceleration magnitude)
    accel_mag = np.linalg.norm(acceleration)
    power += K_ACCEL * accel_mag
    
    # Crypto operations power
    power += P_CRYPTO * crypto_ops_per_sec
    
    return power


def update_battery_energy(
    drone_velocities,
    prev_velocities,
    battery_energy_j,
    dt,
    crypto_ops_count
):
    """
    Update battery energy for all drones based on propulsion physics.
    
    Parameters
    ----------
    drone_velocities : np.ndarray
        Current velocities [num_drones x 3]
    prev_velocities : np.ndarray
        Previous velocities [num_drones x 3]
    battery_energy_j : np.ndarray
        Current battery energy in Joules [num_drones]
    dt : float
        Timestep in seconds
    crypto_ops_count : np.ndarray
        Crypto operations per drone in this timestep [num_drones]
        
    Returns
    -------
    tuple
        (updated_energy_j, battery_levels, power_consumption_w)
    """
    num_drones = len(battery_energy_j)
    power_consumption = np.zeros(num_drones)
    
    for j in range(num_drones):
        # Calculate acceleration from velocity change
        acceleration = (drone_velocities[j] - prev_velocities[j]) / dt
        
        # Crypto operations per second (operations in this timestep / timestep)
        crypto_ops_per_sec = crypto_ops_count[j] / dt if dt > 0 else 0
        
        # Calculate power consumption
        power = calculate_drone_power(
            drone_velocities[j],
            acceleration,
            crypto_ops_per_sec
        )
        power_consumption[j] = power
        
        # Update energy: E_new = E_old - P * dt
        energy_consumed_j = power * dt
        battery_energy_j[j] = max(0.0, battery_energy_j[j] - energy_consumed_j)
    
    # Convert to battery percentage (0.0 to 1.0)
    battery_levels = battery_energy_j / BATTERY_CAPACITY_J
    
    return battery_energy_j, battery_levels, power_consumption


def check_and_handle_rtb(
    drone_id,
    battery_level,
    current_pos,
    base_pos,
    rtb_active,
    leader_vel
):
    """
    Check battery level and handle Return-To-Base logic.
    
    Parameters
    ----------
    drone_id : int
        Drone ID
    battery_level : float
        Battery level (0.0 to 1.0)
    current_pos : np.ndarray
        Current position [x, y, z]
    base_pos : np.ndarray
        Base/home position [x, y, z]
    rtb_active : list
        RTB active status for all drones
    leader_vel : np.ndarray
        Leader's velocity (used when not in RTB)
        
    Returns
    -------
    tuple
        (target_position, target_velocity, rtb_status)
    """
    # Check battery thresholds
    if battery_level < CRITICAL_BATTERY_THRESHOLD:
        # Emergency landing at current location
        target_pos = current_pos.copy()
        target_pos[2] = 0.1  # Descend to ground
        target_vel = np.array([0, 0, -0.5])  # Slow descent
        rtb_active[drone_id] = True
        return target_pos, target_vel, "EMERGENCY_LAND"
        
    elif battery_level < LOW_BATTERY_THRESHOLD and not rtb_active[drone_id]:
        # Trigger RTB
        rtb_active[drone_id] = True
        print(f"\n[RTB] Drone {drone_id} battery at {battery_level*100:.1f}% - Returning to base!")
        
    if rtb_active[drone_id]:
        # Navigate back to base
        diff = base_pos - current_pos
        dist = np.linalg.norm(diff)
        
        if dist < 0.3:  # Reached base
            # Land at base
            target_pos = base_pos.copy()
            target_pos[2] = 0.1
            target_vel = np.zeros(3)
            return target_pos, target_vel, "LANDED"
        else:
            # Move toward base
            target_pos = base_pos.copy()
            direction = diff / (dist + 1e-6)
            target_vel = direction * 1.5  # RTB speed
            return target_pos, target_vel, "RTB_ACTIVE"
    
    # Normal operation
    return None, leader_vel, "NORMAL"


def update_battery_hud(drone_positions, battery_levels, text_ids, rtb_active=None):
    """
    Display battery percentage above each drone using PyBullet debug text.
    
    Parameters
    ----------
    drone_positions : np.ndarray
        Current positions of all drones [num_drones x 3]
    battery_levels : np.ndarray
        Battery levels for all drones (0.0 to 1.0)
    text_ids : list
        List of existing text IDs to update/replace
    rtb_active : list, optional
        RTB active status for all drones
        
    Returns
    -------
    list
        Updated text IDs
    """
    for j in range(len(battery_levels)):
        battery_pct = battery_levels[j] * 100.0
        
        # Show RTB status in text
        if rtb_active is not None and j < len(rtb_active) and rtb_active[j]:
            if battery_pct < 5:
                text = f"Drone {j}: {battery_pct:.1f}% EMERGENCY"
            else:
                text = f"Drone {j}: {battery_pct:.1f}% RTB"
        else:
            text = f"Drone {j}: {battery_pct:.1f}%"
        
        # Choose color based on battery level
        if battery_pct > 60:
            color = [0, 1, 0]  # Green
        elif battery_pct > 30:
            color = [1, 0.8, 0]  # Yellow/Orange
        else:
            color = [1, 0, 0]  # Red
        
        # Position text above the drone
        text_position = drone_positions[j] + np.array([0, 0, 0.5])
        
        # Remove old text if exists
        if text_ids[j] != -1:
            p.removeUserDebugItem(text_ids[j])
        
        # Add new text
        text_ids[j] = p.addUserDebugText(
            text=text,
            textPosition=text_position.tolist(),
            textColorRGB=color,
            textSize=1.2,
            lifeTime=0  # Permanent until manually removed
        )
    
    return text_ids


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
    output_folder=DEFAULT_OUTPUT_FOLDER,
    enable_encryption=ENABLE_ENCRYPTION
):
    assert num_drones == 5, "This file is configured for 5 drones"

    # All quadrotors (5x CF2X)
    init_xyzs = np.array([
        [ 0.0,  0.0, 0.10],   # Leader
        [-1.5, -1.0, 0.10],   # Drone 1
        [ 1.5, -1.0, 0.10],   # Drone 2
        [-3.0,  1.0, 0.10],   # Drone 3
        [ 3.0,  1.0, 0.10],   # Drone 4
    ], dtype=float)
    init_rpys = np.zeros((num_drones, 3), dtype=float)

    # Create environment with all quadrotors
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=physics,
        freq=simulation_freq_hz,
        aggregate_phy_steps=simulation_freq_hz // control_freq_hz,
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

    # All quadrotor controllers
    ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
    action = np.zeros((num_drones, 4))
    
    # Initialize secure communication system
    secure_comm = SecureSwarmComm(
        num_drones=num_drones,
        comm_radius=COMM_RADIUS,
        network_latency_ms=NETWORK_LATENCY_MS,
        enable_encryption=enable_encryption,
        message_freq_hz=control_freq_hz
    )
    
    # Initialize cybersecurity metrics tracker
    cyber_metrics = CyberMetrics(
        num_drones=num_drones,
        crypto_power_draw_mw=CRYPTO_POWER_DRAW_MW,
        control_freq_hz=control_freq_hz
    )
    
    print(f"\n{'='*70}")
    print(f"SECURE MULTI-UAV COORDINATION SIMULATION")
    print(f"{'='*70}")
    print(f"Fleet Configuration: {num_drones} quadrotors")
    print(f"\n[SECURITY STATUS]")
    if enable_encryption:
        print(f"  Encryption: ENABLED (ChaCha20-Poly1305 AEAD)")
        print(f"  Authentication: ENABLED (Message spoofing protection)")
        print(f"  Key Size: 256-bit")
        print(f"  All drone-to-drone messages will be ENCRYPTED")
    else:
        print(f"  Encryption: DISABLED (Baseline mode)")
        print(f"  WARNING: All messages sent in PLAINTEXT")
    
    print(f"\n[COMMUNICATION PARAMETERS]")
    print(f"  Communication Radius: {COMM_RADIUS} m")
    print(f"  Network Latency: {NETWORK_LATENCY_MS} ms (simulated)")
    print(f"  Consensus Gain: {CONSENSUS_GAIN}")
    print(f"  Message Frequency: {control_freq_hz} Hz")
    print(f"{'='*70}")
    print(f"Encryption logs will appear every 1 second during simulation...")
    print(f"{'='*70}\n")

    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones, output_folder=output_folder)

    dt = 1.0 / control_freq_hz
    start = time.time()

    leader_pos = np.copy(init_xyzs[0])
    leader_vel = np.zeros(3, dtype=float)
    target_pos = np.copy(init_xyzs)
    target_vel = np.zeros((num_drones, 3), dtype=float)

    takeoff_steps = int(TAKEOFF_TIME * control_freq_hz)
    step = 0
    
    # Navigation state machine for smooth transitions
    nav_state = "TAKEOFF"  # States: TAKEOFF, HOVER, ACCELERATE, CRUISE, DECELERATE
    hover_timer = 0
    hover_duration = 1.5  # Hover for 1.5 seconds after reaching waypoint
    accel_time = 2.5  # Accelerate over 2.5 seconds (faster but still smooth)
    decel_distance = 4.0  # Start decelerating 4 meters before waypoint (longer at higher speed)
    state_timer = 0  # Timer for current state

    last_saved_steps = [-999999] * num_drones
    
    # Battery simulation (normalized 0-1, starts at 1.0)
    battery_levels = np.ones(num_drones)
    
    # HUD: Battery percentage display text IDs
    battery_text_ids = [-1] * num_drones  # -1 means not created yet
    
    # Battery state: energy in Joules for each drone
    battery_energy_j = np.ones(num_drones) * BATTERY_CAPACITY_J
    battery_levels = np.ones(num_drones)  # 0.0 to 1.0
    prev_velocities = np.zeros((num_drones, 3))  # For acceleration calculation
    
    # RTB (Return To Base) state
    rtb_active = [False] * num_drones  # Which drones are returning to base
    rtb_target = init_xyzs.copy()  # Store initial positions as base locations
    mission_complete_rtb_triggered = False  # Flag for mission complete RTB
    
    # For consensus tracking
    neighbor_states_cache = {i: [] for i in range(num_drones)}

    try:
        while True:
            keys = p.getKeyboardEvents()

            # === Leader motion (takeoff + autopilot) ===
            if step < takeoff_steps:
                alpha = (step + 1) / max(1, takeoff_steps)
                leader_pos[2] = 0.10*(1 - alpha) + TAKEOFF_Z*alpha
                leader_vel[:] = 0.0
                if alpha >= 0.99 and nav_state == "TAKEOFF":
                    nav_state = "ACCELERATE"
                    state_timer = 0
                    print(f"[NAV] Takeoff complete, entering ACCELERATE state")
            else:
                if current_wp_idx < len(waypoints):
                    wp = waypoints[current_wp_idx]
                    diff = wp - leader_pos
                    dist_xy = math.hypot(diff[0], diff[1])

                    # Check if waypoint reached
                    if dist_xy < 0.5 and abs(diff[2]) < 0.3:
                        if nav_state != "HOVER":
                            print(f"[AUTO] Reached waypoint {current_wp_idx}: {wp}")
                            print(f"[DEBUG] Leader state - pos: {leader_pos}, vel: {leader_vel}")
                            print(f"[NAV] Entering HOVER state for {hover_duration}s")
                            nav_state = "HOVER"
                            hover_timer = 0
                            leader_vel[:] = 0.0  # Stop completely
                        else:
                            # In hover state - wait before moving to next waypoint
                            hover_timer += dt
                            leader_vel[:] = 0.0  # Stay stopped
                            
                            if hover_timer >= hover_duration:
                                print(f"[NAV] Hover complete, moving to next waypoint")
                                current_wp_idx += 1
                                nav_state = "ACCELERATE"
                                hover_timer = 0
                                state_timer = 0
                    else:
                        # Moving toward waypoint - smooth velocity control
                        if dist_xy > 1e-6:
                            dir_xy = diff[:2] / dist_xy
                        else:
                            dir_xy = np.zeros(2, dtype=float)
                        
                        # Calculate target speed based on state with smooth transitions
                        if nav_state == "ACCELERATE":
                            # Smooth acceleration using ease-in-out curve
                            state_timer += dt
                            progress = min(1.0, state_timer / accel_time)
                            # Ease-in-out cubic for smooth acceleration
                            if progress < 0.5:
                                smooth_factor = 4 * progress * progress * progress
                            else:
                                smooth_factor = 1 - pow(-2 * progress + 2, 3) / 2
                            target_speed_xy = AUTO_SPEED_XY * smooth_factor * 0.80  # Max 80% speed
                            
                            if progress >= 0.98:
                                nav_state = "CRUISE"
                                state_timer = 0
                                print(f"[NAV] Entering CRUISE state at {target_speed_xy:.2f} m/s")
                        
                        elif dist_xy < decel_distance and nav_state == "CRUISE":
                            # Start deceleration
                            nav_state = "DECELERATE"
                            state_timer = 0
                            print(f"[NAV] Entering DECELERATE state at {dist_xy:.2f}m from target")
                        
                        if nav_state == "DECELERATE":
                            # Smooth deceleration based on distance
                            state_timer += dt
                            progress = dist_xy / decel_distance  # 1.0 at start, 0.0 at waypoint
                            # Ease-out curve for smooth deceleration
                            smooth_factor = 1 - (1 - progress) * (1 - progress)
                            smooth_factor = max(0.25, smooth_factor)  # Min 25% speed
                            target_speed_xy = AUTO_SPEED_XY * smooth_factor * 0.80
                        elif nav_state == "CRUISE":
                            # Constant cruising speed
                            state_timer += dt
                            target_speed_xy = AUTO_SPEED_XY * 0.80  # Cruise at 80% max speed
                        else:
                            # Accelerating or default
                            target_speed_xy = AUTO_SPEED_XY * 0.80
                        
                        # Apply smooth velocity changes
                        leader_vel[0] = dir_xy[0] * target_speed_xy
                        leader_vel[1] = dir_xy[1] * target_speed_xy
                        
                        # Z velocity: proportional control with deadband
                        if abs(diff[2]) > 0.15:
                            # Faster Z velocity for quicker altitude changes
                            z_speed = np.clip(diff[2], -AUTO_SPEED_Z * 0.7, AUTO_SPEED_Z * 0.7)
                            leader_vel[2] = z_speed
                        else:
                            leader_vel[2] = 0.0

                        leader_pos[:2] = clamp_xy(leader_pos[:2] + leader_vel[:2] * dt)
                        leader_pos[2]  = np.clip(leader_pos[2] + leader_vel[2] * dt, Z_MIN, Z_MAX)
                else:
                    # Mission complete: trigger RTB for all drones
                    if not mission_complete_rtb_triggered:
                        print(f"\n{'='*70}")
                        print(f"[MISSION COMPLETE] All {len(waypoints)} waypoints visited!")
                        print(f"[RTB] Initiating Return To Base for all drones...")
                        print(f"{'='*70}\n")
                        # Trigger RTB for all drones
                        for j in range(num_drones):
                            rtb_active[j] = True
                        mission_complete_rtb_triggered = True
                    
                    # Stop leader movement - drones will navigate to base individually
                    leader_vel[:] = 0.0
                
                # Validate leader position for NaN/inf (CRITICAL SAFETY CHECK)
                if not np.isfinite(leader_pos).all():
                    print(f"[ERROR] Invalid leader position detected: {leader_pos}")
                    leader_pos = np.array([0.0, 0.0, TAKEOFF_Z])
                    leader_vel[:] = 0.0
                    print(f"[RECOVERY] Leader reset to safe position: {leader_pos}")

            # Step physics
            obs, _, _, _ = env.step(action)
            
            # Extract drone positions and velocities
            drone_positions = np.zeros((num_drones, 3))
            drone_velocities = np.zeros((num_drones, 3))
            
            for j in range(num_drones):
                drone_positions[j] = obs[str(j)]["state"][0:3]
                drone_velocities[j] = obs[str(j)]["state"][10:13]
            
            # === Stability Monitoring (CRITICAL SAFETY CHECK) ===
            for j in range(num_drones):
                state = obs[str(j)]["state"]
                roll = state[7]   # Roll in radians
                pitch = state[8]  # Pitch in radians
                roll_deg = np.degrees(roll)
                pitch_deg = np.degrees(pitch)
                
                # Warn if angles exceed safe limits
                if abs(roll_deg) > 30 or abs(pitch_deg) > 30:
                    print(f"[STABILITY WARNING] Drone {j}: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°")
                    
                # Emergency stabilization if critically unstable
                if abs(roll_deg) > 60 or abs(pitch_deg) > 60:
                    print(f"[EMERGENCY] Drone {j} critically unstable! Triggering RTB.")
                    rtb_active[j] = True
            
            # Debug logging: Print all drone Z heights periodically (every second)
            if step % 30 == 0 and step > 0:
                z_heights = [f"{drone_positions[j][2]:.2f}" for j in range(num_drones)]
                print(f"[DEBUG] Step {step} - All drone Z heights: {z_heights}")

            # === Formation target positions for all drones ===
            raw_targets = np.zeros((num_drones, 3), dtype=float)
            rtb_overrides = {}
            
            for j in range(num_drones):
                # Check if drone needs RTB
                rtb_pos, rtb_vel, rtb_status = check_and_handle_rtb(
                    j,
                    battery_levels[j],
                    drone_positions[j],
                    rtb_target[j],
                    rtb_active,
                    leader_vel
                )
                
                if rtb_status != "NORMAL":
                    # RTB overrides formation
                    raw_targets[j] = rtb_pos
                    rtb_overrides[j] = (rtb_vel, rtb_status)
                else:
                    # Normal formation following
                    off_xy = W_OFFSETS[j]
                    tgt_xy = leader_pos[:2] + off_xy
                    tgt_z = leader_pos[2]
                    raw_targets[j] = np.array([tgt_xy[0], tgt_xy[1], tgt_z], dtype=float)
                
                # Validate follower target position (CRITICAL SAFETY CHECK)
                if not np.isfinite(raw_targets[j]).all() or raw_targets[j, 2] < Z_MIN:
                    print(f"[WARNING] Drone {j} target invalid: {raw_targets[j]}")
                    raw_targets[j] = np.array([drone_positions[j][0], 
                                                drone_positions[j][1], 
                                                TAKEOFF_Z])
                    print(f"[RECOVERY] Drone {j} target reset to current XY, z={TAKEOFF_Z}")

            # ===== SECURE COMMUNICATION & CONSENSUS PHASE =====
            control_loop_start = time.perf_counter()
            
            # 1. Neighbor discovery
            neighbors = secure_comm.discover_neighbors(drone_positions)
            
            # 2. Broadcast encrypted state messages
            encryption_times = []
            
            # Log encryption activity every 30 steps (every second at 30Hz)
            show_encryption_log = (step % 30 == 0 and step > 0)
            
            for j in range(num_drones):
                state_msg = secure_comm.create_state_message(
                    drone_id=j,
                    position=drone_positions[j],
                    velocity=drone_velocities[j],
                    battery=battery_levels[j],
                    timestamp=step * dt
                )
                
                # Show plaintext message before encryption (verbose logging)
                if show_encryption_log:  # Show for all drones
                    print(f"\n[ENCRYPTION] Step {step} - Drone {j}")
                    print(f"  Plaintext: pos=[{drone_positions[j][0]:.2f}, {drone_positions[j][1]:.2f}, {drone_positions[j][2]:.2f}]")
                    plaintext_json = json.dumps(state_msg)
                    print(f"  Message size: {len(plaintext_json)} bytes")
                
                nonce, ciphertext, enc_time = secure_comm.encrypt_message(state_msg)
                encryption_times.append(enc_time)
                
                # Show encryption results (verbose logging)
                if show_encryption_log:
                    if secure_comm.enable_encryption:
                        cipher_hex = ciphertext[:16].hex() if len(ciphertext) >= 16 else ciphertext.hex()
                        nonce_hex = nonce.hex()
                        print(f"  [ENCRYPTED] Ciphertext (first 16 bytes): {cipher_hex}...")
                        print(f"  Nonce: {nonce_hex}")
                        print(f"  Encryption time: {enc_time:.4f} ms")
                        print(f"  Neighbors: {len(neighbors.get(j, []))} drones will receive encrypted message")
                    else:
                        print(f"  [BASELINE MODE] Message sent in PLAINTEXT (no encryption)")
                        if enable_encryption and not secure_comm.enable_encryption:
                            print(f"  [WARNING] Encryption was requested but cryptography library not available!")
                
                secure_comm.broadcast_to_neighbors(j, nonce, ciphertext, enc_time)
                
                # Log crypto operation for battery tracking
                cyber_metrics.log_crypto_operation(j, enc_time / 1000.0)
            
            # 3. Receive and decrypt neighbor messages
            for j in range(num_drones):
                received_messages = secure_comm.receive_messages(j)
                
                # Show decryption activity (verbose logging)
                if show_encryption_log and len(received_messages) > 0:
                    print(f"\n[DECRYPTION] Step {step} - Drone {j}")
                    print(f"  Received {len(received_messages)} encrypted messages from neighbors")
                
                neighbor_states_cache[j] = []
                for idx, (msg, latency) in enumerate(received_messages):
                    neighbor_states_cache[j].append(msg)
                    
                    # Show first decrypted message details (verbose logging)
                    if show_encryption_log and idx == 0:
                        if enable_encryption:
                            print(f"  [DECRYPTED] Message from drone {msg['drone_id']}")
                            print(f"    Position: [{msg['pos'][0]:.2f}, {msg['pos'][1]:.2f}, {msg['pos'][2]:.2f}]")
                            print(f"    Velocity: [{msg['vel'][0]:.2f}, {msg['vel'][1]:.2f}, {msg['vel'][2]:.2f}]")
                            print(f"    Total latency: {latency:.4f} ms")
                            print(f"    Authentication: VERIFIED ✓")
                        else:
                            print(f"  [BASELINE MODE] Received PLAINTEXT message from drone {msg['drone_id']}")
                    
                    # Log latency metrics
                    cyber_metrics.log_communication_latency(
                        encryption_time_ms=0.0,  # Already tracked
                        decryption_time_ms=0.0,  # Already tracked
                        total_latency_ms=latency
                    )
                    
                    # Log decryption as crypto operation
                    cyber_metrics.log_crypto_operation(j, latency / 1000.0)
            
            # 4. Compute formation velocities + consensus adjustments
            for j in range(num_drones):
                if j in rtb_overrides:
                    # RTB mode: use RTB velocity instead of formation velocity
                    rtb_vel, rtb_status = rtb_overrides[j]
                    target_vel[j] = rtb_vel
                else:
                    # Normal mode: formation velocity + consensus
                    base_vel = leader_vel.copy()
                    consensus_adj = compute_consensus_adjustment(
                        drone_id=j,
                        drone_vel=drone_velocities[j],
                        neighbor_states=neighbor_states_cache[j]
                    )
                    target_vel[j] = base_vel + consensus_adj
            
            control_loop_end = time.perf_counter()
            control_loop_delay_ms = (control_loop_end - control_loop_start) * 1000.0
            cyber_metrics.log_control_loop_delay(control_loop_delay_ms)
            
            # Log formation error
            cyber_metrics.log_formation_error(drone_positions, raw_targets)
            
            # Track crypto operations in this timestep
            crypto_ops_this_step = np.zeros(num_drones)
            # (This is populated during encryption/decryption - approximately 2 ops per drone per step)
            # For now, estimate: each drone sends 1 message and receives ~4 messages = 5 crypto ops
            crypto_ops_this_step[:] = 5.0  # Rough estimate
            
            # Update battery energy based on propulsion physics + crypto
            battery_energy_j, battery_levels, power_w = update_battery_energy(
                drone_velocities,
                prev_velocities,
                battery_energy_j,
                dt,
                crypto_ops_this_step
            )
            
            # Store current velocities for next acceleration calculation
            prev_velocities[:] = drone_velocities

            # Manual capture
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                save_all_drone_cameras(obs, step)

            # AUTO: detect orange and save with green box
            detect_orange_and_save(obs, step, last_saved_steps)

            # Collision avoidance and clamp
            target_pos[:] = raw_targets
            for j in range(num_drones):
                pos_j = obs[str(j)]["state"][0:3]
                sep_push = np.zeros(2, dtype=float)
                for k in range(num_drones):
                    if k == j:
                        continue
                    pos_k = obs[str(k)]["state"][0:3]
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

            # Controllers (all quadrotors with consensus-adjusted velocities)
            for j in range(num_drones):
                raw = ctrls[j].computeControlFromState(
                    control_timestep=dt,
                    state=obs[str(j)]["state"],
                    target_pos=target_pos[j],
                    target_rpy=np.zeros(3),
                    target_vel=target_vel[j],  # Apply consensus velocity
                    target_rpy_rates=np.zeros(3)
                )
                action[j, :] = raw[0]
                logger.log(drone=j, timestamp=step * dt, state=obs[str(j)]["state"])
            
            # Print periodic cybersecurity metrics report (every 5 seconds)
            cyber_metrics.print_periodic_report(current_time=step * dt)
            
            # Update battery HUD display
            battery_text_ids = update_battery_hud(drone_positions, battery_levels, battery_text_ids, rtb_active)

            # Follow camera on leader
            update_follow_cam(leader_pos.copy(), keys)

            env.render()
            if gui:
                sync(step, start, dt)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Clean up HUD elements
        for text_id in battery_text_ids:
            if text_id != -1:
                try:
                    p.removeUserDebugItem(text_id)
                except:
                    pass
        
        env.close()
        
        # Save standard flight logs
        logger.save()
        logger.save_as_csv("auto_orange_detect_wide")
        
        # Export cybersecurity metrics
        print("\n" + "="*70)
        print("EXPORTING CYBERSECURITY METRICS")
        print("="*70)
        
        metrics_data = cyber_metrics.export_to_dict()
        
        # Save metrics to JSON
        metrics_filename = os.path.join(output_folder, "cyber_metrics.json")
        os.makedirs(output_folder, exist_ok=True)
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"✓ Saved metrics to: {metrics_filename}")
        
        # Save communication timing stats to JSON
        comm_stats = secure_comm.get_timing_stats()
        comm_stats_filename = os.path.join(output_folder, "comm_timing_stats.json")
        with open(comm_stats_filename, 'w') as f:
            json.dump(comm_stats, f, indent=2)
        print(f"✓ Saved communication stats to: {comm_stats_filename}")
        
        # Print final summary report
        cyber_metrics.print_periodic_report(force=True)
        
        print("\nCommunication Timing Statistics:")
        print(f"  Encryption:  {comm_stats['mean_encryption_ms']:.3f} ms (mean), "
              f"{comm_stats['max_encryption_ms']:.3f} ms (max)")
        print(f"  Decryption:  {comm_stats['mean_decryption_ms']:.3f} ms (mean), "
              f"{comm_stats['max_decryption_ms']:.3f} ms (max)")
        print(f"  Total Latency: {comm_stats['mean_total_latency_ms']:.3f} ms (mean), "
              f"{comm_stats['max_total_latency_ms']:.3f} ms (max)")
        
        print("\nBattery Impact Summary:")
        for j in range(num_drones):
            print(f"  Drone {j} (Quadrotor): {metrics_data['crypto_energy_mj'][j]:.2f} mJ "
                  f"({metrics_data['total_crypto_operations'][j]} operations)")
        
        print("="*70)
        
        if plot:
            logger.plot()
        
        print("\nSimulation completed. All logs and metrics saved.")
        print(f"Mode: {'ENCRYPTED' if enable_encryption else 'BASELINE (no encryption)'}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Secure Multi-UAV Coordination with Consensus and Encryption"
    )
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, metavar="",
                       help="Enable PyBullet GUI")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="",
                       help="Plot results after simulation")
    parser.add_argument("--encryption", default=ENABLE_ENCRYPTION, type=str2bool, metavar="",
                       help="Enable ChaCha20-Poly1305 encryption (default: True)")
    args = parser.parse_args()
    
    # Map encryption argument to enable_encryption parameter
    run_args = vars(args)
    run_args['enable_encryption'] = run_args.pop('encryption')
    
    run(**run_args)
