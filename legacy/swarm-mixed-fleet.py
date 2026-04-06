"""
Save the Planet: Quadcopter Fleet Waste Detection Mission

- 3 Quadcopters in formation for detailed waste inspection
- Desert environment with dunes and obstacles
- Autonomous waypoint navigation with orange waste detection
- Coordinated multi-drone operation

Controls:
  Movement:        AUTOPILOT (no keyboard needed).
  Manual save:     Press C to save camera image of every drone (PNG files).
  Camera:          Arrow keys (yaw/pitch), Z/X (zoom - unlimited), R (reset).
                   F - Toggle free camera mode (WASD to move, Q/E for height)
                   Mouse - Click+drag to rotate, Shift+drag to pan, Ctrl+drag to zoom
  Exit:            Ctrl + C in terminal / closing window.
"""

import time
import math
import argparse
import os
import sys
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

# Import desert utilities
from desert_utils import create_desert_terrain, spawn_desert_obstacles

# ========= Drone Configuration =========
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_leader", "color": [1.0, 0.2, 0.2, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_follower_1", "color": [0.2, 0.2, 1.0, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_follower_2", "color": [0.2, 0.8, 0.2, 1.0]},
]

NUM_DRONES = len(DRONE_CONFIGS)

# ========= Simulation Defaults =========
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False

DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_DURATION_SEC = 3600
DEFAULT_OUTPUT_FOLDER = "results"

# ========= Flight Parameters =========
TAKEOFF_Z_QUAD = 2.5
TAKEOFF_TIME = 5.0

AUTO_SPEED_XY = 1.5
AUTO_SPEED_Z  = 1.0

WORLD_XY_LIMIT = 20.0
Z_MIN, Z_MAX = 0.8, 6.0

# Formation for quadcopters (3 quads - V formation)
QUAD_FORMATION_OFFSETS = np.array([
    [0.0, 0.0],     # Leader
    [2.0, -1.5],    # Follower 1 (right-back)
    [-2.0, -1.5],   # Follower 2 (left-back)
], dtype=float)

# Collision avoidance
MIN_SEP = 1.5
AVOID_GAIN = 0.6
MAX_PUSH = 0.8

# Camera parameters - UNLIMITED ZOOM AND FREE MOVEMENT
CAM_DIST_DEFAULT  = 15.0
CAM_YAW_DEFAULT   = 0.0
CAM_PITCH_DEFAULT = -60.0

cam_dist  = CAM_DIST_DEFAULT
cam_yaw   = CAM_YAW_DEFAULT
cam_pitch = CAM_PITCH_DEFAULT
cam_target_x = 0.0
cam_target_y = 0.0
cam_target_z = 0.0

CAM_YAW_STEP   = 2.0
CAM_PITCH_STEP = 1.5
CAM_DIST_STEP  = 0.5
CAM_TARGET_STEP = 1.0  # Movement step for free camera
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0
# UNLIMITED ZOOM - removed distance limits

# Onboard camera
CAM_WIDTH  = 200
CAM_HEIGHT = 150
CAM_FOV    = 45.0
CAM_NEAR   = 0.1
CAM_FAR    = 100.0

CAMERA_OUTPUT_DIR = "camera_frames"

# Free camera mode (toggle with F key)
free_camera_mode = False

# ========= Waste positions (spread across desert) =========
WASTE_POSITIONS = [
    (-15.0, -12.0, 0.0),
    (-15.0,   12.0, 0.0),
    (   0.0,  -15.0, 0.0),
    (   0.0,   15.0, 0.0),
    (  15.0,  -12.0, 0.0),
    (  15.0,   12.0, 0.0),
    (  -8.0,    0.0, 0.0),
    (   8.0,    0.0, 0.0),
]

# ========= Colors for waste =========
WASTE_COLORS = [
    "orange", "blue", "orange", "gray",
    "orange", "green", "orange", "gray",
]

COLOR_MAP = {
    "orange": [1.0, 0.5, 0.0, 1.0],
    "blue":   [0.1, 0.4, 1.0, 1.0],
    "gray":   [0.7, 0.7, 0.7, 1.0],
    "green":  [0.2, 0.8, 0.2, 1.0],
}

WASTE_BODY_IDS = []
MIN_STEPS_BETWEEN_CAPTURES = 40


def clamp_xy(xy):
    return np.clip(xy, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)


def update_follow_cam(target_pos_3d, keys):
    """Update camera with unlimited zoom and free movement capabilities.
    
    Press F to toggle between follow mode (follows drones) and free mode (manual control).
    In free mode: WASD to move camera target, Q/E for up/down.
    """
    global cam_dist, cam_yaw, cam_pitch, cam_target_x, cam_target_y, cam_target_z, free_camera_mode

    # Toggle free camera mode
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_camera_mode = not free_camera_mode
        if free_camera_mode:
            # Initialize free camera at current drone position
            cam_target_x = target_pos_3d[0]
            cam_target_y = target_pos_3d[1]
            cam_target_z = target_pos_3d[2]
            print("[Camera] FREE MODE - Use WASD to move camera target, Q/E for height")
        else:
            print("[Camera] FOLLOW MODE - Camera follows drone leader")

    # Camera rotation controls (work in both modes)
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        cam_yaw -= CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        cam_yaw += CAM_YAW_STEP

    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch + CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch - CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)

    # UNLIMITED ZOOM - removed distance limits
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
        cam_dist = max(0.1, cam_dist - CAM_DIST_STEP)  # Only prevent negative values
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
        cam_dist = cam_dist + CAM_DIST_STEP  # No upper limit

    # Free camera movement (WASD + Q/E)
    if free_camera_mode:
        # W/S - forward/backward (relative to yaw)
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            cam_target_x += CAM_TARGET_STEP * np.cos(np.radians(cam_yaw))
            cam_target_y += CAM_TARGET_STEP * np.sin(np.radians(cam_yaw))
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            cam_target_x -= CAM_TARGET_STEP * np.cos(np.radians(cam_yaw))
            cam_target_y -= CAM_TARGET_STEP * np.sin(np.radians(cam_yaw))
        
        # A/D - left/right strafe (perpendicular to yaw)
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            cam_target_x += CAM_TARGET_STEP * np.cos(np.radians(cam_yaw + 90))
            cam_target_y += CAM_TARGET_STEP * np.sin(np.radians(cam_yaw + 90))
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            cam_target_x -= CAM_TARGET_STEP * np.cos(np.radians(cam_yaw + 90))
            cam_target_y -= CAM_TARGET_STEP * np.sin(np.radians(cam_yaw + 90))
        
        # Q/E - up/down
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            cam_target_z += CAM_TARGET_STEP * 0.5
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            cam_target_z -= CAM_TARGET_STEP * 0.5

    # Reset camera
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist  = CAM_DIST_DEFAULT
        cam_yaw   = CAM_YAW_DEFAULT
        cam_pitch = CAM_PITCH_DEFAULT
        free_camera_mode = False
        print("[Camera] Reset to default follow mode")

    # Determine camera target position
    if free_camera_mode:
        camera_target = [cam_target_x, cam_target_y, cam_target_z]
    else:
        camera_target = target_pos_3d.tolist()

    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=camera_target
    )


def get_drone_camera_rgb(obs_drone, client_id):
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
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id
    )

    rgba = np.reshape(img[2], (CAM_HEIGHT, CAM_WIDTH, 4))
    rgb = rgba[:, :, :3].astype(np.uint8)
    return rgb


def save_all_drone_cameras(obs, frame_idx, configs, client_id):
    """Manual C key: save all drones' camera images."""
    if not _HAS_CV2:
        print("[Camera] OpenCV not installed, can't save images.")
        return

    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)

    for j, state in enumerate(obs):
        rgb = get_drone_camera_rgb(state, client_id)
        if rgb is None:
            continue
        role = configs[j]["role"]
        filename = os.path.join(
            CAMERA_OUTPUT_DIR,
            f"manual_{role}_{frame_idx:05d}.png"
        )
        cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[Camera] [Manual] Saved {filename}")


def detect_orange_and_save(obs, frame_idx, last_saved_steps, configs, client_id):
    """AUTO: Detect orange waste and save images with bounding box."""
    if not _HAS_CV2:
        return

    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)

    # Color ranges in HSV
    lower_orange = np.array([5, 150, 100])
    upper_orange = np.array([25, 255, 255])
    
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])
    
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    
    lower_green = np.array([40, 100, 50])
    upper_green = np.array([80, 255, 255])

    for j, state in enumerate(obs):
        if frame_idx - last_saved_steps[j] < MIN_STEPS_BETWEEN_CAPTURES:
            continue

        rgb = get_drone_camera_rgb(state, client_id)
        if rgb is None:
            continue

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Detect all colors
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = np.ones((3, 3), np.uint8)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_DILATE, kernel, iterations=1)
        
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, kernel, iterations=1)
        
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_DILATE, kernel, iterations=1)
        
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # Find contours for orange
        contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_something = False
        detected_orange = False
        
        # Draw green rectangles around orange objects (target waste)
        for contour in contours_orange:
            area = cv2.contourArea(contour)
            if area >= 80:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_something = True
                detected_orange = True
        
        # Draw red rectangles around non-orange objects
        for contour in contours_blue:
            area = cv2.contourArea(contour)
            if area >= 80:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                detected_something = True
        
        for contour in contours_gray:
            area = cv2.contourArea(contour)
            if area >= 80:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                detected_something = True
        
        for contour in contours_green:
            area = cv2.contourArea(contour)
            if area >= 80:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                detected_something = True
        
        if not detected_something:
            continue

        role = configs[j]["role"]
        if detected_orange:
            filename = os.path.join(
                CAMERA_OUTPUT_DIR,
                f"waste_target_{role}_{frame_idx:05d}.png"
            )
            cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            last_saved_steps[j] = frame_idx
            print(f"[Camera] [AUTO] {role} detected WASTE TARGET (orange) → Saved {filename}")
        else:
            filename = os.path.join(
                CAMERA_OUTPUT_DIR,
                f"other_object_{role}_{frame_idx:05d}.png"
            )
            cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            last_saved_steps[j] = frame_idx
            print(f"[Camera] [AUTO] {role} detected other object → Saved {filename}")


def spawn_waste_objects():
    """Spawn waste cubes at fixed positions with various colors."""
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
    """Leader waypoints: fly above waste positions."""
    waypoints = []
    for (x, y, z_ground) in WASTE_POSITIONS:
        waypoints.append(np.array([x, y, TAKEOFF_Z_QUAD], dtype=float))
    return waypoints


def run(
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER
):
    """Run the quadcopter fleet simulation."""
    
    # Initial positions for 3 quadcopters
    init_xyzs = np.array([
        [0.0, 0.0, 0.10],           # Quad leader
        [2.0, -1.5, 0.10],          # Quad follower 1
        [-2.0, -1.5, 0.10],         # Quad follower 2
    ], dtype=float)
    init_rpys = np.zeros((NUM_DRONES, 3), dtype=float)

    # Create environment for quadcopter fleet
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=physics,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        user_debug_gui=False
    )
    
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
        # Keep mouse control enabled for better interaction
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=env.CLIENT)
    
    # Hide the default checkered plane completely
    p.changeVisualShape(
        env.PLANE_ID,
        -1,
        rgbaColor=[0.9, 0.85, 0.75, 0.0],  # Fully transparent
        physicsClientId=env.CLIENT
    )
    
    # Create solid color sand floor (no tiling issues)
    print("[World] Creating solid sand floor...")
    plane_size = 50.0  # Adjusted to match 38.4 unit terrain (256 pixels * 0.15 scale)
    floor_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[plane_size/2, plane_size/2, 0.01],
        rgbaColor=[0.87, 0.72, 0.53, 1.0],  # Sandy tan color
        physicsClientId=env.CLIENT
    )
    
    floor_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=floor_visual,
        basePosition=[0, 0, -0.05],
        physicsClientId=env.CLIENT
    )
    print("[World] Solid sand floor created")
    
    # Color the drones according to their roles
    for i, config in enumerate(DRONE_CONFIGS):
        try:
            p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=config["color"], physicsClientId=env.CLIENT)
        except:
            pass  # Ignore if visual change fails
    
    # Create desert terrain dunes with desert_sand.png texture
    print("[World] Creating small desert with gentle dunes...")
    terrain_id = create_desert_terrain(
        heightmap_path="assets/terrain_desert_dunes.png",  # Desert dunes terrain
        texture_path="assets/desert_sand.png",
        terrain_scale=(0.15, 0.15, 1.5)  # Small gentle dunes
    )
    
    # Spawn desert obstacles
    print("[World] Spawning desert obstacles...")
    obstacle_ids = spawn_desert_obstacles(
        num_rocks=12,
        num_vegetation=6,
        area_size=35.0,
        exclude_positions=WASTE_POSITIONS,
        min_distance=4.0
    )
    
    # Spawn waste objects
    spawn_waste_objects()
    
    # Build waypoints for quad leader
    waypoints = build_search_waypoints()
    current_wp_idx = 0

    # Create controllers for each quadcopter
    ctrls = []
    for config in DRONE_CONFIGS:
        ctrls.append(DSLPIDControl(drone_model=config["model"]))
    
    # Actions for all drones
    action = np.zeros((NUM_DRONES, 4))

    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=NUM_DRONES, output_folder=output_folder)

    dt = 1.0 / control_freq_hz
    start = time.time()

    # Motion state for quad leader
    quad_leader_pos = np.copy(init_xyzs[0])
    quad_leader_vel = np.zeros(3, dtype=float)
    
    target_pos = np.copy(init_xyzs)

    takeoff_steps = int(TAKEOFF_TIME * control_freq_hz)
    step = 0

    last_saved_steps = [-999999] * NUM_DRONES

    print(f"\n[Mission] Starting 'Save the Planet' with quadcopter fleet:")
    print(f"  - 3 Quadcopters in V-formation for detailed inspection")
    print(f"  - {len(WASTE_POSITIONS)} waste sites to survey")
    print(f"  - Desert terrain with enhanced dunes and obstacles")
    print(f"\n[Camera Controls]")
    print(f"  - Arrow keys: Rotate camera")
    print(f"  - Z/X: Zoom in/out (unlimited)")
    print(f"  - F: Toggle free camera mode (then use WASD + Q/E)")
    print(f"  - Mouse: Drag to rotate, Shift+drag to pan, Ctrl+drag to zoom")
    print(f"  - R: Reset camera\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)

            # Get observations (step environment later)
            obs = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]
            
            # === Quad Leader Motion (waypoint navigation) ===
            if step < takeoff_steps:
                alpha = (step + 1) / max(1, takeoff_steps)
                quad_leader_pos[2] = 0.10*(1 - alpha) + TAKEOFF_Z_QUAD*alpha
                quad_leader_vel[:] = 0.0
            else:
                if current_wp_idx < len(waypoints):
                    wp = waypoints[current_wp_idx]
                    diff = wp - quad_leader_pos
                    dist_xy = math.hypot(diff[0], diff[1])

                    if dist_xy < 1.0 and abs(diff[2]) < 0.4:
                        print(f"[Mission] Quad leader reached waypoint {current_wp_idx}: {wp}")
                        current_wp_idx += 1
                        quad_leader_vel[:] = 0.0
                    else:
                        if dist_xy > 1e-6:
                            dir_xy = diff[:2] / dist_xy
                        else:
                            dir_xy = np.zeros(2, dtype=float)

                        quad_leader_vel[0] = dir_xy[0] * AUTO_SPEED_XY
                        quad_leader_vel[1] = dir_xy[1] * AUTO_SPEED_XY
                        quad_leader_vel[2] = np.clip(diff[2], -AUTO_SPEED_Z, AUTO_SPEED_Z)

                        quad_leader_pos[:2] = clamp_xy(quad_leader_pos[:2] + quad_leader_vel[:2] * dt)
                        quad_leader_pos[2]  = np.clip(quad_leader_pos[2] + quad_leader_vel[2] * dt, Z_MIN, Z_MAX)
                else:
                    quad_leader_vel[:] = 0.0

            # === Quad Formation Targets ===
            for i in range(NUM_DRONES):  # All three quadcopters
                off_xy = QUAD_FORMATION_OFFSETS[i]
                tgt_xy = quad_leader_pos[:2] + off_xy
                tgt_z  = quad_leader_pos[2]
                target_pos[i] = np.array([tgt_xy[0], tgt_xy[1], tgt_z], dtype=float)

            # Collision avoidance (inter-drone)
            for i in range(NUM_DRONES):
                pos_i = obs[i][0:3]
                sep_push = np.zeros(2, dtype=float)
                for j in range(NUM_DRONES):
                    if j == i:
                        continue
                    pos_j = obs[j][0:3]
                    dx = pos_i[0] - pos_j[0]
                    dy = pos_i[1] - pos_j[1]
                    dist = math.hypot(dx, dy)
                    if dist < MIN_SEP and dist > 1e-6:
                        push = AVOID_GAIN * (MIN_SEP - dist) / dist * np.array([dx, dy])
                        sep_push += push
                n = np.linalg.norm(sep_push)
                if n > MAX_PUSH and n > 0:
                    sep_push = sep_push * (MAX_PUSH / n)
                target_pos[i, 0:2] += sep_push
                target_pos[i, 0:2] = clamp_xy(target_pos[i, 0:2])
                target_pos[i, 2]   = np.clip(target_pos[i, 2], Z_MIN, Z_MAX)

            # Compute controls for each drone
            for i, ctrl in enumerate(ctrls):
                raw = ctrl.computeControlFromState(
                    control_timestep=dt,
                    state=obs[i],
                    target_pos=target_pos[i],
                    target_rpy=np.zeros(3),
                    target_vel=np.zeros(3),
                    target_rpy_rates=np.zeros(3)
                )
                action[i, :] = raw[0]
                logger.log(drone=i, timestamp=step * dt, state=obs[i])

            # Step environment with all drone actions
            obs, _, _, _, _ = env.step(action)

            # Manual camera save
            # if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
            #     save_all_drone_cameras(obs, step, DRONE_CONFIGS, env.CLIENT)

            # Auto orange detection - DISABLED for performance
            # detect_orange_and_save(obs, step, last_saved_steps, DRONE_CONFIGS, env.CLIENT)

            # Follow camera on quad leader
            update_follow_cam(quad_leader_pos.copy(), keys)

            # Render
            env.render()
            if gui:
                sync(step, start, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        logger.save()
        logger.save_as_csv("quad_fleet_mission")
        if plot:
            logger.plot()
        print("\n[Mission] Complete. Logs saved.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadcopter fleet waste detection mission")
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
