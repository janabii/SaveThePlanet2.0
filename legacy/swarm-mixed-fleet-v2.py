"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission

- 1 Fixed-wing drone for wide-area scouting (high altitude, fast lawnmower pattern)
- 3 Quadcopters in V-formation for detailed waste inspection
- Desert environment with dunes and obstacles
- Autonomous waypoint navigation with orange waste detection

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
import numpy as np
import pybullet as p

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
from desert_utils import create_desert_terrain, spawn_desert_obstacles

# ========= Quadcopter Configuration =========
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_leader",     "color": [1.0, 0.2, 0.2, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_follower_1", "color": [0.2, 0.2, 1.0, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_follower_2", "color": [0.2, 0.8, 0.2, 1.0]},
]
NUM_DRONES = len(DRONE_CONFIGS)

# ========= Fixed-wing Configuration =========
FW_ALTITUDE  = 5.5
FW_SPEED     = 4.0
FW_WP_RADIUS = 2.5
FW_DETECTION_RADIUS = 6.0

# Lawnmower survey pattern
FW_WAYPOINTS = np.array([
    [-18.0, -18.0, FW_ALTITUDE],
    [ 18.0, -18.0, FW_ALTITUDE],
    [ 18.0,  -6.0, FW_ALTITUDE],
    [-18.0,  -6.0, FW_ALTITUDE],
    [-18.0,   6.0, FW_ALTITUDE],
    [ 18.0,   6.0, FW_ALTITUDE],
    [ 18.0,  18.0, FW_ALTITUDE],
    [-18.0,  18.0, FW_ALTITUDE],
], dtype=float)

# ========= Simulation Defaults =========
DEFAULT_PHYSICS            = Physics("pyb")
DEFAULT_GUI                = True
DEFAULT_PLOT               = False
DEFAULT_USER_DEBUG_GUI     = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_DURATION_SEC       = 3600
DEFAULT_OUTPUT_FOLDER      = "results"

# ========= Quad Flight Parameters =========
TAKEOFF_Z_QUAD = 2.5
TAKEOFF_TIME   = 5.0
AUTO_SPEED_XY  = 1.5
AUTO_SPEED_Z   = 1.0
WORLD_XY_LIMIT = 20.0
Z_MIN, Z_MAX   = 0.8, 6.0

QUAD_FORMATION_OFFSETS = np.array([
    [ 0.0,  0.0],
    [ 2.0, -1.5],
    [-2.0, -1.5],
], dtype=float)

MIN_SEP    = 1.5
AVOID_GAIN = 0.6
MAX_PUSH   = 0.8

# ========= Camera =========
CAM_DIST_DEFAULT  = 20.0
CAM_YAW_DEFAULT   = 0.0
CAM_PITCH_DEFAULT = -45.0
cam_dist = CAM_DIST_DEFAULT
cam_yaw  = CAM_YAW_DEFAULT
cam_pitch = CAM_PITCH_DEFAULT
cam_target_x = cam_target_y = cam_target_z = 0.0
CAM_YAW_STEP    = 2.0
CAM_PITCH_STEP  = 1.5
CAM_DIST_STEP   = 0.5
CAM_TARGET_STEP = 1.0
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0
free_camera_mode = False
CAM_WIDTH, CAM_HEIGHT = 200, 150
CAM_FOV, CAM_NEAR, CAM_FAR = 45.0, 0.1, 100.0
CAMERA_OUTPUT_DIR = "camera_frames"

# ========= Waste =========
WASTE_POSITIONS = [
    (-15.0, -12.0, 0.0), (-15.0,  12.0, 0.0),
    (  0.0, -15.0, 0.0), (  0.0,  15.0, 0.0),
    ( 15.0, -12.0, 0.0), ( 15.0,  12.0, 0.0),
    ( -8.0,   0.0, 0.0), (  8.0,   0.0, 0.0),
]
WASTE_COLORS = ["orange", "blue", "orange", "gray", "orange", "green", "orange", "gray"]
COLOR_MAP = {
    "orange": [1.0, 0.5, 0.0, 1.0],
    "blue":   [0.1, 0.4, 1.0, 1.0],
    "gray":   [0.7, 0.7, 0.7, 1.0],
    "green":  [0.2, 0.8, 0.2, 1.0],
}
WASTE_BODY_IDS   = []
fw_detected_ids  = set()
MIN_STEPS_BETWEEN_CAPTURES = 40


# ─────────────────────────────────────────────────────────────
# FIXED-WING: spawn as 4 separate parts that move together
#
# Layout (top view, flying along +X):
#
#         [tail_v]  <- vertical fin (up)
#   [htail]  [fuselage] ========>  [nose]
#         [wings span Y]
#
# All 4 parts are moved every frame to stay glued together.
# ─────────────────────────────────────────────────────────────

def _make_box(half, color, cid, mass=0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(mass, col, vis, [0, 0, -100], physicsClientId=cid)


def spawn_fixedwing(cid):
    """
    Creates 4 kinematic box bodies:
      0 - fuselage  (long, narrow, yellow)
      1 - wings     (short front-back, wide span, yellow)
      2 - htail     (small horizontal tail, dark yellow)
      3 - vtail     (vertical fin, dark yellow)
    Returns list of 4 body IDs.
    """
    yellow      = [1.0, 0.85, 0.0, 1.0]
    dark_yellow = [0.8, 0.6,  0.0, 1.0]

    ids = [
        _make_box([0.55, 0.055, 0.055], yellow,      cid),  # fuselage
        _make_box([0.07, 0.85,  0.018], yellow,      cid),  # main wings
        _make_box([0.06, 0.28,  0.012], dark_yellow, cid),  # horizontal tail
        _make_box([0.05, 0.012, 0.20 ], dark_yellow, cid),  # vertical fin
    ]
    print(f"[Fixed-wing] Spawned — fuselage ID: {ids[0]}, wings: {ids[1]}, "
          f"htail: {ids[2]}, vtail: {ids[3]}")
    return ids


def _move_part(body_id, pos, quat, cid):
    p.resetBasePositionAndOrientation(body_id, pos, quat, physicsClientId=cid)


def update_fixedwing(fw_ids, fw_pos, fw_wp_idx, dt, cid):
    """
    Move all 4 fixed-wing parts as one rigid unit.
    Parts are offset from the center in the LOCAL frame, then
    rotated by the current yaw before being placed in the world.
    """
    global fw_detected_ids

    # ── Waypoint navigation ──
    wp   = FW_WAYPOINTS[fw_wp_idx % len(FW_WAYPOINTS)]
    diff = wp - fw_pos
    dist = math.hypot(diff[0], diff[1])

    if dist < FW_WP_RADIUS:
        fw_wp_idx = (fw_wp_idx + 1) % len(FW_WAYPOINTS)
        wp   = FW_WAYPOINTS[fw_wp_idx]
        diff = wp - fw_pos
        print(f"[Fixed-wing] → Waypoint {fw_wp_idx}: {wp[:2]}")

    mag = math.hypot(diff[0], diff[1])
    direction = diff / (np.linalg.norm(diff) + 1e-8) if mag > 1e-4 else np.array([1.0, 0.0, 0.0])

    fw_pos     = fw_pos + direction * FW_SPEED * dt
    fw_pos[2]  = FW_ALTITUDE
    yaw        = math.atan2(direction[1], direction[0])
    cy, sy     = math.cos(yaw), math.sin(yaw)

    # Rotation matrix (yaw only) to rotate local offsets to world frame
    def rotate_xy(lx, ly):
        return cy*lx - sy*ly, sy*lx + cy*ly

    quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])

    # ── Local offsets (x=forward, y=left, z=up) ──
    # fuselage  : center of plane
    # wings     : same center (they span Y)
    # htail     : 0.55m behind center
    # vtail     : 0.55m behind, 0.15m up
    offsets = [
        (0.0,   0.0,  0.0  ),   # fuselage
        (0.0,   0.0,  0.0  ),   # wings (same center, different shape)
        (-0.52, 0.0,  0.0  ),   # horizontal tail (behind)
        (-0.52, 0.0,  0.16 ),   # vertical fin (behind + up)
    ]

    for body_id, (lx, ly, lz) in zip(fw_ids, offsets):
        wx, wy = rotate_xy(lx, ly)
        world_pos = [fw_pos[0]+wx, fw_pos[1]+wy, fw_pos[2]+lz]
        _move_part(body_id, world_pos, quat, cid)

    # ── Waste detection reporting ──
    for idx, (wx, wy, _) in enumerate(WASTE_POSITIONS):
        if idx not in fw_detected_ids and math.hypot(fw_pos[0]-wx, fw_pos[1]-wy) < FW_DETECTION_RADIUS:
            fw_detected_ids.add(idx)
            print(f"[Fixed-wing] DETECTED {WASTE_COLORS[idx].upper()} waste at ({wx},{wy}) "
                  f"→ Alerting quad team!")

    return fw_pos, fw_wp_idx


# ─────────────────────────────────────────────────────────────
# CAMERA (unchanged)
# ─────────────────────────────────────────────────────────────

def clamp_xy(xy):
    return np.clip(xy, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)


def update_follow_cam(target_pos_3d, keys):
    global cam_dist, cam_yaw, cam_pitch, cam_target_x, cam_target_y, cam_target_z, free_camera_mode

    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_camera_mode = not free_camera_mode
        if free_camera_mode:
            cam_target_x, cam_target_y, cam_target_z = target_pos_3d
            print("[Camera] FREE MODE")
        else:
            print("[Camera] FOLLOW MODE")

    if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN: cam_yaw -= CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: cam_yaw += CAM_YAW_STEP
    if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch + CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch - CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN: cam_dist = max(0.1, cam_dist - CAM_DIST_STEP)
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN: cam_dist += CAM_DIST_STEP

    if free_camera_mode:
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            cam_target_x += CAM_TARGET_STEP * np.cos(np.radians(cam_yaw))
            cam_target_y += CAM_TARGET_STEP * np.sin(np.radians(cam_yaw))
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            cam_target_x -= CAM_TARGET_STEP * np.cos(np.radians(cam_yaw))
            cam_target_y -= CAM_TARGET_STEP * np.sin(np.radians(cam_yaw))
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            cam_target_x += CAM_TARGET_STEP * np.cos(np.radians(cam_yaw + 90))
            cam_target_y += CAM_TARGET_STEP * np.sin(np.radians(cam_yaw + 90))
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            cam_target_x -= CAM_TARGET_STEP * np.cos(np.radians(cam_yaw + 90))
            cam_target_y -= CAM_TARGET_STEP * np.sin(np.radians(cam_yaw + 90))
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN: cam_target_z += CAM_TARGET_STEP * 0.5
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN: cam_target_z -= CAM_TARGET_STEP * 0.5

    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist, cam_yaw, cam_pitch = CAM_DIST_DEFAULT, CAM_YAW_DEFAULT, CAM_PITCH_DEFAULT
        free_camera_mode = False
        print("[Camera] Reset")

    camera_target = [cam_target_x, cam_target_y, cam_target_z] if free_camera_mode else target_pos_3d.tolist()
    p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, camera_target)


# ─────────────────────────────────────────────────────────────
# ONBOARD CAMERA & DETECTION (unchanged)
# ─────────────────────────────────────────────────────────────

def get_drone_camera_rgb(obs_drone, client_id):
    if not _HAS_CV2:
        return None
    pos  = np.array(obs_drone[0:3], dtype=float)
    quat = np.array(obs_drone[3:7], dtype=float)
    rm      = p.getMatrixFromQuaternion(quat)
    forward = np.array([rm[0], rm[3], rm[6]], dtype=float)
    up      = np.array([rm[2], rm[5], rm[8]], dtype=float)
    cam_pos    = pos + 0.05 * up
    cam_target_pt = cam_pos + forward
    view = p.computeViewMatrix(cam_pos.tolist(), cam_target_pt.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(CAM_FOV, CAM_WIDTH/CAM_HEIGHT, CAM_NEAR, CAM_FAR)
    img  = p.getCameraImage(CAM_WIDTH, CAM_HEIGHT, view, proj,
                            renderer=p.ER_TINY_RENDERER, physicsClientId=client_id)
    rgba = np.reshape(img[2], (CAM_HEIGHT, CAM_WIDTH, 4))
    return rgba[:, :, :3].astype(np.uint8)


def detect_orange_and_save(obs, frame_idx, last_saved_steps, configs, client_id):
    if not _HAS_CV2:
        return
    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)
    lower_orange = np.array([ 5,150,100]); upper_orange = np.array([25,255,255])
    lower_blue   = np.array([100,150, 50]); upper_blue   = np.array([130,255,255])
    lower_gray   = np.array([  0,  0, 50]); upper_gray   = np.array([180, 50,200])
    lower_green  = np.array([ 40,100, 50]); upper_green  = np.array([ 80,255,255])
    kernel = np.ones((3,3), np.uint8)

    for j, state in enumerate(obs):
        if frame_idx - last_saved_steps[j] < MIN_STEPS_BETWEEN_CAPTURES:
            continue
        rgb = get_drone_camera_rgb(state, client_id)
        if rgb is None:
            continue
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        def proc(lo, hi):
            m = cv2.inRange(hsv, lo, hi)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,   kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel, iterations=1)
            return cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        detected = det_orange = False
        for color_range, rect_color, is_orange in [
            ((lower_orange, upper_orange), (0,255,0), True),
            ((lower_blue,   upper_blue  ), (0,0,255), False),
            ((lower_gray,   upper_gray  ), (0,0,255), False),
            ((lower_green,  upper_green ), (0,0,255), False),
        ]:
            for cnt in proc(*color_range):
                if cv2.contourArea(cnt) >= 80:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(rgb, (x,y), (x+w,y+h), rect_color, 2)
                    detected = True
                    if is_orange: det_orange = True

        if not detected:
            continue
        role   = configs[j]["role"]
        prefix = "waste_target" if det_orange else "other_object"
        label  = "WASTE TARGET (orange)" if det_orange else "other object"
        fname  = os.path.join(CAMERA_OUTPUT_DIR, f"{prefix}_{role}_{frame_idx:05d}.png")
        cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        last_saved_steps[j] = frame_idx
        print(f"[Camera] [AUTO] {role} detected {label} → {fname}")


# ─────────────────────────────────────────────────────────────
# WORLD SETUP (unchanged)
# ─────────────────────────────────────────────────────────────

def spawn_waste_objects():
    global WASTE_BODY_IDS
    try:
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    except ImportError:
        pass
    ids = []
    for pos, color_name in zip(WASTE_POSITIONS, WASTE_COLORS):
        body_id = p.loadURDF("cube.urdf", basePosition=pos, globalScaling=0.5)
        p.changeVisualShape(body_id, -1, rgbaColor=COLOR_MAP.get(color_name, [1,1,1,1]))
        ids.append(body_id)
    WASTE_BODY_IDS = ids
    print("[World] Waste cubes spawned:")
    for pos, col in zip(WASTE_POSITIONS, WASTE_COLORS):
        print(f"   {pos} -> {col}")


def build_search_waypoints():
    return [np.array([x, y, TAKEOFF_Z_QUAD], dtype=float) for (x, y, _) in WASTE_POSITIONS]


# ─────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────

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
    init_xyzs = np.array([
        [ 0.0,  0.0, 0.10],
        [ 2.0, -1.5, 0.10],
        [-2.0, -1.5, 0.10],
    ], dtype=float)
    init_rpys = np.zeros((NUM_DRONES, 3), dtype=float)

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
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,          0, physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1, physicsClientId=env.CLIENT)

    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0.9,0.85,0.75,0.0], physicsClientId=env.CLIENT)

    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[25,25,0.01],
                                    rgbaColor=[0.87,0.72,0.53,1.0], physicsClientId=env.CLIENT)
    p.createMultiBody(0, -1, floor_vis, [0,0,-0.05], physicsClientId=env.CLIENT)

    for i, cfg in enumerate(DRONE_CONFIGS):
        try: p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=cfg["color"], physicsClientId=env.CLIENT)
        except: pass

    print("[World] Building desert terrain...")
    create_desert_terrain(
        heightmap_path="assets/terrain_desert_dunes.png",
        texture_path="assets/desert_sand.png",
        terrain_scale=(0.15, 0.15, 1.5)
    )
    spawn_desert_obstacles(num_rocks=12, num_vegetation=6, area_size=35.0,
                           exclude_positions=WASTE_POSITIONS, min_distance=4.0)
    spawn_waste_objects()

    # ── Fixed-wing ──
    fw_ids    = spawn_fixedwing(env.CLIENT)
    fw_pos    = np.copy(FW_WAYPOINTS[0])
    fw_wp_idx = 0

    # ── Quads ──
    waypoints      = build_search_waypoints()
    current_wp_idx = 0
    ctrls          = [DSLPIDControl(drone_model=cfg["model"]) for cfg in DRONE_CONFIGS]
    action         = np.zeros((NUM_DRONES, 4))
    logger         = Logger(logging_freq_hz=control_freq_hz, num_drones=NUM_DRONES,
                            output_folder=output_folder)
    dt              = 1.0 / control_freq_hz
    start           = time.time()
    quad_leader_pos = np.copy(init_xyzs[0])
    quad_leader_vel = np.zeros(3, dtype=float)
    target_pos      = np.copy(init_xyzs)
    takeoff_steps   = int(TAKEOFF_TIME * control_freq_hz)
    step            = 0
    last_saved_steps = [-999999] * NUM_DRONES

    print("\n[Mission] MIXED FLEET STARTED")
    print("  Fixed-wing  : Yellow plane | 5.5m altitude | 4.0 m/s | lawnmower survey")
    print("  Quadcopters : Red/Blue/Green | 2.5m altitude | V-formation inspection")
    print("  Waste sites : 8 locations")
    print("\n[Controls] Arrows: rotate | Z/X: zoom | F: free cam | R: reset\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # Fixed-wing step
            fw_pos, fw_wp_idx = update_fixedwing(fw_ids, fw_pos, fw_wp_idx, dt, env.CLIENT)

            # Quad leader takeoff / waypoint nav
            if step < takeoff_steps:
                alpha = (step + 1) / max(1, takeoff_steps)
                quad_leader_pos[2] = 0.10*(1-alpha) + TAKEOFF_Z_QUAD*alpha
                quad_leader_vel[:] = 0.0
            else:
                if current_wp_idx < len(waypoints):
                    wp      = waypoints[current_wp_idx]
                    diff    = wp - quad_leader_pos
                    dist_xy = math.hypot(diff[0], diff[1])
                    if dist_xy < 1.0 and abs(diff[2]) < 0.4:
                        print(f"[Quads] Leader reached waypoint {current_wp_idx}: {wp}")
                        current_wp_idx += 1
                        quad_leader_vel[:] = 0.0
                    else:
                        dir_xy = diff[:2]/dist_xy if dist_xy > 1e-6 else np.zeros(2)
                        quad_leader_vel[0] = dir_xy[0] * AUTO_SPEED_XY
                        quad_leader_vel[1] = dir_xy[1] * AUTO_SPEED_XY
                        quad_leader_vel[2] = np.clip(diff[2], -AUTO_SPEED_Z, AUTO_SPEED_Z)
                        quad_leader_pos[:2] = clamp_xy(quad_leader_pos[:2] + quad_leader_vel[:2]*dt)
                        quad_leader_pos[2]  = np.clip(quad_leader_pos[2]+quad_leader_vel[2]*dt, Z_MIN, Z_MAX)
                else:
                    quad_leader_vel[:] = 0.0

            # V-formation targets
            for i in range(NUM_DRONES):
                off = QUAD_FORMATION_OFFSETS[i]
                target_pos[i] = np.array([quad_leader_pos[0]+off[0],
                                          quad_leader_pos[1]+off[1],
                                          quad_leader_pos[2]], dtype=float)

            # Collision avoidance
            for i in range(NUM_DRONES):
                push = np.zeros(2)
                for j in range(NUM_DRONES):
                    if j == i: continue
                    dx = obs[i][0]-obs[j][0]; dy = obs[i][1]-obs[j][1]
                    d  = math.hypot(dx, dy)
                    if 1e-6 < d < MIN_SEP:
                        push += AVOID_GAIN*(MIN_SEP-d)/d * np.array([dx,dy])
                n = np.linalg.norm(push)
                if n > MAX_PUSH: push *= MAX_PUSH/n
                target_pos[i,0:2] = clamp_xy(target_pos[i,0:2]+push)
                target_pos[i,2]   = np.clip(target_pos[i,2], Z_MIN, Z_MAX)

            # PID control for quads
            for i, ctrl in enumerate(ctrls):
                raw = ctrl.computeControlFromState(
                    control_timestep=dt, state=obs[i],
                    target_pos=target_pos[i],
                    target_rpy=np.zeros(3), target_vel=np.zeros(3),
                    target_rpy_rates=np.zeros(3)
                )
                action[i,:] = raw[0]
                logger.log(drone=i, timestamp=step*dt, state=obs[i])

            obs, _, _, _, _ = env.step(action)

            # Auto detection (uncomment to enable)
            # detect_orange_and_save(obs, step, last_saved_steps, DRONE_CONFIGS, env.CLIENT)

            update_follow_cam(quad_leader_pos.copy(), keys)
            env.render()
            if gui:
                sync(step, start, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        logger.save()
        logger.save_as_csv("mixed_fleet_mission")
        if plot: logger.plot()
        print("\n[Mission] Complete. Logs saved.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixed fleet: fixed-wing scout + quad inspectors")
    parser.add_argument("--gui",  default=DEFAULT_GUI,  type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
