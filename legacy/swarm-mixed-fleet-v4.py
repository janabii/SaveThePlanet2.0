"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission

- 1 Fixed-wing drone as LEADER (replaces red quad, same altitude, visits waste sites)
- 2 Quadcopters in a LINE behind the fixed-wing (blue & green)
- Desert environment with dunes and obstacles
- Fixed-wing leads the patrol, quads follow for close inspection

Controls:
  Movement:        AUTOPILOT (no keyboard needed).
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

# ========= Quadcopter Configuration (2 quads — red replaced by fixed-wing) =========
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_blue",  "color": [0.2, 0.2, 1.0, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_green", "color": [0.2, 0.8, 0.2, 1.0]},
]
NUM_DRONES = len(DRONE_CONFIGS)

# ========= Fixed-wing (Leader — replaces red quad) =========
FW_ALTITUDE     = 2.5    # Same altitude as quads
FW_SPEED        = 2.0    # m/s — match quad speed so followers keep up
FW_WP_RADIUS    = 1.0    # Same reach radius as original quad leader
FW_DETECTION_RADIUS = 6.0

# Fixed-wing visits the same waste sites as the old red quad leader
FW_WAYPOINTS = np.array([
    [-15.0, -12.0, FW_ALTITUDE],
    [-15.0,  12.0, FW_ALTITUDE],
    [  0.0, -15.0, FW_ALTITUDE],
    [  0.0,  15.0, FW_ALTITUDE],
    [ 15.0, -12.0, FW_ALTITUDE],
    [ 15.0,  12.0, FW_ALTITUDE],
    [ -8.0,   0.0, FW_ALTITUDE],
    [  8.0,   0.0, FW_ALTITUDE],
], dtype=float)

# Quads trail behind fixed-wing in a straight LINE
LINE_DISTANCES = [4.0, 8.0]   # Blue = 4m behind, Green = 8m behind

# ========= Simulation Defaults =========
DEFAULT_PHYSICS            = Physics("pyb")
DEFAULT_GUI                = True
DEFAULT_PLOT               = False
DEFAULT_USER_DEBUG_GUI     = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_DURATION_SEC       = 3600
DEFAULT_OUTPUT_FOLDER      = "results"

# ========= Flight Parameters =========
TAKEOFF_Z_QUAD = 2.5
TAKEOFF_TIME   = 5.0
WORLD_XY_LIMIT = 20.0
Z_MIN, Z_MAX   = 0.8, 6.0
MIN_SEP        = 1.5
AVOID_GAIN     = 0.6
MAX_PUSH       = 0.8

# ========= Camera =========
CAM_DIST_DEFAULT  = 15.0
CAM_YAW_DEFAULT   = 0.0
CAM_PITCH_DEFAULT = -45.0
cam_dist  = CAM_DIST_DEFAULT
cam_yaw   = CAM_YAW_DEFAULT
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
WASTE_COLORS = ["orange","blue","orange","gray","orange","green","orange","gray"]
COLOR_MAP = {
    "orange":[1.0,0.5,0.0,1.0], "blue":[0.1,0.4,1.0,1.0],
    "gray"  :[0.7,0.7,0.7,1.0], "green":[0.2,0.8,0.2,1.0],
}
WASTE_BODY_IDS  = []
fw_detected_ids = set()
MIN_STEPS_BETWEEN_CAPTURES = 40


# ─────────────────────────────────────────────────────────────
# FIXED-WING: 4-part model (fuselage + wings + htail + vtail)
# All parts move together every frame.
# ─────────────────────────────────────────────────────────────

def _make_box(half, color, cid):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    vis = p.createVisualShape(  p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(0, col, vis, [0, 0, -100], physicsClientId=cid)


def spawn_fixedwing(cid):
    yellow      = [1.0, 0.85, 0.0, 1.0]
    dark_yellow = [0.8, 0.60, 0.0, 1.0]
    ids = [
        _make_box([0.55, 0.055, 0.055], yellow,      cid),  # fuselage
        _make_box([0.07, 0.85,  0.018], yellow,      cid),  # main wings
        _make_box([0.06, 0.28,  0.012], dark_yellow, cid),  # horizontal tail
        _make_box([0.05, 0.012, 0.20 ], dark_yellow, cid),  # vertical fin
    ]
    print(f"[Fixed-wing] Spawned (Leader) — fuselage ID: {ids[0]}")
    return ids


def update_fixedwing(fw_ids, fw_pos, fw_wp_idx, dt, cid):
    """Move fixed-wing toward next waste waypoint. Returns (pos, wp_idx, direction)."""
    global fw_detected_ids

    wp   = FW_WAYPOINTS[fw_wp_idx % len(FW_WAYPOINTS)]
    diff = wp - fw_pos
    dist = math.hypot(diff[0], diff[1])

    if dist < FW_WP_RADIUS:
        fw_wp_idx = (fw_wp_idx + 1) % len(FW_WAYPOINTS)
        wp   = FW_WAYPOINTS[fw_wp_idx]
        diff = wp - fw_pos
        print(f"[Fixed-wing] → Waste site {fw_wp_idx}: {wp[:2]}")

    mag       = math.hypot(diff[0], diff[1])
    direction = diff / (np.linalg.norm(diff) + 1e-8) if mag > 1e-4 else np.array([1.0, 0.0, 0.0])

    fw_pos    = fw_pos + direction * FW_SPEED * dt
    fw_pos[0] = np.clip(fw_pos[0], -WORLD_XY_LIMIT, WORLD_XY_LIMIT)
    fw_pos[1] = np.clip(fw_pos[1], -WORLD_XY_LIMIT, WORLD_XY_LIMIT)
    fw_pos[2] = FW_ALTITUDE

    yaw  = math.atan2(direction[1], direction[0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])

    def rot(lx, ly):
        return cy*lx - sy*ly, sy*lx + cy*ly

    # Part offsets in local frame (forward=x, left=y, up=z)
    offsets = [(0.0,0.0,0.0), (0.0,0.0,0.0), (-0.52,0.0,0.0), (-0.52,0.0,0.16)]
    for bid, (lx, ly, lz) in zip(fw_ids, offsets):
        wx, wy = rot(lx, ly)
        p.resetBasePositionAndOrientation(
            bid, [fw_pos[0]+wx, fw_pos[1]+wy, fw_pos[2]+lz], quat, physicsClientId=cid
        )

    # Waste detection
    for idx, (wx, wy, _) in enumerate(WASTE_POSITIONS):
        if idx not in fw_detected_ids and math.hypot(fw_pos[0]-wx, fw_pos[1]-wy) < FW_DETECTION_RADIUS:
            fw_detected_ids.add(idx)
            print(f"[Fixed-wing] DETECTED {WASTE_COLORS[idx].upper()} waste at ({wx},{wy})"
                  f" → Quads closing in!")

    return fw_pos, fw_wp_idx, direction


# ─────────────────────────────────────────────────────────────
# CAMERA (unchanged from original)
# ─────────────────────────────────────────────────────────────

def clamp_xy(xy):
    return np.clip(xy, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)


def update_follow_cam(target_pos_3d, keys):
    global cam_dist, cam_yaw, cam_pitch, cam_target_x, cam_target_y, cam_target_z, free_camera_mode

    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_camera_mode = not free_camera_mode
        if free_camera_mode:
            cam_target_x, cam_target_y, cam_target_z = target_pos_3d
            print("[Camera] FREE MODE - WASD to move, Q/E for height")
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
            cam_target_x += CAM_TARGET_STEP * np.cos(np.radians(cam_yaw+90))
            cam_target_y += CAM_TARGET_STEP * np.sin(np.radians(cam_yaw+90))
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            cam_target_x -= CAM_TARGET_STEP * np.cos(np.radians(cam_yaw+90))
            cam_target_y -= CAM_TARGET_STEP * np.sin(np.radians(cam_yaw+90))
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN: cam_target_z += CAM_TARGET_STEP*0.5
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN: cam_target_z -= CAM_TARGET_STEP*0.5

    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist, cam_yaw, cam_pitch = CAM_DIST_DEFAULT, CAM_YAW_DEFAULT, CAM_PITCH_DEFAULT
        free_camera_mode = False
        print("[Camera] Reset")

    tgt = [cam_target_x, cam_target_y, cam_target_z] if free_camera_mode else target_pos_3d.tolist()
    p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, tgt)


# ─────────────────────────────────────────────────────────────
# ONBOARD CAMERA & DETECTION (unchanged)
# ─────────────────────────────────────────────────────────────

def get_drone_camera_rgb(obs_drone, client_id):
    if not _HAS_CV2: return None
    pos  = np.array(obs_drone[0:3], dtype=float)
    quat = np.array(obs_drone[3:7], dtype=float)
    rm      = p.getMatrixFromQuaternion(quat)
    forward = np.array([rm[0], rm[3], rm[6]], dtype=float)
    up      = np.array([rm[2], rm[5], rm[8]], dtype=float)
    cam_pos = pos + 0.05*up
    view = p.computeViewMatrix(cam_pos.tolist(), (cam_pos+forward).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(CAM_FOV, CAM_WIDTH/CAM_HEIGHT, CAM_NEAR, CAM_FAR)
    img  = p.getCameraImage(CAM_WIDTH, CAM_HEIGHT, view, proj,
                            renderer=p.ER_TINY_RENDERER, physicsClientId=client_id)
    return np.reshape(img[2], (CAM_HEIGHT, CAM_WIDTH, 4))[:,:,:3].astype(np.uint8)


def detect_orange_and_save(obs, frame_idx, last_saved_steps, configs, client_id):
    if not _HAS_CV2: return
    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)
    ranges = [
        (np.array([ 5,150,100]), np.array([25,255,255]), (0,255,0), True),
        (np.array([100,150, 50]), np.array([130,255,255]), (0,0,255), False),
        (np.array([  0,  0, 50]), np.array([180, 50,200]), (0,0,255), False),
        (np.array([ 40,100, 50]), np.array([ 80,255,255]), (0,0,255), False),
    ]
    kernel = np.ones((3,3), np.uint8)
    for j, state in enumerate(obs):
        if frame_idx - last_saved_steps[j] < MIN_STEPS_BETWEEN_CAPTURES: continue
        rgb = get_drone_camera_rgb(state, client_id)
        if rgb is None: continue
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        detected = det_orange = False
        for lo, hi, col, is_orange in ranges:
            m = cv2.inRange(hsv, lo, hi)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,   kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel, iterations=1)
            for cnt in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(cnt) >= 80:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(rgb, (x,y),(x+w,y+h), col, 2)
                    detected = True
                    if is_orange: det_orange = True
        if not detected: continue
        role   = configs[j]["role"]
        prefix = "waste_target" if det_orange else "other_object"
        fname  = os.path.join(CAMERA_OUTPUT_DIR, f"{prefix}_{role}_{frame_idx:05d}.png")
        cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        last_saved_steps[j] = frame_idx
        print(f"[Camera] {role} → {'ORANGE WASTE' if det_orange else 'other'} → {fname}")


# ─────────────────────────────────────────────────────────────
# WORLD SETUP
# ─────────────────────────────────────────────────────────────

def spawn_waste_objects():
    global WASTE_BODY_IDS
    try:
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    except ImportError: pass
    ids = []
    for pos, col in zip(WASTE_POSITIONS, WASTE_COLORS):
        bid = p.loadURDF("cube.urdf", basePosition=pos, globalScaling=0.5)
        p.changeVisualShape(bid, -1, rgbaColor=COLOR_MAP.get(col,[1,1,1,1]))
        ids.append(bid)
    WASTE_BODY_IDS = ids
    print("[World] Waste cubes spawned:")
    for pos, col in zip(WASTE_POSITIONS, WASTE_COLORS):
        print(f"   {pos} -> {col}")


# ─────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────

def run(
    physics=DEFAULT_PHYSICS, gui=DEFAULT_GUI, plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER
):
    # 2 quads start in a line behind where the fixed-wing will begin
    init_xyzs = np.array([
        [ 0.0, -4.0, 0.10],   # Blue quad  (4m behind fixed-wing start)
        [ 0.0, -8.0, 0.10],   # Green quad (8m behind fixed-wing start)
    ], dtype=float)
    init_rpys = np.zeros((NUM_DRONES, 3), dtype=float)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=NUM_DRONES,
        initial_xyzs=init_xyzs, initial_rpys=init_rpys,
        physics=physics, pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz, gui=gui, user_debug_gui=False
    )

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,          0, physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1, physicsClientId=env.CLIENT)

    # Hide default plane
    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0,0,0,0], physicsClientId=env.CLIENT)

    # Sand floor
    fv = p.createVisualShape(p.GEOM_BOX, halfExtents=[25,25,0.01],
                             rgbaColor=[0.87,0.72,0.53,1.0], physicsClientId=env.CLIENT)
    p.createMultiBody(0, -1, fv, [0,0,-0.05], physicsClientId=env.CLIENT)

    # Quad colors
    for i, cfg in enumerate(DRONE_CONFIGS):
        try: p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=cfg["color"], physicsClientId=env.CLIENT)
        except: pass

    # Desert terrain + obstacles
    print("[World] Building desert terrain...")
    create_desert_terrain("assets/terrain_desert_dunes.png","assets/desert_sand.png",(0.15,0.15,1.5))
    spawn_desert_obstacles(12, 6, 35.0, WASTE_POSITIONS, 4.0)
    spawn_waste_objects()

    # ── Fixed-wing (spawns above quad starting position) ──
    fw_ids    = spawn_fixedwing(env.CLIENT)
    fw_pos    = np.array([0.0, 0.0, FW_ALTITUDE], dtype=float)
    fw_wp_idx = 0
    fw_dir    = np.array([0.0, -1.0, 0.0], dtype=float)  # initially point backward (toward first WP)

    # ── Quad setup ──
    ctrls  = [DSLPIDControl(drone_model=cfg["model"]) for cfg in DRONE_CONFIGS]
    action = np.zeros((NUM_DRONES, 4))
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=NUM_DRONES, output_folder=output_folder)
    dt            = 1.0 / control_freq_hz
    start         = time.time()
    takeoff_steps = int(TAKEOFF_TIME * control_freq_hz)
    target_pos    = np.copy(init_xyzs)
    step          = 0
    last_saved_steps = [-999999] * NUM_DRONES

    print("\n[Mission] MIXED FLEET — Fixed-wing Leader + 2 Quad Followers")
    print(f"  Fixed-wing  : Yellow | {FW_ALTITUDE}m alt | {FW_SPEED} m/s | visits all waste sites")
    print(f"  Quad Blue   : 4m behind fixed-wing in line")
    print(f"  Quad Green  : 8m behind fixed-wing in line")
    print(f"  Waste sites : {len(WASTE_POSITIONS)}")
    print("\n[Controls] Arrows: rotate | Z/X: zoom | F: free cam | R: reset\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # ── Fixed-wing update (always moves) ──
            fw_pos, fw_wp_idx, fw_dir = update_fixedwing(fw_ids, fw_pos, fw_wp_idx, dt, env.CLIENT)

            # ── Quad targets ──
            behind = -fw_dir   # unit vector pointing opposite to flight direction

            if step < takeoff_steps:
                # Phase 1: rise vertically to TAKEOFF_Z_QUAD
                alpha = (step + 1) / max(1, takeoff_steps)
                for i in range(NUM_DRONES):
                    target_pos[i] = np.array([
                        init_xyzs[i][0],
                        init_xyzs[i][1],
                        0.10*(1-alpha) + TAKEOFF_Z_QUAD*alpha
                    ], dtype=float)
            else:
                # Phase 2: follow fixed-wing in a straight LINE
                for i in range(NUM_DRONES):
                    d  = LINE_DISTANCES[i]
                    tx = np.clip(fw_pos[0] + behind[0]*d, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)
                    ty = np.clip(fw_pos[1] + behind[1]*d, -WORLD_XY_LIMIT, WORLD_XY_LIMIT)
                    target_pos[i] = np.array([tx, ty, TAKEOFF_Z_QUAD], dtype=float)

            # ── Collision avoidance between quads ──
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

            # ── PID control ──
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

            # Camera follows fixed-wing position
            update_follow_cam(fw_pos.copy(), keys)
            env.render()
            if gui: sync(step, start, env.CTRL_TIMESTEP)
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
    parser = argparse.ArgumentParser(description="Mixed fleet: fixed-wing leader + 2 quad followers")
    parser.add_argument("--gui",  default=DEFAULT_GUI,  type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
