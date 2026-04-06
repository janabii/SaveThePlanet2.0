"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission

MISSION FLOW:
  1. TAKEOFF     — All drones rise at bottom-left corner
  2. FW_SCANNING — Fixed-wing scans all 8 waste sites. Quads hover at corner.
  3. FW_RETURN   — Fixed-wing returns to corner.
  4. SQUAD_GO    — All drones fly to ORANGE waste sites only. Quads follow fixed-wing.
  5. SQUAD_BACK  — All drones return to corner.
  6. DONE        — All hover at corner.

Controls:
  Camera: Arrow keys (yaw/pitch) | Z/X (zoom) | R (reset)
          F - free camera (WASD move, Q/E height) | Mouse drag
  Exit:   Ctrl + C
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

# ========= Mission States =========
TAKEOFF     = 0
FW_SCANNING = 1
FW_RETURN   = 2
SQUAD_GO    = 3
SQUAD_BACK  = 4
DONE        = 5
STATE_NAMES = {0:"TAKEOFF", 1:"FW_SCANNING", 2:"FW_RETURNING",
               3:"SQUAD_INSPECTION", 4:"RETURNING_HOME", 5:"DONE"}

# ========= Home Corner (bottom-left) =========
HOME_XY  = np.array([-17.0, -17.0])
HOME_Z   = 3.0    # fixed-wing hover altitude
QUAD_ALT = 2.5    # quad cruise altitude

# Quad hover spots at corner (side by side, slightly offset)
QUAD_HOVER = np.array([
    [-17.0, -15.5, QUAD_ALT],
    [-17.0, -14.0, QUAD_ALT],
])

# ========= Quadcopter Configuration =========
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_blue",  "color": [0.2, 0.2, 1.0, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_green", "color": [0.2, 0.8, 0.2, 1.0]},
]
NUM_DRONES = len(DRONE_CONFIGS)

# ========= Fixed-wing =========
FW_ALTITUDE = HOME_Z
FW_SPEED    = 3.0    # m/s
FW_RADIUS   = 1.5    # waypoint reached radius
LINE_DISTS  = [4.0, 8.0]   # metres behind FW for blue, green quad

# ========= Waste & Waypoints =========
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

# All 8 scan waypoints at FW altitude
SCAN_WPS = np.array([[x, y, FW_ALTITUDE] for (x,y,z) in WASTE_POSITIONS], dtype=float)

# Only orange waste waypoints for squad inspection
ORANGE_WPS = np.array(
    [[x, y, FW_ALTITUDE] for (x,y,z), c in zip(WASTE_POSITIONS, WASTE_COLORS) if c == "orange"],
    dtype=float
)
ORANGE_POSITIONS = [(x,y,z) for (x,y,z),c in zip(WASTE_POSITIONS, WASTE_COLORS) if c=="orange"]

WASTE_BODY_IDS = []
HOME_WP  = np.array([HOME_XY[0], HOME_XY[1], FW_ALTITUDE])

# ========= Simulation Defaults =========
DEFAULT_PHYSICS            = Physics("pyb")
DEFAULT_GUI                = True
DEFAULT_PLOT               = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_OUTPUT_FOLDER      = "results"
TAKEOFF_TIME  = 5.0
WORLD_LIMIT   = 19.0
Z_MIN, Z_MAX  = 0.8, 6.0
MIN_SEP       = 1.5
AVOID_GAIN    = 0.6
MAX_PUSH      = 0.8

# ========= Camera =========
CAM_DIST_DEFAULT  = 18.0
CAM_YAW_DEFAULT   = 45.0
CAM_PITCH_DEFAULT = -40.0
cam_dist  = CAM_DIST_DEFAULT
cam_yaw   = CAM_YAW_DEFAULT
cam_pitch = CAM_PITCH_DEFAULT
cam_target_x = cam_target_y = cam_target_z = 0.0
CAM_YAW_STEP = 2.0; CAM_PITCH_STEP = 1.5; CAM_DIST_STEP = 0.5; CAM_TARGET_STEP = 1.0
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0
free_camera_mode = False
CAMERA_OUTPUT_DIR = "camera_frames"
CAM_WIDTH, CAM_HEIGHT = 200, 150
CAM_FOV, CAM_NEAR, CAM_FAR = 45.0, 0.1, 100.0


# ─────────────────────────────────────────────────────────────
# FIXED-WING HELPERS
# ─────────────────────────────────────────────────────────────

def _make_box(half, color, cid):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    vis = p.createVisualShape(  p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(0, col, vis, [0, 0, -200], physicsClientId=cid)


def spawn_fixedwing(cid):
    yellow = [1.0, 0.85, 0.0, 1.0]; dark = [0.8, 0.60, 0.0, 1.0]
    ids = [
        _make_box([0.55, 0.055, 0.055], yellow, cid),  # fuselage
        _make_box([0.07, 0.85,  0.018], yellow, cid),  # wings
        _make_box([0.06, 0.28,  0.012], dark,   cid),  # h-tail
        _make_box([0.05, 0.012, 0.20 ], dark,   cid),  # v-fin
    ]
    print(f"[Fixed-wing] Spawned at corner — ID: {ids[0]}")
    return ids


def _place_fixedwing(fw_ids, pos, direction, cid):
    """Update all 4 parts visually."""
    yaw  = math.atan2(direction[1], direction[0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    offsets = [(0,0,0),(0,0,0),(-0.52,0,0),(-0.52,0,0.16)]
    for bid, (lx,ly,lz) in zip(fw_ids, offsets):
        wx = cy*lx - sy*ly; wy = sy*lx + cy*ly
        p.resetBasePositionAndOrientation(
            bid, [pos[0]+wx, pos[1]+wy, pos[2]+lz], quat, physicsClientId=cid)


def move_fixedwing(fw_ids, fw_pos, fw_dir, target, dt, cid):
    """Fly toward target. Returns (new_pos, new_dir, reached)."""
    diff = target - fw_pos
    dist = math.hypot(diff[0], diff[1])

    if dist < FW_RADIUS:
        _place_fixedwing(fw_ids, fw_pos, fw_dir, cid)
        return fw_pos.copy(), fw_dir.copy(), True

    new_dir = diff / (np.linalg.norm(diff) + 1e-8)
    new_pos = fw_pos + new_dir * FW_SPEED * dt
    new_pos[0] = np.clip(new_pos[0], -WORLD_LIMIT, WORLD_LIMIT)
    new_pos[1] = np.clip(new_pos[1], -WORLD_LIMIT, WORLD_LIMIT)
    new_pos[2] = target[2]
    _place_fixedwing(fw_ids, new_pos, new_dir, cid)
    return new_pos, new_dir, False


# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────

def update_follow_cam(target, keys):
    global cam_dist, cam_yaw, cam_pitch, cam_target_x, cam_target_y, cam_target_z, free_camera_mode

    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_camera_mode = not free_camera_mode
        if free_camera_mode: cam_target_x,cam_target_y,cam_target_z = target
        print("[Camera] " + ("FREE" if free_camera_mode else "FOLLOW"))

    if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN: cam_yaw -= CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: cam_yaw += CAM_YAW_STEP
    if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch+CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch-CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN: cam_dist = max(0.1, cam_dist-CAM_DIST_STEP)
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN: cam_dist += CAM_DIST_STEP
    if free_camera_mode:
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            cam_target_x+=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw))
            cam_target_y+=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw))
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            cam_target_x-=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw))
            cam_target_y-=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw))
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            cam_target_x+=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw+90))
            cam_target_y+=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw+90))
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            cam_target_x-=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw+90))
            cam_target_y-=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw+90))
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN: cam_target_z+=CAM_TARGET_STEP*0.5
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN: cam_target_z-=CAM_TARGET_STEP*0.5
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist,cam_yaw,cam_pitch = CAM_DIST_DEFAULT,CAM_YAW_DEFAULT,CAM_PITCH_DEFAULT
        free_camera_mode = False

    tgt = [cam_target_x,cam_target_y,cam_target_z] if free_camera_mode else list(target)
    p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, tgt)


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
    print("[World] Waste cubes spawned")


def _state_banner(state):
    msgs = {
        TAKEOFF:     "TAKEOFF — All drones rising at corner",
        FW_SCANNING: "FW SCANNING — Fixed-wing visiting all waste sites | Quads hover at corner",
        FW_RETURN:   "FW RETURNING — Fixed-wing coming back to corner",
        SQUAD_GO:    "SQUAD INSPECTION — All drones flying to ORANGE waste sites",
        SQUAD_BACK:  "RETURNING HOME — All drones flying back to corner",
        DONE:        "MISSION COMPLETE — All drones hovering at corner",
    }
    print(f"\n{'='*60}")
    print(f"  [{STATE_NAMES[state]}]  {msgs[state]}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────

def run(
    physics=DEFAULT_PHYSICS, gui=DEFAULT_GUI, plot=DEFAULT_PLOT,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    output_folder=DEFAULT_OUTPUT_FOLDER
):
    # Both quads spawn at corner, side by side
    init_xyzs = np.array([
        [HOME_XY[0], HOME_XY[1]+1.5, 0.10],  # Blue  (offset +Y so no overlap)
        [HOME_XY[0], HOME_XY[1]+3.0, 0.10],  # Green
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

    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0,0,0,0], physicsClientId=env.CLIENT)

    fv = p.createVisualShape(p.GEOM_BOX, halfExtents=[25,25,0.01],
                             rgbaColor=[0.87,0.72,0.53,1.0], physicsClientId=env.CLIENT)
    p.createMultiBody(0, -1, fv, [0,0,-0.05], physicsClientId=env.CLIENT)

    for i, cfg in enumerate(DRONE_CONFIGS):
        try: p.changeVisualShape(env.DRONE_IDS[i], -1, rgbaColor=cfg["color"], physicsClientId=env.CLIENT)
        except: pass

    print("[World] Building desert terrain...")
    create_desert_terrain("assets/terrain_desert_dunes.png","assets/desert_sand.png",(0.15,0.15,1.5))
    spawn_desert_obstacles(12, 6, 35.0, WASTE_POSITIONS, 4.0)
    spawn_waste_objects()

    # ── Fixed-wing ──
    fw_ids = spawn_fixedwing(env.CLIENT)
    fw_pos = np.array([HOME_XY[0], HOME_XY[1], 0.10], dtype=float)   # ground level
    fw_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    _place_fixedwing(fw_ids, fw_pos, fw_dir, env.CLIENT)

    # ── Mission state ──
    mission_state = TAKEOFF
    scan_idx      = 0
    orange_idx    = 0
    _state_banner(TAKEOFF)

    # ── Quad setup ──
    ctrls  = [DSLPIDControl(drone_model=cfg["model"]) for cfg in DRONE_CONFIGS]
    action = np.zeros((NUM_DRONES, 4))
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=NUM_DRONES, output_folder=output_folder)
    dt            = 1.0 / control_freq_hz
    start_time    = time.time()
    takeoff_steps = int(TAKEOFF_TIME * control_freq_hz)
    target_pos    = np.copy(init_xyzs)
    step          = 0

    print("[Fleet]  Yellow fixed-wing  = LEADER")
    print("[Fleet]  Blue  quad         = Follower 1")
    print("[Fleet]  Green quad         = Follower 2")
    print(f"[Fleet]  {len(SCAN_WPS)} waste sites to scan | {len(ORANGE_WPS)} orange targets for squad")
    print("\n[Controls] Arrows: rotate | Z/X: zoom | F: free cam | R: reset\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # ─────────────────────────────────────────────
            # STATE MACHINE
            # ─────────────────────────────────────────────

            if mission_state == TAKEOFF:
                alpha = min(1.0, (step + 1) / max(1, takeoff_steps))

                # Fixed-wing rises at corner
                fw_pos[2] = 0.10*(1-alpha) + FW_ALTITUDE*alpha
                _place_fixedwing(fw_ids, fw_pos, fw_dir, env.CLIENT)

                # Quads rise at their hover spots
                for i in range(NUM_DRONES):
                    target_pos[i] = np.array([
                        QUAD_HOVER[i][0], QUAD_HOVER[i][1],
                        0.10*(1-alpha) + QUAD_ALT*alpha
                    ])

                if alpha >= 1.0:
                    mission_state = FW_SCANNING
                    _state_banner(FW_SCANNING)

            elif mission_state == FW_SCANNING:
                # Fixed-wing flies to each scan waypoint
                fw_pos, fw_dir, reached = move_fixedwing(
                    fw_ids, fw_pos, fw_dir, SCAN_WPS[scan_idx], dt, env.CLIENT)

                if reached:
                    wx,wy,_ = WASTE_POSITIONS[scan_idx]
                    col     = WASTE_COLORS[scan_idx]
                    print(f"[Fixed-wing] Scanned site {scan_idx+1}/{len(SCAN_WPS)}: "
                          f"{col.upper()} waste at ({wx},{wy})")
                    scan_idx += 1
                    if scan_idx >= len(SCAN_WPS):
                        mission_state = FW_RETURN
                        _state_banner(FW_RETURN)

                # Quads hover at corner
                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

            elif mission_state == FW_RETURN:
                # Fixed-wing flies back to corner
                fw_pos, fw_dir, reached = move_fixedwing(
                    fw_ids, fw_pos, fw_dir, HOME_WP, dt, env.CLIENT)

                if reached:
                    mission_state = SQUAD_GO
                    orange_idx    = 0
                    _state_banner(SQUAD_GO)

                # Quads still hover at corner
                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

            elif mission_state == SQUAD_GO:
                # Fixed-wing leads to orange waste sites
                fw_pos, fw_dir, reached = move_fixedwing(
                    fw_ids, fw_pos, fw_dir, ORANGE_WPS[orange_idx], dt, env.CLIENT)

                if reached:
                    ox,oy,_ = ORANGE_POSITIONS[orange_idx]
                    print(f"[Squad] Inspecting ORANGE waste {orange_idx+1}/{len(ORANGE_WPS)} "
                          f"at ({ox},{oy})")
                    orange_idx += 1
                    if orange_idx >= len(ORANGE_WPS):
                        mission_state = SQUAD_BACK
                        _state_banner(SQUAD_BACK)

                # Quads follow fixed-wing in line
                behind = -fw_dir
                for i in range(NUM_DRONES):
                    d  = LINE_DISTS[i]
                    tx = np.clip(fw_pos[0] + behind[0]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    ty = np.clip(fw_pos[1] + behind[1]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    target_pos[i] = np.array([tx, ty, QUAD_ALT], dtype=float)

            elif mission_state == SQUAD_BACK:
                # Fixed-wing leads everyone back to corner
                fw_pos, fw_dir, reached = move_fixedwing(
                    fw_ids, fw_pos, fw_dir, HOME_WP, dt, env.CLIENT)

                # Quads follow in line toward home
                behind = -fw_dir
                for i in range(NUM_DRONES):
                    d  = LINE_DISTS[i]
                    tx = np.clip(fw_pos[0] + behind[0]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    ty = np.clip(fw_pos[1] + behind[1]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    target_pos[i] = np.array([tx, ty, QUAD_ALT], dtype=float)

                if reached:
                    mission_state = DONE
                    _state_banner(DONE)

            elif mission_state == DONE:
                # All hover at corner indefinitely
                _place_fixedwing(fw_ids, HOME_WP, fw_dir, env.CLIENT)
                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

            # ─────────────────────────────────────────────
            # COLLISION AVOIDANCE (between quads)
            # ─────────────────────────────────────────────
            for i in range(NUM_DRONES):
                push = np.zeros(2)
                for j in range(NUM_DRONES):
                    if j == i: continue
                    dx = obs[i][0]-obs[j][0]; dy = obs[i][1]-obs[j][1]
                    d  = math.hypot(dx,dy)
                    if 1e-6 < d < MIN_SEP:
                        push += AVOID_GAIN*(MIN_SEP-d)/d*np.array([dx,dy])
                n = np.linalg.norm(push)
                if n > MAX_PUSH: push *= MAX_PUSH/n
                target_pos[i,0:2] = np.clip(target_pos[i,0:2]+push,-WORLD_LIMIT,WORLD_LIMIT)
                target_pos[i,2]   = np.clip(target_pos[i,2], Z_MIN, Z_MAX)

            # ─────────────────────────────────────────────
            # PID CONTROL
            # ─────────────────────────────────────────────
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

            # Camera follows fixed-wing
            update_follow_cam(fw_pos.tolist(), keys)
            env.render()
            if gui: sync(step, start_time, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        logger.save()
        logger.save_as_csv("mixed_fleet_mission")
        if plot: logger.plot()
        print("\n[Mission] Logs saved.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixed fleet mission")
    parser.add_argument("--gui",  default=DEFAULT_GUI,  type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
