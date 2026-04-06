"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission

MISSION FLOW:
  1. FW_SCANNING  — Fixed-wing flies lawnmower grid survey. Quads hover at corner.
                    Auto-detects waste by proximity as it passes overhead.
  2. FW_RETURN    — Fixed-wing returns to corner.
  3. SQUAD_GO     — All drones fly to ORANGE waste sites only.
                    FW leads at 4m, quads follow in line at 2.5m.
  4. SQUAD_BACK   — All drones return to corner.
  5. DONE         — All hover at corner.

Controls:
  Camera: Arrow keys | Z/X zoom | F free-cam (WASD+QE) | Mouse | R reset
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
FW_SCANNING = 0
FW_RETURN   = 1
SQUAD_GO    = 2
SQUAD_BACK  = 3
DONE        = 4
STATE_NAMES = {0:"FW_SCANNING", 1:"FW_RETURNING", 2:"SQUAD_INSPECTION",
               3:"RETURNING_HOME", 4:"MISSION_COMPLETE"}

# ========= Positions =========
HOME_XY  = np.array([-15.0, -15.0])     # Bottom-left corner
FW_ALT   = 4.0                           # Fixed-wing altitude (above quads)
QUAD_ALT = 2.5                           # Quad cruise altitude

HOME_FW   = np.array([HOME_XY[0], HOME_XY[1], FW_ALT])
HOME_QUAD = np.array([
    [HOME_XY[0], HOME_XY[1] + 1.5, QUAD_ALT],  # Blue hover spot
    [HOME_XY[0], HOME_XY[1] + 3.0, QUAD_ALT],  # Green hover spot
])

# ========= Fixed-wing Survey (boustrophedon lawnmower grid) =========
# 6 parallel survey lines spaced 6m apart, covering y=-16 to y=14
_ys = [-16.0, -10.0, -4.0, 2.0, 8.0, 14.0]
_grid = []
for _i, _y in enumerate(_ys):
    if _i % 2 == 0:  # left → right
        _grid.extend([[-17.0, _y, FW_ALT], [17.0, _y, FW_ALT]])
    else:             # right → left
        _grid.extend([[17.0, _y, FW_ALT], [-17.0, _y, FW_ALT]])
FW_SCAN_WPS = np.array(_grid, dtype=float)   # 12 waypoints

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
WASTE_BODY_IDS = []

# Orange sites only — squad inspection targets
ORANGE_POS = [(x,y,z) for (x,y,z),c in zip(WASTE_POSITIONS,WASTE_COLORS) if c=="orange"]
ORANGE_WPS  = np.array([[x, y, FW_ALT] for (x,y,z) in ORANGE_POS], dtype=float)

FW_DETECT_RADIUS = 6.0   # metres — FW detects waste within this radius

# ========= Quadcopter Config =========
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_blue",  "color": [0.2, 0.2, 1.0, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_green", "color": [0.2, 0.8, 0.2, 1.0]},
]
NUM_DRONES = len(DRONE_CONFIGS)
LINE_DISTS = [5.0, 10.0]  # metres behind FW (blue=5, green=10)

# ========= Fixed-wing params =========
FW_SPEED    = 4.0    # m/s survey speed
SQUAD_SPEED = 2.5    # m/s slower when leading quads
FW_RADIUS   = 1.8    # waypoint reached radius

# ========= Simulation Defaults =========
DEFAULT_PHYSICS            = Physics("pyb")
DEFAULT_GUI                = True
DEFAULT_PLOT               = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_OUTPUT_FOLDER      = "results"
WORLD_LIMIT = 19.0
Z_MIN, Z_MAX = 0.8, 6.0
MIN_SEP      = 1.5
AVOID_GAIN   = 0.6
MAX_PUSH     = 0.8

# ========= Camera =========
CAM_DIST_DEFAULT  = 20.0
CAM_YAW_DEFAULT   = 45.0
CAM_PITCH_DEFAULT = -40.0
cam_dist  = CAM_DIST_DEFAULT
cam_yaw   = CAM_YAW_DEFAULT
cam_pitch = CAM_PITCH_DEFAULT
cam_target_x = cam_target_y = cam_target_z = 0.0
CAM_YAW_STEP=2.0; CAM_PITCH_STEP=1.5; CAM_DIST_STEP=0.5; CAM_TARGET_STEP=1.0
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0
free_camera_mode = False
CAM_WIDTH, CAM_HEIGHT = 200, 150
CAM_FOV, CAM_NEAR, CAM_FAR = 45.0, 0.1, 100.0
CAMERA_OUTPUT_DIR = "camera_frames"


# ─────────────────────────────────────────────────────────────
# FIXED-WING HELPERS
# ─────────────────────────────────────────────────────────────

def _make_box(half, color, cid):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    v = p.createVisualShape(   p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(0, c, v, [0,0,-200], physicsClientId=cid)


def spawn_fixedwing(cid):
    Y=[1.0,0.85,0.0,1.0]; D=[0.8,0.60,0.0,1.0]
    ids = [
        _make_box([0.55, 0.055, 0.055], Y, cid),  # fuselage
        _make_box([0.07, 0.85,  0.018], Y, cid),  # wings
        _make_box([0.06, 0.28,  0.012], D, cid),  # h-tail
        _make_box([0.05, 0.012, 0.20 ], D, cid),  # v-fin
    ]
    print(f"[Fixed-wing] Spawned — ID: {ids[0]}")
    return ids


def _place_fw(fw_ids, pos, direction, cid):
    """Update all 4 parts to pos with orientation from direction."""
    yaw    = math.atan2(direction[1], direction[0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    quat   = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    for bid, (lx,ly,lz) in zip(fw_ids, [(0,0,0),(0,0,0),(-0.52,0,0),(-0.52,0,0.16)]):
        wx = cy*lx - sy*ly;  wy = sy*lx + cy*ly
        p.resetBasePositionAndOrientation(
            bid, [pos[0]+wx, pos[1]+wy, pos[2]+lz], quat, physicsClientId=cid)


def move_fw(fw_ids, fw_pos, fw_dir, target, dt, cid, speed=None):
    """Fly fixed-wing toward target. Returns (new_pos, new_dir, reached)."""
    spd  = speed if speed else FW_SPEED
    diff = target - fw_pos
    dist = math.hypot(diff[0], diff[1])

    if dist < FW_RADIUS:
        _place_fw(fw_ids, fw_pos, fw_dir, cid)
        return fw_pos.copy(), fw_dir.copy(), True

    new_dir = diff / (np.linalg.norm(diff) + 1e-8)
    new_pos = fw_pos + new_dir * spd * dt
    new_pos[0] = np.clip(new_pos[0], -WORLD_LIMIT, WORLD_LIMIT)
    new_pos[1] = np.clip(new_pos[1], -WORLD_LIMIT, WORLD_LIMIT)
    new_pos[2] = target[2]
    _place_fw(fw_ids, new_pos, new_dir, cid)
    return new_pos, new_dir, False


# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────

def update_cam(target, keys):
    global cam_dist,cam_yaw,cam_pitch,cam_target_x,cam_target_y,cam_target_z,free_camera_mode
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_camera_mode = not free_camera_mode
        if free_camera_mode: cam_target_x,cam_target_y,cam_target_z=target[0],target[1],target[2]
        print("[Camera] " + ("FREE" if free_camera_mode else "FOLLOW"))
    if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN: cam_yaw  -= CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: cam_yaw  += CAM_YAW_STEP
    if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch+CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch-CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN: cam_dist = max(0.1,cam_dist-CAM_DIST_STEP)
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
        cam_dist,cam_yaw,cam_pitch=CAM_DIST_DEFAULT,CAM_YAW_DEFAULT,CAM_PITCH_DEFAULT
        free_camera_mode=False
    tgt = [cam_target_x,cam_target_y,cam_target_z] if free_camera_mode else list(target)
    p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, tgt)


# ─────────────────────────────────────────────────────────────
# ONBOARD CAMERA
# ─────────────────────────────────────────────────────────────

def get_drone_camera_rgb(obs_drone, client_id):
    if not _HAS_CV2: return None
    pos=np.array(obs_drone[0:3]); quat=np.array(obs_drone[3:7])
    rm=p.getMatrixFromQuaternion(quat)
    fwd=np.array([rm[0],rm[3],rm[6]]); up=np.array([rm[2],rm[5],rm[8]])
    cam_p=pos+0.05*up
    view=p.computeViewMatrix(cam_p.tolist(),(cam_p+fwd).tolist(),up.tolist())
    proj=p.computeProjectionMatrixFOV(CAM_FOV,CAM_WIDTH/CAM_HEIGHT,CAM_NEAR,CAM_FAR)
    img=p.getCameraImage(CAM_WIDTH,CAM_HEIGHT,view,proj,renderer=p.ER_TINY_RENDERER,physicsClientId=client_id)
    return np.reshape(img[2],(CAM_HEIGHT,CAM_WIDTH,4))[:,:,:3].astype(np.uint8)


def detect_and_save(obs, frame_idx, last_saved, configs, client_id):
    if not _HAS_CV2: return
    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)
    ranges=[(np.array([5,150,100]),np.array([25,255,255]),(0,255,0),True),
            (np.array([100,150,50]),np.array([130,255,255]),(0,0,255),False),
            (np.array([0,0,50]),np.array([180,50,200]),(0,0,255),False),
            (np.array([40,100,50]),np.array([80,255,255]),(0,0,255),False)]
    kernel=np.ones((3,3),np.uint8)
    for j,state in enumerate(obs):
        if frame_idx-last_saved[j]<40: continue
        rgb=get_drone_camera_rgb(state,client_id)
        if rgb is None: continue
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        det=det_o=False
        for lo,hi,col,is_o in ranges:
            m=cv2.inRange(hsv,lo,hi)
            m=cv2.morphologyEx(m,cv2.MORPH_OPEN,kernel,iterations=1)
            m=cv2.morphologyEx(m,cv2.MORPH_DILATE,kernel,iterations=1)
            for cnt in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(cnt)>=80:
                    x,y,w,h=cv2.boundingRect(cnt); cv2.rectangle(rgb,(x,y),(x+w,y+h),col,2)
                    det=True
                    if is_o: det_o=True
        if not det: continue
        role=configs[j]["role"]
        fname=os.path.join(CAMERA_OUTPUT_DIR,f"{'waste_target' if det_o else 'other'}_{role}_{frame_idx:05d}.png")
        cv2.imwrite(fname,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        last_saved[j]=frame_idx
        print(f"[Camera] {role} → {'ORANGE' if det_o else 'other'} → {fname}")


# ─────────────────────────────────────────────────────────────
# WORLD
# ─────────────────────────────────────────────────────────────

def spawn_waste_objects():
    global WASTE_BODY_IDS
    try:
        import pybullet_data; p.setAdditionalSearchPath(pybullet_data.getDataPath())
    except ImportError: pass
    ids=[]
    for pos,col in zip(WASTE_POSITIONS,WASTE_COLORS):
        bid=p.loadURDF("cube.urdf",basePosition=pos,globalScaling=0.5)
        p.changeVisualShape(bid,-1,rgbaColor=COLOR_MAP.get(col,[1,1,1,1]))
        ids.append(bid)
    WASTE_BODY_IDS=ids
    print("[World] Waste cubes spawned")


def banner(state, msg=""):
    n = STATE_NAMES[state]
    print(f"\n{'='*55}\n  [{n}] {msg}\n{'='*55}\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run(
    physics=DEFAULT_PHYSICS, gui=DEFAULT_GUI, plot=DEFAULT_PLOT,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    output_folder=DEFAULT_OUTPUT_FOLDER
):
    # ── Quads spawn already airborne at corner (avoids terrain collision) ──
    init_xyzs = np.array([
        [HOME_XY[0], HOME_XY[1]+1.5, QUAD_ALT],
        [HOME_XY[0], HOME_XY[1]+3.0, QUAD_ALT],
    ], dtype=float)
    init_rpys = np.zeros((NUM_DRONES, 3))

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=NUM_DRONES,
        initial_xyzs=init_xyzs, initial_rpys=init_rpys,
        physics=physics, pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz, gui=gui, user_debug_gui=False
    )
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,          0, physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1, physicsClientId=env.CLIENT)
    p.changeVisualShape(env.PLANE_ID,-1,rgbaColor=[0,0,0,0],physicsClientId=env.CLIENT)
    fv=p.createVisualShape(p.GEOM_BOX,halfExtents=[25,25,0.01],
                           rgbaColor=[0.87,0.72,0.53,1.0],physicsClientId=env.CLIENT)
    p.createMultiBody(0,-1,fv,[0,0,-0.05],physicsClientId=env.CLIENT)
    for i,cfg in enumerate(DRONE_CONFIGS):
        try: p.changeVisualShape(env.DRONE_IDS[i],-1,rgbaColor=cfg["color"],physicsClientId=env.CLIENT)
        except: pass

    print("[World] Building desert terrain...")
    create_desert_terrain("assets/terrain_desert_dunes.png","assets/desert_sand.png",(0.15,0.15,1.5))
    spawn_desert_obstacles(12,6,35.0,WASTE_POSITIONS,4.0)
    spawn_waste_objects()

    # ── Fixed-wing: spawns at home corner, already at altitude ──
    fw_ids = spawn_fixedwing(env.CLIENT)
    fw_pos = np.array([HOME_XY[0], HOME_XY[1], FW_ALT], dtype=float)
    fw_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)

    # ── Mission state ──
    mission_state  = FW_SCANNING
    scan_idx       = 0
    orange_idx     = 0
    detected_sites = set()
    banner(FW_SCANNING, f"Scanning {len(FW_SCAN_WPS)} survey lines | Quads hover at corner")

    # ── Quad control ──
    ctrls      = [DSLPIDControl(drone_model=cfg["model"]) for cfg in DRONE_CONFIGS]
    action     = np.zeros((NUM_DRONES,4))
    logger     = Logger(logging_freq_hz=control_freq_hz,num_drones=NUM_DRONES,output_folder=output_folder)
    target_pos = np.copy(init_xyzs)          # quads start at their hover spots
    dt         = 1.0 / control_freq_hz
    t0         = time.time()
    step       = 0
    last_saved = [-999999]*NUM_DRONES

    print("[Fleet]  Yellow fixed-wing = LEADER")
    print("[Fleet]  Blue  quad        = Follower 1 (5m behind)")
    print("[Fleet]  Green quad        = Follower 2 (10m behind)")
    print(f"[Survey] {len(FW_SCAN_WPS)} waypoints | {len(ORANGE_WPS)} orange targets")
    print("[Controls] Arrows: rotate | Z/X: zoom | F: free cam | R: reset\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # ── MISSION STATE MACHINE ──────────────────────

            if mission_state == FW_SCANNING:
                # FW flies lawnmower grid
                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, FW_SCAN_WPS[scan_idx], dt, env.CLIENT)

                # Auto-detect waste by proximity
                for idx,(wx,wy,_) in enumerate(WASTE_POSITIONS):
                    if idx not in detected_sites and math.hypot(fw_pos[0]-wx,fw_pos[1]-wy) < FW_DETECT_RADIUS:
                        detected_sites.add(idx)
                        col = WASTE_COLORS[idx]
                        marker = "*** TARGET ***" if col=="orange" else ""
                        print(f"[FW Scan] Detected {col.upper()} waste at ({wx},{wy}) {marker}")

                if reached:
                    scan_idx += 1
                    print(f"[FW Scan] Survey line {scan_idx}/{len(FW_SCAN_WPS)} complete")
                    if scan_idx >= len(FW_SCAN_WPS):
                        missing = [i for i in range(len(WASTE_POSITIONS)) if i not in detected_sites]
                        print(f"[FW Scan] Grid complete. Detected {len(detected_sites)}/8 sites.")
                        if missing:
                            print(f"[FW Scan] Missed site IDs: {missing} (too far from grid lines)")
                        mission_state = FW_RETURN
                        banner(FW_RETURN, "Fixed-wing returning to corner")

                # Quads hover at their corner spots
                for i in range(NUM_DRONES):
                    target_pos[i] = HOME_QUAD[i].copy()

            elif mission_state == FW_RETURN:
                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, HOME_FW, dt, env.CLIENT)

                if reached:
                    mission_state = SQUAD_GO
                    orange_idx    = 0
                    banner(SQUAD_GO,
                           f"Flying to {len(ORANGE_WPS)} ORANGE waste sites | Quads follow in line")

                for i in range(NUM_DRONES):
                    target_pos[i] = HOME_QUAD[i].copy()

            elif mission_state == SQUAD_GO:
                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, ORANGE_WPS[orange_idx], dt, env.CLIENT,
                    speed=SQUAD_SPEED)

                if reached:
                    ox,oy,_ = ORANGE_POS[orange_idx]
                    print(f"[Squad] Inspecting orange site {orange_idx+1}/{len(ORANGE_WPS)} at ({ox},{oy})")
                    orange_idx += 1
                    if orange_idx >= len(ORANGE_WPS):
                        mission_state = SQUAD_BACK
                        banner(SQUAD_BACK, "All drones returning to corner")

                # Quads follow FW in straight LINE
                behind = -fw_dir
                for i in range(NUM_DRONES):
                    d  = LINE_DISTS[i]
                    tx = np.clip(fw_pos[0]+behind[0]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    ty = np.clip(fw_pos[1]+behind[1]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    target_pos[i] = np.array([tx, ty, QUAD_ALT])

            elif mission_state == SQUAD_BACK:
                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, HOME_FW, dt, env.CLIENT, speed=SQUAD_SPEED)

                behind = -fw_dir
                for i in range(NUM_DRONES):
                    d  = LINE_DISTS[i]
                    tx = np.clip(fw_pos[0]+behind[0]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    ty = np.clip(fw_pos[1]+behind[1]*d, -WORLD_LIMIT, WORLD_LIMIT)
                    target_pos[i] = np.array([tx, ty, QUAD_ALT])

                if reached:
                    mission_state = DONE
                    banner(DONE, "All drones hovering at corner — mission complete")

            elif mission_state == DONE:
                # Everything hovers at corner forever
                _place_fw(fw_ids, HOME_FW, fw_dir, env.CLIENT)
                for i in range(NUM_DRONES):
                    target_pos[i] = HOME_QUAD[i].copy()

            # ── SAFETY: always re-place FW visual (prevents disappearing) ──
            _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)

            # ── COLLISION AVOIDANCE ──
            for i in range(NUM_DRONES):
                push = np.zeros(2)
                for j in range(NUM_DRONES):
                    if j==i: continue
                    dx=obs[i][0]-obs[j][0]; dy=obs[i][1]-obs[j][1]
                    d=math.hypot(dx,dy)
                    if 1e-6<d<MIN_SEP: push+=AVOID_GAIN*(MIN_SEP-d)/d*np.array([dx,dy])
                n=np.linalg.norm(push)
                if n>MAX_PUSH: push*=MAX_PUSH/n
                target_pos[i,0:2]=np.clip(target_pos[i,0:2]+push,-WORLD_LIMIT,WORLD_LIMIT)
                target_pos[i,2]  =np.clip(target_pos[i,2],Z_MIN,Z_MAX)

            # ── PID CONTROL ──
            for i,ctrl in enumerate(ctrls):
                raw=ctrl.computeControlFromState(
                    control_timestep=dt, state=obs[i],
                    target_pos=target_pos[i],
                    target_rpy=np.zeros(3), target_vel=np.zeros(3),
                    target_rpy_rates=np.zeros(3))
                action[i,:]=raw[0]
                logger.log(drone=i, timestamp=step*dt, state=obs[i])

            obs,_,_,_,_ = env.step(action)

            # Auto detection on quad cameras (uncomment to enable)
            # detect_and_save(obs, step, last_saved, DRONE_CONFIGS, env.CLIENT)

            # Camera follows FW
            update_cam(fw_pos.tolist(), keys)
            env.render()
            if gui: sync(step, t0, env.CTRL_TIMESTEP)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",  default=DEFAULT_GUI,  type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))
