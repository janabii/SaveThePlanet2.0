"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission

MISSION FLOW:
  1. FW_SCANNING — Fixed-wing scans desert with lawnmower grid. Quads hover at corner.
  2. FW_RETURN   — Fixed-wing flies BACK to quads at corner.
  3. FW_LAND     — Fixed-wing descends at corner (safely offset from quads).
  4. SQUAD_GO    — BOTH quads fly to each ORANGE site. Camera follows Blue quad.
  5. SQUAD_BACK  — Both quads fly back to corner near landed fixed-wing.
  6. QUAD_LAND   — Both quads descend and land next to fixed-wing.
  7. DONE        — All on ground.

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
FW_LAND     = 2
SQUAD_GO    = 3
SQUAD_BACK  = 4
QUAD_LAND   = 5
DONE        = 6
STATE_NAMES = {0:"FW_SCANNING",1:"FW_RETURNING",2:"FW_LANDING",
               3:"SQUAD_INSPECTION",4:"SQUAD_RETURNING",5:"QUAD_LANDING",6:"DONE"}

# ========= Home Corner (bottom-left) =========
HOME_XY  = np.array([-15.0, -15.0])
FW_ALT   = 5.0
QUAD_ALT = 3.5

# Fixed-wing lands here (corner anchor point)
FW_LAND_XY  = HOME_XY.copy()
HOME_FW     = np.array([FW_LAND_XY[0], FW_LAND_XY[1], FW_ALT])

# Quad hover spots — 2 m away from FW land spot so no collision on descent
QUAD_HOVER = np.array([
    [HOME_XY[0],      HOME_XY[1]+2.5,  QUAD_ALT],   # Blue
    [HOME_XY[0]+2.5,  HOME_XY[1],      QUAD_ALT],   # Green
])

# ========= Fixed-wing survey (boustrophedon lawnmower) =========
_ys = [-16.0, -10.0, -4.0, 2.0, 8.0, 14.0]
_wps = []
for _i, _y in enumerate(_ys):
    if _i % 2 == 0: _wps += [[-17.0, _y, FW_ALT], [17.0, _y, FW_ALT]]
    else:           _wps += [[17.0, _y, FW_ALT], [-17.0, _y, FW_ALT]]
FW_SCAN_WPS = np.array(_wps, dtype=float)   # 12 waypoints

FW_SPEED      = 4.5
FW_WP_RADIUS  = 2.0
FW_DETECT_RAD = 6.0
FW_LAND_TIME  = 4.0    # seconds to descend
SQUAD_SPD     = 2.0    # m/s quad cruise speed
LAND_TIME     = 4.0    # seconds for quads to descend

# ========= Waste =========
WASTE_POSITIONS = [
    (-15.0,-12.0,0.0),(-15.0,12.0,0.0),
    (0.0,-15.0,0.0),(0.0,15.0,0.0),
    (15.0,-12.0,0.0),(15.0,12.0,0.0),
    (-8.0,0.0,0.0),(8.0,0.0,0.0),
]
WASTE_COLORS = ["orange","blue","orange","gray","orange","green","orange","gray"]
COLOR_MAP = {
    "orange":[1.0,0.5,0.0,1.0],"blue":[0.1,0.4,1.0,1.0],
    "gray":[0.7,0.7,0.7,1.0],"green":[0.2,0.8,0.2,1.0],
}
WASTE_BODY_IDS = []
ORANGE_POS = [(x,y,z) for (x,y,z),c in zip(WASTE_POSITIONS,WASTE_COLORS) if c=="orange"]

# Blue targets exact orange site; Green flies 3m north (+Y) of same site
BLUE_WPS  = np.array([[x, y, QUAD_ALT] for x,y,z in ORANGE_POS], dtype=float)
GREEN_WPS = np.array([[x, y+3.0, QUAD_ALT] for x,y,z in ORANGE_POS], dtype=float)

# ========= Quadcopter Config =========
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_blue",  "color": [0.2, 0.2, 1.0, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_green", "color": [0.2, 0.8, 0.2, 1.0]},
]
NUM_DRONES = len(DRONE_CONFIGS)

# ========= Simulation =========
DEFAULT_PHYSICS            = Physics("pyb")
DEFAULT_GUI                = True
DEFAULT_PLOT               = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_OUTPUT_FOLDER      = "results"
WORLD_LIMIT = 19.0
Z_MIN, Z_MAX = 1.0, 7.0
MIN_SEP = 2.2; AVOID_GAIN = 0.8; MAX_PUSH = 1.0

# ========= Camera =========
CAM_DIST_DEFAULT=20.0; CAM_YAW_DEFAULT=45.0; CAM_PITCH_DEFAULT=-40.0
cam_dist=CAM_DIST_DEFAULT; cam_yaw=CAM_YAW_DEFAULT; cam_pitch=CAM_PITCH_DEFAULT
cam_tx=cam_ty=cam_tz=0.0
CAM_YAW_STEP=2.0; CAM_PITCH_STEP=1.5; CAM_DIST_STEP=0.5; CAM_TARGET_STEP=1.0
CAM_PITCH_MIN,CAM_PITCH_MAX=-89.0,-10.0
free_cam=False
CAM_W,CAM_H=200,150; CAM_FOV,CAM_NEAR,CAM_FAR=45.0,0.1,100.0
CAMERA_OUTPUT_DIR="camera_frames"


# ─────────────────────────────────────────────────────────────
# FIXED-WING
# ─────────────────────────────────────────────────────────────

def _make_box(half,color,cid):
    c=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=cid)
    v=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color,physicsClientId=cid)
    return p.createMultiBody(0,c,v,[0,0,-300],physicsClientId=cid)

def spawn_fixedwing(cid):
    Y=[1.0,0.85,0.0,1.0]; D=[0.8,0.60,0.0,1.0]
    ids=[_make_box([0.55,0.055,0.055],Y,cid),
         _make_box([0.07,0.85, 0.018],Y,cid),
         _make_box([0.06,0.28, 0.012],D,cid),
         _make_box([0.05,0.012,0.20 ],D,cid)]
    print(f"[Fixed-wing] Spawned at corner (ID:{ids[0]})")
    return ids

def _place_fw(fw_ids, pos, direction, cid):
    yaw=math.atan2(direction[1],direction[0])
    cy,sy=math.cos(yaw),math.sin(yaw)
    q=p.getQuaternionFromEuler([0,0,yaw])
    for bid,(lx,ly,lz) in zip(fw_ids,[(0,0,0),(0,0,0),(-0.52,0,0),(-0.52,0,0.16)]):
        wx=cy*lx-sy*ly; wy=sy*lx+cy*ly
        p.resetBasePositionAndOrientation(bid,[pos[0]+wx,pos[1]+wy,pos[2]+lz],q,physicsClientId=cid)

def move_fw(fw_ids, fw_pos, fw_dir, target, dt, cid):
    """Move FW one step toward target. Returns (pos, dir, reached)."""
    diff=target-fw_pos; dist=math.hypot(diff[0],diff[1])
    if dist<FW_WP_RADIUS:
        _place_fw(fw_ids,fw_pos,fw_dir,cid)
        return fw_pos.copy(), fw_dir.copy(), True
    nd=diff/(np.linalg.norm(diff)+1e-8)
    np_=fw_pos+nd*FW_SPEED*dt
    np_[0]=np.clip(np_[0],-WORLD_LIMIT,WORLD_LIMIT)
    np_[1]=np.clip(np_[1],-WORLD_LIMIT,WORLD_LIMIT)
    np_[2]=target[2]
    _place_fw(fw_ids,np_,nd,cid)
    return np_, nd, False


# ─────────────────────────────────────────────────────────────
# QUAD SOFTWARE LEADER — smooth waypoint navigation
# ─────────────────────────────────────────────────────────────

def move_leader(ldr, wp, dt, speed=SQUAD_SPD):
    """Move software leader toward wp. Returns (new_ldr, reached)."""
    diff  = wp[:2] - ldr[:2]
    dist  = np.linalg.norm(diff)
    if dist < 1.0:
        new_ldr = ldr.copy(); new_ldr[2] = wp[2]
        return new_ldr, True
    dir2d  = diff / dist
    new_ldr = ldr.copy()
    new_ldr[:2] = ldr[:2] + dir2d * speed * dt
    new_ldr[0]  = np.clip(new_ldr[0], -WORLD_LIMIT, WORLD_LIMIT)
    new_ldr[1]  = np.clip(new_ldr[1], -WORLD_LIMIT, WORLD_LIMIT)
    new_ldr[2]  = wp[2]
    return new_ldr, False


# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────

def update_cam(target, keys):
    global cam_dist,cam_yaw,cam_pitch,cam_tx,cam_ty,cam_tz,free_cam
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_cam=not free_cam
        if free_cam: cam_tx,cam_ty,cam_tz=target[0],target[1],target[2]
        print("[Camera] FREE" if free_cam else "[Camera] FOLLOW")
    if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN: cam_yaw-=CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: cam_yaw+=CAM_YAW_STEP
    if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN:
        cam_pitch=np.clip(cam_pitch+CAM_PITCH_STEP,CAM_PITCH_MIN,CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN:
        cam_pitch=np.clip(cam_pitch-CAM_PITCH_STEP,CAM_PITCH_MIN,CAM_PITCH_MAX)
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN: cam_dist=max(0.1,cam_dist-CAM_DIST_STEP)
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN: cam_dist+=CAM_DIST_STEP
    if free_cam:
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            cam_tx+=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw)); cam_ty+=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw))
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            cam_tx-=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw)); cam_ty-=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw))
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            cam_tx+=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw+90)); cam_ty+=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw+90))
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            cam_tx-=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw+90)); cam_ty-=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw+90))
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN: cam_tz+=CAM_TARGET_STEP*0.5
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN: cam_tz-=CAM_TARGET_STEP*0.5
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist,cam_yaw,cam_pitch=CAM_DIST_DEFAULT,CAM_YAW_DEFAULT,CAM_PITCH_DEFAULT; free_cam=False
    tgt=[cam_tx,cam_ty,cam_tz] if free_cam else list(target)
    p.resetDebugVisualizerCamera(cam_dist,cam_yaw,cam_pitch,tgt)


# ─────────────────────────────────────────────────────────────
# ONBOARD CAMERA
# ─────────────────────────────────────────────────────────────

def get_rgb(obs_d, cid):
    if not _HAS_CV2: return None
    pos=np.array(obs_d[0:3]); quat=np.array(obs_d[3:7])
    rm=p.getMatrixFromQuaternion(quat)
    fwd=np.array([rm[0],rm[3],rm[6]]); up=np.array([rm[2],rm[5],rm[8]])
    cp=pos+0.05*up
    view=p.computeViewMatrix(cp.tolist(),(cp+fwd).tolist(),up.tolist())
    proj=p.computeProjectionMatrixFOV(CAM_FOV,CAM_W/CAM_H,CAM_NEAR,CAM_FAR)
    img=p.getCameraImage(CAM_W,CAM_H,view,proj,renderer=p.ER_TINY_RENDERER,physicsClientId=cid)
    return np.reshape(img[2],(CAM_H,CAM_W,4))[:,:,:3].astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# WORLD
# ─────────────────────────────────────────────────────────────

def spawn_waste():
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
    print(f"\n{'='*55}\n  [{STATE_NAMES[state]}]  {msg}\n{'='*55}\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run(
    physics=DEFAULT_PHYSICS, gui=DEFAULT_GUI, plot=DEFAULT_PLOT,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    output_folder=DEFAULT_OUTPUT_FOLDER
):
    # Quads spawn at QUAD_ALT (above all terrain) at their corner hover spots
    init_xyzs = np.array([
        [QUAD_HOVER[0][0], QUAD_HOVER[0][1], QUAD_ALT],
        [QUAD_HOVER[1][0], QUAD_HOVER[1][1], QUAD_ALT],
    ], dtype=float)
    init_rpys = np.zeros((NUM_DRONES,3))

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=NUM_DRONES,
        initial_xyzs=init_xyzs, initial_rpys=init_rpys,
        physics=physics, pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz, gui=gui, user_debug_gui=False
    )
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1,physicsClientId=env.CLIENT)
    p.changeVisualShape(env.PLANE_ID,-1,rgbaColor=[0,0,0,0],physicsClientId=env.CLIENT)
    fv=p.createVisualShape(p.GEOM_BOX,halfExtents=[25,25,0.01],
                           rgbaColor=[0.87,0.72,0.53,1.0],physicsClientId=env.CLIENT)
    p.createMultiBody(0,-1,fv,[0,0,-0.05],physicsClientId=env.CLIENT)
    for i,cfg in enumerate(DRONE_CONFIGS):
        try: p.changeVisualShape(env.DRONE_IDS[i],-1,rgbaColor=cfg["color"],physicsClientId=env.CLIENT)
        except: pass

    print("[World] Building desert terrain...")
    create_desert_terrain("assets/terrain_desert_dunes.png","assets/desert_sand.png",(0.15,0.15,1.5))
    exclusion = list(WASTE_POSITIONS) + [
        (HOME_XY[0],     HOME_XY[1],     0.0),
        (QUAD_HOVER[0][0], QUAD_HOVER[0][1], 0.0),
        (QUAD_HOVER[1][0], QUAD_HOVER[1][1], 0.0),
    ]
    spawn_desert_obstacles(12,6,35.0,exclusion,5.0)
    spawn_waste()

    # ── Fixed-wing: starts at corner at altitude ──
    fw_ids = spawn_fixedwing(env.CLIENT)
    fw_pos = HOME_FW.copy()
    fw_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)

    # ── Mission variables ──
    mission_state  = FW_SCANNING
    scan_idx       = 0
    detected_sites = set()
    fw_land_step   = 0
    fw_land_max    = int(FW_LAND_TIME * control_freq_hz)
    orange_idx     = 0
    quad_land_step = 0
    quad_land_max  = int(LAND_TIME * control_freq_hz)

    # Software leaders for smooth quad navigation (start at hover spots)
    blue_ldr  = QUAD_HOVER[0].copy().astype(float)
    green_ldr = QUAD_HOVER[1].copy().astype(float)

    banner(FW_SCANNING, f"Grid scanning | {len(FW_SCAN_WPS)} lines | Quads hover at corner")

    # ── Quad control setup ──
    ctrls      = [DSLPIDControl(drone_model=cfg["model"]) for cfg in DRONE_CONFIGS]
    action     = np.zeros((NUM_DRONES,4))
    logger     = Logger(logging_freq_hz=control_freq_hz,num_drones=NUM_DRONES,output_folder=output_folder)
    target_pos = np.copy(init_xyzs)
    dt         = 1.0/control_freq_hz
    t0         = time.time()
    step       = 0

    print("[Fleet]  Yellow fixed-wing = SCOUT (phase 1 + 2)")
    print("[Fleet]  Blue  quad        = INSPECTOR leader (phase 3+)")
    print("[Fleet]  Green quad        = INSPECTOR follower (phase 3+)")
    print(f"[Survey] {len(FW_SCAN_WPS)} grid lines | {len(BLUE_WPS)} orange targets")
    print("[Controls] Arrows:rotate | Z/X:zoom | F:free-cam | R:reset\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # ════════════════════════════════════════════════
            # STATE MACHINE
            # ════════════════════════════════════════════════

            if mission_state == FW_SCANNING:
                # ── FW flies lawnmower grid ──
                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, FW_SCAN_WPS[scan_idx], dt, env.CLIENT)

                for idx,(wx,wy,_) in enumerate(WASTE_POSITIONS):
                    if idx not in detected_sites and math.hypot(fw_pos[0]-wx,fw_pos[1]-wy)<FW_DETECT_RAD:
                        detected_sites.add(idx)
                        col = WASTE_COLORS[idx]
                        tag = "  *** ORANGE TARGET ***" if col=="orange" else ""
                        print(f"  [FW Scan] {col.upper()} waste at ({wx:.0f},{wy:.0f}){tag}")

                if reached:
                    scan_idx += 1
                    print(f"  [FW] Waypoint {scan_idx}/{len(FW_SCAN_WPS)}")
                    if scan_idx >= len(FW_SCAN_WPS):
                        print(f"  [FW] Scan complete — {len(detected_sites)}/8 sites found")
                        mission_state = FW_RETURN
                        banner(FW_RETURN, "Fixed-wing returning to corner")

                # Quads hover at corner
                target_pos[0] = QUAD_HOVER[0].copy()
                target_pos[1] = QUAD_HOVER[1].copy()

            elif mission_state == FW_RETURN:
                # ── FW flies back to corner ──
                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, HOME_FW, dt, env.CLIENT)

                if reached:
                    mission_state = FW_LAND
                    fw_land_step  = 0
                    banner(FW_LAND, "Fixed-wing descending at corner")

                target_pos[0] = QUAD_HOVER[0].copy()
                target_pos[1] = QUAD_HOVER[1].copy()

            elif mission_state == FW_LAND:
                # ── FW descends smoothly from FW_ALT → 0.2m ──
                alpha     = min(1.0, fw_land_step / max(1, fw_land_max))
                fw_pos[2] = FW_ALT*(1.0-alpha) + 0.2*alpha
                _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)
                fw_land_step += 1

                if fw_land_step >= fw_land_max:
                    print("  [Fixed-wing] Landed. Quads launching for inspection.")
                    mission_state = SQUAD_GO
                    orange_idx    = 0
                    banner(SQUAD_GO,
                           f"Camera → Blue quad | {len(BLUE_WPS)} orange sites to inspect")

                target_pos[0] = QUAD_HOVER[0].copy()
                target_pos[1] = QUAD_HOVER[1].copy()

            elif mission_state == SQUAD_GO:
                # ── Both quads fly independently to their orange site targets ──
                if orange_idx < len(BLUE_WPS):
                    # Move blue leader toward its orange wp
                    blue_ldr, blue_reached = move_leader(blue_ldr, BLUE_WPS[orange_idx], dt)

                    # Move green leader toward its offset orange wp (3m north of blue target)
                    green_ldr, _ = move_leader(green_ldr, GREEN_WPS[orange_idx], dt)

                    if blue_reached:
                        ox,oy,_ = ORANGE_POS[orange_idx]
                        print(f"  [Squad] Inspected orange site {orange_idx+1}/{len(BLUE_WPS)}"
                              f" at ({ox:.0f},{oy:.0f})")
                        orange_idx += 1

                else:
                    mission_state = SQUAD_BACK
                    banner(SQUAD_BACK, "Inspection done — returning to corner")

                target_pos[0] = blue_ldr.copy()
                target_pos[1] = green_ldr.copy()

            elif mission_state == SQUAD_BACK:
                # ── Both quads fly back to their corner hover spots ──
                blue_ldr,  blue_home  = move_leader(blue_ldr,  QUAD_HOVER[0], dt)
                green_ldr, green_home = move_leader(green_ldr, QUAD_HOVER[1], dt)

                target_pos[0] = blue_ldr.copy()
                target_pos[1] = green_ldr.copy()

                if blue_home and green_home:
                    mission_state  = QUAD_LAND
                    quad_land_step = 0
                    banner(QUAD_LAND, "Both quads descending next to fixed-wing")

            elif mission_state == QUAD_LAND:
                # ── Both quads descend smoothly from QUAD_ALT → 0.2m ──
                alpha = min(1.0, quad_land_step / max(1, quad_land_max))
                z_now = QUAD_ALT*(1.0-alpha) + 0.2*alpha
                for i in range(NUM_DRONES):
                    target_pos[i] = np.array([QUAD_HOVER[i][0], QUAD_HOVER[i][1], z_now])
                quad_land_step += 1

                if quad_land_step >= quad_land_max:
                    mission_state = DONE
                    banner(DONE, "All drones landed — mission complete!")

            elif mission_state == DONE:
                # Everything stays on ground
                for i in range(NUM_DRONES):
                    target_pos[i] = np.array([QUAD_HOVER[i][0], QUAD_HOVER[i][1], 0.2])
                _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)

            # ── Safety: always refresh FW visual ──
            _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)

            # ── Collision avoidance (between quads only) ──
            for i in range(NUM_DRONES):
                push=np.zeros(2)
                for j in range(NUM_DRONES):
                    if j==i: continue
                    dx=obs[i][0]-obs[j][0]; dy=obs[i][1]-obs[j][1]; d=math.hypot(dx,dy)
                    if 1e-6<d<MIN_SEP: push+=AVOID_GAIN*(MIN_SEP-d)/d*np.array([dx,dy])
                n=np.linalg.norm(push)
                if n>MAX_PUSH: push*=MAX_PUSH/n
                target_pos[i,0:2]=np.clip(target_pos[i,0:2]+push,-WORLD_LIMIT,WORLD_LIMIT)
                target_pos[i,2]  =np.clip(target_pos[i,2],Z_MIN,Z_MAX)

            # ── PID control ──
            for i,ctrl in enumerate(ctrls):
                raw=ctrl.computeControlFromState(
                    control_timestep=dt, state=obs[i],
                    target_pos=target_pos[i],
                    target_rpy=np.zeros(3), target_vel=np.zeros(3),
                    target_rpy_rates=np.zeros(3))
                action[i,:]=raw[0]
                logger.log(drone=i,timestamp=step*dt,state=obs[i])

            obs,_,_,_,_ = env.step(action)

            # ── Camera: follow FW during scan/return/land → then Blue quad ──
            if mission_state in (FW_SCANNING, FW_RETURN, FW_LAND):
                cam_target = fw_pos.tolist()
            else:
                cam_target = list(obs[0][0:3])    # Blue quad

            update_cam(cam_target, keys)
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
