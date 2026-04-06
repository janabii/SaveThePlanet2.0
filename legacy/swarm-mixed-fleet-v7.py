"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission

MISSION FLOW:
  1. FW_SCANNING — Fixed-wing scans desert with lawnmower grid. Quads hover at corner.
  2. FW_LANDING  — Fixed-wing descends and lands. Quads still hover.
  3. SQUAD_GO    — Camera switches to Blue quad. Quads fly to ORANGE sites only.
  4. SQUAD_BACK  — Quads return to corner.
  5. DONE        — Quads hover at corner.

Controls:
  Camera: Arrow keys | Z/X zoom | F free-cam | Mouse | R reset
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
FW_LANDING  = 1
SQUAD_GO    = 2
SQUAD_BACK  = 3
DONE        = 4
STATE_NAMES = {0:"FW_SCANNING",1:"FW_LANDING",2:"SQUAD_INSPECTION",
               3:"RETURNING_HOME",4:"MISSION_COMPLETE"}

# ========= Positions =========
HOME_XY  = np.array([-15.0, -15.0])
FW_ALT   = 5.0    # fixed-wing flies higher
QUAD_ALT = 3.5    # quads high enough to clear all terrain (max 1.5m)
QUAD_SPD = 2.0    # m/s quad movement speed

HOME_FW  = np.array([HOME_XY[0], HOME_XY[1], FW_ALT])

# Quad hover spots at corner — spaced 2m apart so no collision
QUAD_HOVER = np.array([
    [HOME_XY[0],       HOME_XY[1]+2.0, QUAD_ALT],
    [HOME_XY[0]+2.0,   HOME_XY[1],     QUAD_ALT],
])

# ========= Fixed-wing survey (boustrophedon lawnmower) =========
_ys  = [-16.0, -10.0, -4.0, 2.0, 8.0, 14.0]
_wps = []
for _i, _y in enumerate(_ys):
    if _i % 2 == 0: _wps += [[-17.0,_y,FW_ALT],[17.0,_y,FW_ALT]]
    else:           _wps += [[17.0,_y,FW_ALT],[-17.0,_y,FW_ALT]]
FW_SCAN_WPS = np.array(_wps, dtype=float)   # 12 waypoints

FW_SPEED       = 4.5
FW_WP_RADIUS   = 2.0
FW_DETECT_RAD  = 6.0
FW_LAND_TIME   = 4.0    # seconds to descend

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
ORANGE_POS  = [(x,y,z) for (x,y,z),c in zip(WASTE_POSITIONS,WASTE_COLORS) if c=="orange"]
ORANGE_WPS  = np.array([[x,y,QUAD_ALT] for x,y,z in ORANGE_POS], dtype=float)

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
MIN_SEP = 2.5; AVOID_GAIN = 0.8; MAX_PUSH = 1.0

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
# FIXED-WING VISUALS
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
    print(f"[Fixed-wing] Spawned — ID:{ids[0]}")
    return ids

def _place_fw(fw_ids,pos,direction,cid):
    yaw=math.atan2(direction[1],direction[0])
    cy,sy=math.cos(yaw),math.sin(yaw)
    q=p.getQuaternionFromEuler([0,0,yaw])
    for bid,(lx,ly,lz) in zip(fw_ids,[(0,0,0),(0,0,0),(-0.52,0,0),(-0.52,0,0.16)]):
        wx=cy*lx-sy*ly; wy=sy*lx+cy*ly
        p.resetBasePositionAndOrientation(bid,[pos[0]+wx,pos[1]+wy,pos[2]+lz],q,physicsClientId=cid)

def move_fw(fw_ids,fw_pos,fw_dir,target,dt,cid):
    diff=target-fw_pos; dist=math.hypot(diff[0],diff[1])
    if dist<FW_WP_RADIUS:
        _place_fw(fw_ids,fw_pos,fw_dir,cid)
        return fw_pos.copy(),fw_dir.copy(),True
    nd=diff/(np.linalg.norm(diff)+1e-8)
    np_=fw_pos+nd*FW_SPEED*dt
    np_[0]=np.clip(np_[0],-WORLD_LIMIT,WORLD_LIMIT)
    np_[1]=np.clip(np_[1],-WORLD_LIMIT,WORLD_LIMIT)
    np_[2]=target[2]
    _place_fw(fw_ids,np_,nd,cid)
    return np_,nd,False


# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────

def update_cam(target,keys):
    global cam_dist,cam_yaw,cam_pitch,cam_tx,cam_ty,cam_tz,free_cam
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_cam=not free_cam
        if free_cam: cam_tx,cam_ty,cam_tz=target[0],target[1],target[2]
        print("[Camera] FREE" if free_cam else "[Camera] FOLLOW")
    if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  &p.KEY_IS_DOWN: cam_yaw-=CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] &p.KEY_IS_DOWN: cam_yaw+=CAM_YAW_STEP
    if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    &p.KEY_IS_DOWN:
        cam_pitch=np.clip(cam_pitch+CAM_PITCH_STEP,CAM_PITCH_MIN,CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  &p.KEY_IS_DOWN:
        cam_pitch=np.clip(cam_pitch-CAM_PITCH_STEP,CAM_PITCH_MIN,CAM_PITCH_MAX)
    if ord('z') in keys and keys[ord('z')] &p.KEY_IS_DOWN: cam_dist=max(0.1,cam_dist-CAM_DIST_STEP)
    if ord('x') in keys and keys[ord('x')] &p.KEY_IS_DOWN: cam_dist+=CAM_DIST_STEP
    if free_cam:
        if ord('w') in keys and keys[ord('w')] &p.KEY_IS_DOWN:
            cam_tx+=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw)); cam_ty+=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw))
        if ord('s') in keys and keys[ord('s')] &p.KEY_IS_DOWN:
            cam_tx-=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw)); cam_ty-=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw))
        if ord('a') in keys and keys[ord('a')] &p.KEY_IS_DOWN:
            cam_tx+=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw+90)); cam_ty+=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw+90))
        if ord('d') in keys and keys[ord('d')] &p.KEY_IS_DOWN:
            cam_tx-=CAM_TARGET_STEP*np.cos(np.radians(cam_yaw+90)); cam_ty-=CAM_TARGET_STEP*np.sin(np.radians(cam_yaw+90))
        if ord('q') in keys and keys[ord('q')] &p.KEY_IS_DOWN: cam_tz+=CAM_TARGET_STEP*0.5
        if ord('e') in keys and keys[ord('e')] &p.KEY_IS_DOWN: cam_tz-=CAM_TARGET_STEP*0.5
    if ord('r') in keys and keys[ord('r')] &p.KEY_WAS_TRIGGERED:
        cam_dist,cam_yaw,cam_pitch=CAM_DIST_DEFAULT,CAM_YAW_DEFAULT,CAM_PITCH_DEFAULT; free_cam=False
    tgt=[cam_tx,cam_ty,cam_tz] if free_cam else list(target)
    p.resetDebugVisualizerCamera(cam_dist,cam_yaw,cam_pitch,tgt)


# ─────────────────────────────────────────────────────────────
# ONBOARD CAMERA & DETECTION
# ─────────────────────────────────────────────────────────────

def get_rgb(obs_d,cid):
    if not _HAS_CV2: return None
    pos=np.array(obs_d[0:3]); quat=np.array(obs_d[3:7])
    rm=p.getMatrixFromQuaternion(quat)
    fwd=np.array([rm[0],rm[3],rm[6]]); up=np.array([rm[2],rm[5],rm[8]])
    cp=pos+0.05*up
    view=p.computeViewMatrix(cp.tolist(),(cp+fwd).tolist(),up.tolist())
    proj=p.computeProjectionMatrixFOV(CAM_FOV,CAM_W/CAM_H,CAM_NEAR,CAM_FAR)
    img=p.getCameraImage(CAM_W,CAM_H,view,proj,renderer=p.ER_TINY_RENDERER,physicsClientId=cid)
    return np.reshape(img[2],(CAM_H,CAM_W,4))[:,:,:3].astype(np.uint8)

def detect_save(obs,step,last,cfgs,cid):
    if not _HAS_CV2: return
    os.makedirs(CAMERA_OUTPUT_DIR,exist_ok=True)
    rngs=[(np.array([5,150,100]),np.array([25,255,255]),(0,255,0),True),
          (np.array([100,150,50]),np.array([130,255,255]),(0,0,255),False),
          (np.array([0,0,50]),np.array([180,50,200]),(0,0,255),False),
          (np.array([40,100,50]),np.array([80,255,255]),(0,0,255),False)]
    k=np.ones((3,3),np.uint8)
    for j,st in enumerate(obs):
        if step-last[j]<40: continue
        rgb=get_rgb(st,cid)
        if rgb is None: continue
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        det=det_o=False
        for lo,hi,col,io in rngs:
            m=cv2.inRange(hsv,lo,hi)
            m=cv2.morphologyEx(m,cv2.MORPH_OPEN,k,iterations=1)
            m=cv2.morphologyEx(m,cv2.MORPH_DILATE,k,iterations=1)
            for cnt in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(cnt)>=80:
                    x,y,w,h=cv2.boundingRect(cnt); cv2.rectangle(rgb,(x,y),(x+w,y+h),col,2)
                    det=True
                    if io: det_o=True
        if not det: continue
        fn=os.path.join(CAMERA_OUTPUT_DIR,f"{'waste_target' if det_o else 'other'}_{cfgs[j]['role']}_{step:05d}.png")
        cv2.imwrite(fn,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        last[j]=step
        print(f"[Camera] {cfgs[j]['role']} → {'ORANGE' if det_o else 'other'} → {fn}")


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

def banner(state,msg=""):
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
    # Quad spawn positions: well above terrain (z=QUAD_ALT=3.5m) at corner
    init_xyzs = np.array([
        [HOME_XY[0],      HOME_XY[1]+2.0, QUAD_ALT],   # Blue
        [HOME_XY[0]+2.0,  HOME_XY[1],     QUAD_ALT],   # Green
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
    # Exclude drone spawn area AND waste positions from obstacles
    exclusion = list(WASTE_POSITIONS) + [
        (HOME_XY[0],     HOME_XY[1],     0.0),
        (HOME_XY[0],     HOME_XY[1]+2.0, 0.0),
        (HOME_XY[0]+2.0, HOME_XY[1],     0.0),
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
    land_step      = 0
    land_steps_max = int(FW_LAND_TIME * control_freq_hz)
    orange_idx     = 0
    # Blue quad leader software position (for smooth waypoint following)
    blue_leader    = np.array([HOME_XY[0], HOME_XY[1]+2.0, QUAD_ALT], dtype=float)

    banner(FW_SCANNING, f"Scanning desert | {len(FW_SCAN_WPS)} survey lines | Quads hover at corner")

    # ── Quad setup ──
    ctrls      = [DSLPIDControl(drone_model=cfg["model"]) for cfg in DRONE_CONFIGS]
    action     = np.zeros((NUM_DRONES,4))
    logger     = Logger(logging_freq_hz=control_freq_hz,num_drones=NUM_DRONES,output_folder=output_folder)
    target_pos = np.copy(init_xyzs)    # both quads start targeting their spawn = hover immediately
    dt         = 1.0/control_freq_hz
    t0         = time.time()
    step       = 0
    last_saved = [-999999]*NUM_DRONES

    print("[Fleet]  🟡 Yellow fixed-wing = SCOUT (phase 1)")
    print("[Fleet]  🔵 Blue  quad        = LEADER (phase 2)")
    print("[Fleet]  🟢 Green quad        = FOLLOWER (phase 2)")
    print(f"[Survey] {len(FW_SCAN_WPS)} grid lines | {len(ORANGE_WPS)} orange targets for squad")
    print("[Controls] Arrows:rotate | Z/X:zoom | F:free-cam | R:reset\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # ── STATE MACHINE ─────────────────────────────

            if mission_state == FW_SCANNING:

                # Move FW along survey grid
                fw_pos, fw_dir, reached = move_fw(fw_ids, fw_pos, fw_dir,
                                                   FW_SCAN_WPS[scan_idx], dt, env.CLIENT)

                # Proximity detection of waste sites
                for idx,(wx,wy,_) in enumerate(WASTE_POSITIONS):
                    if idx not in detected_sites and math.hypot(fw_pos[0]-wx,fw_pos[1]-wy)<FW_DETECT_RAD:
                        detected_sites.add(idx)
                        col=WASTE_COLORS[idx]
                        tag="*** ORANGE TARGET ***" if col=="orange" else ""
                        print(f"  [FW Scan] {col.upper()} waste at ({wx},{wy}) {tag}")

                if reached:
                    scan_idx += 1
                    print(f"  [FW Scan] Waypoint {scan_idx}/{len(FW_SCAN_WPS)} done")
                    if scan_idx >= len(FW_SCAN_WPS):
                        print(f"  [FW Scan] Complete — found {len(detected_sites)}/8 waste sites")
                        mission_state = FW_LANDING
                        land_step     = 0
                        banner(FW_LANDING, "Fixed-wing descending — quads prepare to launch")

                # Quads stay at their hover spots
                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

            elif mission_state == FW_LANDING:

                # Smoothly lower FW from FW_ALT → 0.2m
                alpha     = min(1.0, land_step / max(1, land_steps_max))
                fw_pos[2] = FW_ALT*(1.0-alpha) + 0.2*alpha
                _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)
                land_step += 1

                if land_step >= land_steps_max:
                    print("  [Fixed-wing] Landed.")
                    mission_state = SQUAD_GO
                    orange_idx    = 0
                    banner(SQUAD_GO,
                           f"Camera → Blue quad | Flying to {len(ORANGE_WPS)} orange waste sites")

                # Quads still hover
                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

            elif mission_state == SQUAD_GO:

                # Blue quad flies to each orange site as leader
                if orange_idx < len(ORANGE_WPS):
                    wp      = ORANGE_WPS[orange_idx]
                    diff    = wp[:2] - blue_leader[:2]
                    dist_xy = np.linalg.norm(diff)

                    if dist_xy < 1.0:
                        print(f"  [Squad] Reached orange site {orange_idx+1}/{len(ORANGE_WPS)}"
                              f" at {ORANGE_POS[orange_idx][:2]}")
                        orange_idx += 1
                    else:
                        dir_xy        = diff / (dist_xy + 1e-8)
                        blue_leader[:2] += dir_xy * QUAD_SPD * dt
                        blue_leader[2]   = QUAD_ALT

                    target_pos[0] = blue_leader.copy()

                    # Green follows blue in a line, 4m behind
                    if dist_xy > 0.5:
                        behind = -(diff / (dist_xy + 1e-8))
                    else:
                        behind = np.array([0.0, -1.0])
                    target_pos[1] = np.array([
                        blue_leader[0] + behind[0]*4.0,
                        blue_leader[1] + behind[1]*4.0,
                        QUAD_ALT
                    ])

                else:
                    mission_state = SQUAD_BACK
                    banner(SQUAD_BACK, "Inspection complete — returning to corner")

            elif mission_state == SQUAD_BACK:

                # Blue leader flies back to corner
                home_target = np.array([HOME_XY[0], HOME_XY[1]+2.0, QUAD_ALT])
                diff        = home_target[:2] - blue_leader[:2]
                dist_xy     = np.linalg.norm(diff)

                if dist_xy > 1.0:
                    dir_xy        = diff / (dist_xy + 1e-8)
                    blue_leader[:2] += dir_xy * QUAD_SPD * dt
                    blue_leader[2]   = QUAD_ALT
                else:
                    mission_state = DONE
                    banner(DONE, "All drones at corner — mission complete!")

                target_pos[0] = blue_leader.copy()
                behind = -(diff / (dist_xy + 1e-8)) if dist_xy > 0.5 else np.array([0.0, -1.0])
                target_pos[1] = np.array([
                    blue_leader[0] + behind[0]*4.0,
                    blue_leader[1] + behind[1]*4.0,
                    QUAD_ALT
                ])

            elif mission_state == DONE:
                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()
                _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)   # FW stays landed

            # ── Collision avoidance (quads only) ──
            for i in range(NUM_DRONES):
                push=np.zeros(2)
                for j in range(NUM_DRONES):
                    if j==i: continue
                    dx=obs[i][0]-obs[j][0]; dy=obs[i][1]-obs[j][1]
                    d=math.hypot(dx,dy)
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

            # Uncomment to enable quad onboard camera detection:
            # detect_save(obs, step, last_saved, DRONE_CONFIGS, env.CLIENT)

            # Camera: follow FW during scan/land, follow Blue quad after
            if mission_state in (FW_SCANNING, FW_LANDING):
                cam_target = fw_pos.tolist()
            else:
                cam_target = list(obs[0][0:3])   # Blue quad position

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
