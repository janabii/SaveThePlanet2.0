"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission
CAMERA-BASED DETECTION + HEATMAP + AUTO-EXIT REPORT

MISSION FLOW:
  1. FW_SCANNING — Fixed-wing scans desert with lawnmower grid.
                   Downward camera detects WHITE waste cubes via HSV.
                   Detections are accumulated into a 34x34 heatmap grid.
  2. FW_RETURN   — Fixed-wing flies back to corner.
  3. FW_LAND     — Fixed-wing descends. Heatmap peaks are extracted,
                   detection report is printed, heatmap PNG saved,
                   and the simulation exits automatically.

  NOTE: Waste cubes are WHITE so they are unambiguously distinct from
        the orange/tan desert sand, eliminating false positives.

Controls (during scan):
  Camera: Arrow keys | Z/X zoom | F free-cam (WASD+QE) | Mouse | R reset
  Exit early: Ctrl+C
"""

import time
import math
import argparse
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

# ======================================================================
# Mission States
# ======================================================================
FW_SCANNING = 0
FW_RETURN   = 1
FW_LAND     = 2
SQUAD_GO    = 3
SQUAD_BACK  = 4
QUAD_LAND   = 5
DONE        = 6
STATE_NAMES = {
    0: "FW_SCANNING",
    1: "FW_RETURNING",
    2: "FW_LANDING",
    3: "SQUAD_INSPECTION",
    4: "SQUAD_RETURNING",
    5: "QUAD_LANDING",
    6: "DONE",
}

# ======================================================================
# World / Flight Parameters
# ======================================================================
HOME_XY = np.array([-15.0, -15.0])
FW_ALT  = 5.0
QUAD_ALT = 3.5

HOME_FW = np.array([HOME_XY[0], HOME_XY[1], FW_ALT])

QUAD_HOVER = np.array([
    [HOME_XY[0],       HOME_XY[1] + 2.5, QUAD_ALT],
    [HOME_XY[0] + 2.5, HOME_XY[1],       QUAD_ALT],
])

# Boustrophedon lawnmower waypoints
_ys  = [-16.0, -10.0, -4.0, 2.0, 8.0, 14.0]
_wps = []
for _i, _y in enumerate(_ys):
    if _i % 2 == 0: _wps += [[-17.0, _y, FW_ALT], [17.0, _y, FW_ALT]]
    else:           _wps += [[ 17.0, _y, FW_ALT], [-17.0, _y, FW_ALT]]
FW_SCAN_WPS = np.array(_wps, dtype=float)

FW_SPEED     = 4.5
FW_WP_RADIUS = 2.0
FW_LAND_TIME = 4.0
SQUAD_SPD    = 2.0
LAND_TIME    = 4.0

WORLD_LIMIT = 19.0
Z_MIN, Z_MAX = 1.0, 7.0
MIN_SEP = 2.2; AVOID_GAIN = 0.8; MAX_PUSH = 1.0

# ======================================================================
# Waste Ground Truth  (positions only - cubes are placed here)
# ======================================================================
WASTE_POSITIONS = [
    (-15.0, -12.0, 0.0), (-15.0,  12.0, 0.0),
    (  0.0, -15.0, 0.0), (  0.0,  15.0, 0.0),
    ( 15.0, -12.0, 0.0), ( 15.0,  12.0, 0.0),
    ( -8.0,   0.0, 0.0), (  8.0,   0.0, 0.0),
]
# WHITE cubes - maximum contrast against tan/orange sand
CUBE_COLOR  = [1.0, 1.0, 1.0, 1.0]
NUM_TARGETS = len(WASTE_POSITIONS)

# ======================================================================
# Drone Config
# ======================================================================
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_blue",  "color": [0.2, 0.2, 1.0, 1.0]},
    {"model": DroneModel.CF2X, "role": "quad_green", "color": [0.2, 0.8, 0.2, 1.0]},
]
NUM_DRONES = len(DRONE_CONFIGS)

# ======================================================================
# Simulation defaults
# ======================================================================
DEFAULT_PHYSICS            = Physics("pyb")
DEFAULT_GUI                = True
DEFAULT_PLOT               = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ    = 30
DEFAULT_OUTPUT_FOLDER      = "results"

# ======================================================================
# Main viewport camera
# ======================================================================
CAM_DIST_DEFAULT  = 20.0; CAM_YAW_DEFAULT = 45.0; CAM_PITCH_DEFAULT = -40.0
cam_dist  = CAM_DIST_DEFAULT
cam_yaw   = CAM_YAW_DEFAULT
cam_pitch = CAM_PITCH_DEFAULT
cam_tx = cam_ty = cam_tz = 0.0
CAM_YAW_STEP   = 2.0; CAM_PITCH_STEP   = 1.5
CAM_DIST_STEP  = 0.5; CAM_TARGET_STEP  = 1.0
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0
free_cam = False

# ======================================================================
# Fixed-wing downward camera
# ======================================================================
FW_CAM_FOV    = 90.0    # wide-angle: ~10 m footprint at 5 m altitude
FW_CAM_W      = 320
FW_CAM_H      = 240
FW_CAM_OFFSET = 0.35    # metres above FW centroid
FW_CAM_SAMPLE = 4       # capture 1 frame every N control steps

# WHITE detection (HSV): low saturation + very high brightness
# Desert sand: H~17, S~100-140 -- the S gate alone rejects sand entirely
WHITE_HSV_LO = np.array([  0,  0, 210], dtype=np.uint8)
WHITE_HSV_HI = np.array([179, 55, 255], dtype=np.uint8)
MIN_WHITE_PX  = 4       # min pixels per frame to accumulate
MAX_PROJ_SAMP = 200     # max pixels back-projected per frame

# ======================================================================
# Heatmap parameters
# ======================================================================
GRID_MIN  = -17.0
GRID_MAX  =  17.0
GRID_CELL =  1.0
GRID_N    = int((GRID_MAX - GRID_MIN) / GRID_CELL)   # 34

HEAT_SPREAD   = 1.0     # Gaussian sigma in cells
PEAK_FRAC     = 0.25    # threshold = PEAK_FRAC x grid_max
PEAK_MIN_HEAT = 3.0     # absolute floor
PEAK_MIN_SEP  = 4.0     # min distance (m) between accepted peaks
HEATMAP_PNG   = "heatmap_detection.png"


# ======================================================================
# WasteHeatmap
# ======================================================================
class WasteHeatmap:
    """2-D occupancy heatmap for camera-detected white waste cubes."""

    def __init__(self):
        self.grid    = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self.viz_ids = []

    def world_to_cell(self, wx, wy):
        cx = int(np.clip((wx - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        cy = int(np.clip((wy - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        return cx, cy

    def cell_to_world(self, cx, cy):
        return (GRID_MIN + (cx + 0.5) * GRID_CELL,
                GRID_MIN + (cy + 0.5) * GRID_CELL)

    def accumulate(self, world_pts, weight=1.0):
        spread = int(math.ceil(HEAT_SPREAD * 2)) + 1
        for wx, wy in world_pts:
            if not (GRID_MIN <= wx <= GRID_MAX and GRID_MIN <= wy <= GRID_MAX):
                continue
            cx, cy = self.world_to_cell(wx, wy)
            for dx in range(-spread, spread + 1):
                for dy in range(-spread, spread + 1):
                    ncx, ncy = cx + dx, cy + dy
                    if 0 <= ncx < GRID_N and 0 <= ncy < GRID_N:
                        d_sq = dx*dx + dy*dy
                        self.grid[ncx, ncy] += weight * math.exp(
                            -d_sq / (2.0 * HEAT_SPREAD**2))

    def find_peaks(self):
        gmax = float(self.grid.max())
        if gmax < PEAK_MIN_HEAT:
            return []
        threshold  = max(PEAK_FRAC * gmax, PEAK_MIN_HEAT)
        candidates = []
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                v = float(self.grid[cx, cy])
                if v >= threshold:
                    wx, wy = self.cell_to_world(cx, cy)
                    candidates.append((v, wx, wy))
        candidates.sort(reverse=True)
        peaks = []
        for _, wx, wy in candidates:
            if all(math.hypot(wx - px, wy - py) >= PEAK_MIN_SEP for px, py in peaks):
                peaks.append((wx, wy))
        return peaks

    def visualize(self, cid):
        if self.viz_ids:
            return
        gmax = float(self.grid.max())
        if gmax < 1e-6:
            return
        norm  = self.grid / gmax
        count = 0
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                v = float(norm[cx, cy])
                if v < 0.05:
                    continue
                wx, wy = self.cell_to_world(cx, cy)
                r = float(np.clip(v * 2.0,       0.0, 1.0))
                g = float(np.clip(v * 2.0 - 1.0, 0.0, 1.0))
                b = float(np.clip(v * 4.0 - 3.0, 0.0, 1.0))
                vs  = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[GRID_CELL*0.46, GRID_CELL*0.46, 0.012],
                    rgbaColor=[r, g, b, 0.55],
                    physicsClientId=cid)
                bid = p.createMultiBody(0, -1, vs, [wx, wy, 0.06], physicsClientId=cid)
                self.viz_ids.append(bid)
                count += 1
        print(f"  [Heatmap] Rendered {count} tiles on ground")

    def mark_peaks(self, peaks, cid):
        for wx, wy in peaks:
            vs = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.7, length=0.05,
                rgbaColor=[1.0, 1.0, 0.0, 0.95],
                physicsClientId=cid)
            p.createMultiBody(0, -1, vs, [wx, wy, 0.18], physicsClientId=cid)

    def save_png(self, path=HEATMAP_PNG):
        gmax = float(self.grid.max())
        if gmax < 1e-6:
            print("  [Heatmap] Grid empty - nothing saved.")
            return
        if _HAS_CV2:
            norm = (self.grid / gmax * 255.0).astype(np.uint8)
            img  = cv2.applyColorMap(norm.T[::-1, :], cv2.COLORMAP_HOT)
            h_img, w_img = img.shape[:2]
            for i in range(0, GRID_N + 1, 5):
                xi = int(i * w_img / GRID_N)
                yi = int(i * h_img / GRID_N)
                cv2.line(img, (xi, 0),    (xi, h_img), (50, 50, 50), 1)
                cv2.line(img, (0,  yi),   (w_img, yi), (50, 50, 50), 1)
                wv = int(GRID_MIN + i * GRID_CELL)
                cv2.putText(img, str(wv), (xi + 2, h_img - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
            cv2.imwrite(path, img)
            print(f"  [Heatmap] PNG saved -> {path}")
        else:
            csv = path.replace(".png", ".csv")
            np.savetxt(csv, self.grid, fmt="%.3f")
            print(f"  [Heatmap] CSV saved -> {csv}  (pip install opencv-python for PNG)")


# ======================================================================
# Fixed-wing downward camera
# ======================================================================

def get_fw_camera_rgb(fw_pos, fw_dir, cid):
    eye    = [fw_pos[0], fw_pos[1], fw_pos[2] + FW_CAM_OFFSET]
    target = [fw_pos[0], fw_pos[1], 0.0]
    fwd_xy = np.array([fw_dir[0], fw_dir[1], 0.0], dtype=float)
    n = np.linalg.norm(fwd_xy)
    fwd_xy = fwd_xy / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    view = p.computeViewMatrix(eye, target, fwd_xy.tolist())
    proj = p.computeProjectionMatrixFOV(FW_CAM_FOV, FW_CAM_W / FW_CAM_H, 0.1, 60.0)
    raw  = p.getCameraImage(FW_CAM_W, FW_CAM_H, view, proj,
                            renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
    return np.reshape(raw[2], (FW_CAM_H, FW_CAM_W, 4))[:, :, :3].astype(np.uint8)


def detect_white_world_pts(rgb, fw_pos, fw_dir):
    """
    Detect WHITE pixels in camera image and back-project to world XY.

    White = HSV(any hue, S<55, V>210).
    Sand  = HSV(H~17, S~100-140) -> completely rejected by S gate.
    """
    if _HAS_CV2:
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, WHITE_HSV_LO, WHITE_HSV_HI)
    else:
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        mask = ((r > 210) & (g > 210) & (b > 210)
                & (np.abs(r - g) < 30) & (np.abs(g - b) < 30)
                & (np.abs(r - b) < 30)).astype(np.uint8) * 255

    px_count = int(np.count_nonzero(mask))
    if px_count < MIN_WHITE_PX:
        return [], px_count

    ys, xs  = np.where(mask > 0)
    n_samp  = min(len(xs), MAX_PROJ_SAMP)
    idx     = np.random.choice(len(xs), n_samp, replace=False)

    cam_h    = fw_pos[2] + FW_CAM_OFFSET
    fov_rad  = math.radians(FW_CAM_FOV)
    half_fwd = cam_h * math.tan(fov_rad / 2)
    half_rgt = half_fwd * (FW_CAM_W / FW_CAM_H)

    yaw = math.atan2(fw_dir[1], fw_dir[0])
    fwd = np.array([ math.cos(yaw),  math.sin(yaw)])
    rgt = np.array([ math.sin(yaw), -math.cos(yaw)])

    world_pts = []
    for px, py in zip(xs[idx], ys[idx]):
        ndc_r =  (px / FW_CAM_W - 0.5) * 2.0
        ndc_u = -(py / FW_CAM_H - 0.5) * 2.0
        wx = fw_pos[0] + ndc_u * half_fwd * fwd[0] + ndc_r * half_rgt * rgt[0]
        wy = fw_pos[1] + ndc_u * half_fwd * fwd[1] + ndc_r * half_rgt * rgt[1]
        world_pts.append((wx, wy))

    return world_pts, px_count


# ======================================================================
# Fixed-wing visual model
# ======================================================================

def _make_box(half, color, cid):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    v = p.createVisualShape( p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(0, c, v, [0, 0, -300], physicsClientId=cid)

def spawn_fixedwing(cid):
    Y = [1.0, 0.85, 0.0, 1.0]; D = [0.8, 0.60, 0.0, 1.0]
    ids = [
        _make_box([0.55, 0.055, 0.055], Y, cid),
        _make_box([0.07, 0.85,  0.018], Y, cid),
        _make_box([0.06, 0.28,  0.012], D, cid),
        _make_box([0.05, 0.012, 0.20 ], D, cid),
    ]
    print(f"[Fixed-wing] Spawned (body ID:{ids[0]})")
    return ids

def _place_fw(fw_ids, pos, direction, cid):
    yaw = math.atan2(direction[1], direction[0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    q = p.getQuaternionFromEuler([0, 0, yaw])
    for bid, (lx, ly, lz) in zip(fw_ids,
            [(0,0,0),(0,0,0),(-0.52,0,0),(-0.52,0,0.16)]):
        wx = cy*lx - sy*ly; wy = sy*lx + cy*ly
        p.resetBasePositionAndOrientation(
            bid, [pos[0]+wx, pos[1]+wy, pos[2]+lz], q, physicsClientId=cid)

def move_fw(fw_ids, fw_pos, fw_dir, target, dt, cid):
    diff = target - fw_pos
    dist = math.hypot(diff[0], diff[1])
    if dist < FW_WP_RADIUS:
        _place_fw(fw_ids, fw_pos, fw_dir, cid)
        return fw_pos.copy(), fw_dir.copy(), True
    nd  = diff / (np.linalg.norm(diff) + 1e-8)
    np_ = fw_pos + nd * FW_SPEED * dt
    np_[0] = np.clip(np_[0], -WORLD_LIMIT, WORLD_LIMIT)
    np_[1] = np.clip(np_[1], -WORLD_LIMIT, WORLD_LIMIT)
    np_[2] = target[2]
    _place_fw(fw_ids, np_, nd, cid)
    return np_, nd, False


def move_leader(ldr, wp, dt, speed=SQUAD_SPD):
    """Move software leader toward wp. Returns (new_ldr, reached)."""
    diff = wp[:2] - ldr[:2]
    dist = np.linalg.norm(diff)
    if dist < 1.0:
        new_ldr = ldr.copy(); new_ldr[2] = wp[2]
        return new_ldr, True
    dir2d = diff / dist
    new_ldr = ldr.copy()
    new_ldr[:2] = ldr[:2] + dir2d * speed * dt
    new_ldr[0]  = np.clip(new_ldr[0], -WORLD_LIMIT, WORLD_LIMIT)
    new_ldr[1]  = np.clip(new_ldr[1], -WORLD_LIMIT, WORLD_LIMIT)
    new_ldr[2]  = wp[2]
    return new_ldr, False


# ======================================================================
# Main viewport camera
# ======================================================================

def update_cam(target, keys):
    global cam_dist, cam_yaw, cam_pitch, cam_tx, cam_ty, cam_tz, free_cam
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        free_cam = not free_cam
        if free_cam: cam_tx, cam_ty, cam_tz = target[0], target[1], target[2]
    if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN: cam_yaw -= CAM_YAW_STEP
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: cam_yaw += CAM_YAW_STEP
    if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch + CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN:
        cam_pitch = np.clip(cam_pitch - CAM_PITCH_STEP, CAM_PITCH_MIN, CAM_PITCH_MAX)
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN: cam_dist = max(0.1, cam_dist - CAM_DIST_STEP)
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN: cam_dist += CAM_DIST_STEP
    if free_cam:
        cy_ = np.cos(np.radians(cam_yaw)); sy_ = np.sin(np.radians(cam_yaw))
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN: cam_tx += CAM_TARGET_STEP*cy_; cam_ty += CAM_TARGET_STEP*sy_
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN: cam_tx -= CAM_TARGET_STEP*cy_; cam_ty -= CAM_TARGET_STEP*sy_
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN: cam_tz += CAM_TARGET_STEP*0.5
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN: cam_tz -= CAM_TARGET_STEP*0.5
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        cam_dist, cam_yaw, cam_pitch = CAM_DIST_DEFAULT, CAM_YAW_DEFAULT, CAM_PITCH_DEFAULT
        free_cam = False
    tgt = [cam_tx, cam_ty, cam_tz] if free_cam else list(target)
    p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, tgt)


# ======================================================================
# World
# ======================================================================

def spawn_waste(cid):
    """Spawn WHITE cubes - unambiguously distinct from the sandy terrain."""
    ids = []
    half = 0.30
    for pos in WASTE_POSITIONS:
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3, physicsClientId=cid)
        vs = p.createVisualShape(   p.GEOM_BOX, halfExtents=[half]*3,
                                    rgbaColor=CUBE_COLOR, physicsClientId=cid)
        bid = p.createMultiBody(0, cs, vs, basePosition=pos, physicsClientId=cid)
        ids.append(bid)
    print(f"[World] {len(ids)} WHITE waste cubes spawned")
    return ids


def print_report(peaks, heatmap, total_frames, total_px):
    """Print a precision/recall detection report vs ground truth."""
    sep = "=" * 62
    print(f"\n{sep}")
    print("  SCAN COMPLETE - DETECTION REPORT")
    print(sep)
    print(f"  Camera frames with detections  : {total_frames}")
    print(f"  Cumulative white pixels seen   : {total_px}")
    print(f"  Heatmap peak cell value        : {heatmap.grid.max():.2f}")
    print(f"  Ground-truth cube sites        : {NUM_TARGETS}")
    print(f"  Detected hotspots              : {len(peaks)}")
    print()

    MATCH_RADIUS = 4.0    # metres - peak within this = true positive
    true_pos     = 0
    false_pos    = 0
    matched_gt   = set()

    if peaks:
        print("  Hotspot breakdown:")
        for i, (px, py) in enumerate(peaks):
            dists  = [math.hypot(px - gx, py - gy) for gx, gy, _ in WASTE_POSITIONS]
            near_d = min(dists)
            near_i = int(np.argmin(dists))
            heat_v = heatmap.grid[heatmap.world_to_cell(px, py)]

            if near_d <= MATCH_RADIUS and near_i not in matched_gt:
                matched_gt.add(near_i)
                true_pos += 1
                verdict = f"TRUE  POS  (GT#{near_i+1} dist={near_d:.1f}m)"
            else:
                false_pos += 1
                verdict = f"FALSE POS  (nearest GT dist={near_d:.1f}m)"

            print(f"    Peak {i+1:2d}: ({px:6.1f},{py:6.1f})  "
                  f"heat={heat_v:6.1f}  {verdict}")
    else:
        print("  No hotspots found.")

    missed    = NUM_TARGETS - len(matched_gt)
    precision = true_pos / max(len(peaks), 1) * 100
    recall    = true_pos / max(NUM_TARGETS, 1) * 100

    print()
    print(f"  True Positives  : {true_pos} / {NUM_TARGETS}")
    print(f"  False Positives : {false_pos}")
    print(f"  Missed (FN)     : {missed}")
    print(f"  Precision       : {precision:.1f}%")
    print(f"  Recall          : {recall:.1f}%")
    print(f"\n  Heatmap PNG     : {HEATMAP_PNG}")
    print(sep + "\n")


def banner(state, msg=""):
    print(f"\n{'='*60}\n  [{STATE_NAMES[state]}]  {msg}\n{'='*60}\n")


# ======================================================================
# MAIN
# ======================================================================

def run(
    physics            = DEFAULT_PHYSICS,
    gui                = DEFAULT_GUI,
    plot               = DEFAULT_PLOT,
    simulation_freq_hz = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz    = DEFAULT_CONTROL_FREQ_HZ,
    output_folder      = DEFAULT_OUTPUT_FOLDER,
):
    init_xyzs = np.array([
        [QUAD_HOVER[0][0], QUAD_HOVER[0][1], QUAD_ALT],
        [QUAD_HOVER[1][0], QUAD_HOVER[1][1], QUAD_ALT],
    ], dtype=float)
    init_rpys = np.zeros((NUM_DRONES, 3))

    env = CtrlAviary(
        drone_model    = DroneModel.CF2X,
        num_drones     = NUM_DRONES,
        initial_xyzs   = init_xyzs,
        initial_rpys   = init_rpys,
        physics        = physics,
        pyb_freq       = simulation_freq_hz,
        ctrl_freq      = control_freq_hz,
        gui            = gui,
        user_debug_gui = False,
    )

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,          0, physicsClientId=env.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1, physicsClientId=env.CLIENT)

    # Sand ground plane
    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0,0,0,0], physicsClientId=env.CLIENT)
    fv = p.createVisualShape(p.GEOM_BOX, halfExtents=[25,25,0.01],
                             rgbaColor=[0.87, 0.72, 0.53, 1.0], physicsClientId=env.CLIENT)
    p.createMultiBody(0, -1, fv, [0,0,-0.05], physicsClientId=env.CLIENT)

    for i, cfg in enumerate(DRONE_CONFIGS):
        try:
            p.changeVisualShape(env.DRONE_IDS[i], -1,
                                rgbaColor=cfg["color"], physicsClientId=env.CLIENT)
        except Exception:
            pass

    print("[World] Building desert terrain...")
    create_desert_terrain(
        "assets/terrain_desert_dunes.png",
        "assets/desert_sand.png",
        (0.15, 0.15, 1.5))

    exclusion = list(WASTE_POSITIONS) + [
        (HOME_XY[0],       HOME_XY[1],       0.0),
        (QUAD_HOVER[0][0], QUAD_HOVER[0][1], 0.0),
        (QUAD_HOVER[1][0], QUAD_HOVER[1][1], 0.0),
    ]
    spawn_desert_obstacles(12, 6, 35.0, exclusion, 5.0)
    waste_ids = spawn_waste(env.CLIENT)

    # Fixed-wing
    fw_ids = spawn_fixedwing(env.CLIENT)
    fw_pos = HOME_FW.copy().astype(float)
    fw_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)

    # Heatmap + counters
    heatmap      = WasteHeatmap()
    fw_cam_ctr   = 0
    total_frames = 0
    total_px     = 0

    # Mission state
    mission_state  = FW_SCANNING
    scan_idx       = 0
    fw_land_step   = 0
    fw_land_max    = int(FW_LAND_TIME * control_freq_hz)
    orange_idx     = 0
    quad_land_step = 0
    quad_land_max  = int(LAND_TIME * control_freq_hz)

    # Populated from heatmap after FW lands
    detected_sites = []
    BLUE_WPS  = np.empty((0, 3), dtype=float)
    GREEN_WPS = np.empty((0, 3), dtype=float)

    # Software leaders for smooth quad navigation
    blue_ldr  = QUAD_HOVER[0].copy().astype(float)
    green_ldr = QUAD_HOVER[1].copy().astype(float)

    banner(FW_SCANNING,
           f"WHITE cube detection | {len(FW_SCAN_WPS)} scan lines | "
           f"FOV={FW_CAM_FOV}deg | sample=1/{FW_CAM_SAMPLE}")

    ctrls      = [DSLPIDControl(drone_model=cfg["model"]) for cfg in DRONE_CONFIGS]
    action     = np.zeros((NUM_DRONES, 4))
    logger     = Logger(logging_freq_hz=control_freq_hz,
                        num_drones=NUM_DRONES, output_folder=output_folder)
    target_pos = np.copy(init_xyzs)
    dt         = 1.0 / control_freq_hz
    t0         = time.time()
    step       = 0

    print("[Fleet]  Yellow FW = camera scout | Blue quad = inspector leader | Green = wingman")
    print("[Info]   Quads fly to heatmap hotspots after FW lands, then sim exits with report")
    print("[Controls] Arrows:rotate | Z/X:zoom | F:free | R:reset | Ctrl+C:abort\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=env.CLIENT)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # ════════════════════════════════════════════════════════
            if mission_state == FW_SCANNING:

                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, FW_SCAN_WPS[scan_idx], dt, env.CLIENT)

                # Downward camera every FW_CAM_SAMPLE steps
                fw_cam_ctr += 1
                if fw_cam_ctr >= FW_CAM_SAMPLE:
                    fw_cam_ctr = 0
                    rgb = get_fw_camera_rgb(fw_pos, fw_dir, env.CLIENT)
                    pts, n_px = detect_white_world_pts(rgb, fw_pos, fw_dir)
                    if n_px >= MIN_WHITE_PX:
                        total_frames += 1
                        total_px     += n_px
                        heatmap.accumulate(pts, weight=1.0)
                        if total_frames % 8 == 1:
                            print(f"  [Cam] FW@({fw_pos[0]:5.1f},{fw_pos[1]:5.1f}) "
                                  f"white_px={n_px:4d}  heat_max={heatmap.grid.max():.1f}  "
                                  f"frames={total_frames}")

                if reached:
                    scan_idx += 1
                    print(f"  [FW] WP {scan_idx}/{len(FW_SCAN_WPS)} | "
                          f"heat_max={heatmap.grid.max():.1f} | frames={total_frames}")
                    if scan_idx >= len(FW_SCAN_WPS):
                        mission_state = FW_RETURN
                        banner(FW_RETURN, "Scan complete - returning to land")

                # Quads hold at base during scan
                target_pos[0] = QUAD_HOVER[0].copy()
                target_pos[1] = QUAD_HOVER[1].copy()

            # ════════════════════════════════════════════════════════
            elif mission_state == FW_RETURN:

                fw_pos, fw_dir, reached = move_fw(
                    fw_ids, fw_pos, fw_dir, HOME_FW, dt, env.CLIENT)
                if reached:
                    mission_state = FW_LAND
                    fw_land_step  = 0
                    banner(FW_LAND, "Landing - analysing heatmap...")
                target_pos[0] = QUAD_HOVER[0].copy()
                target_pos[1] = QUAD_HOVER[1].copy()

            # ════════════════════════════════════════════════════════
            elif mission_state == FW_LAND:

                alpha     = min(1.0, fw_land_step / max(1, fw_land_max))
                fw_pos[2] = FW_ALT * (1.0 - alpha) + 0.2 * alpha
                _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)
                fw_land_step += 1
                target_pos[0] = QUAD_HOVER[0].copy()
                target_pos[1] = QUAD_HOVER[1].copy()

                if fw_land_step >= fw_land_max:
                    # Extract peaks from heatmap and build quad waypoints
                    peaks = heatmap.find_peaks()
                    heatmap.visualize(env.CLIENT)
                    heatmap.mark_peaks(peaks, env.CLIENT)
                    heatmap.save_png(HEATMAP_PNG)

                    if not peaks:
                        print("  [Heatmap] No hotspots found - quads staying grounded.")
                        detected_sites = []
                        BLUE_WPS  = np.empty((0, 3), dtype=float)
                        GREEN_WPS = np.empty((0, 3), dtype=float)
                    else:
                        detected_sites = peaks
                        BLUE_WPS  = np.array([[x, y,       QUAD_ALT] for x, y in peaks], dtype=float)
                        GREEN_WPS = np.array([[x, y + 3.0, QUAD_ALT] for x, y in peaks], dtype=float)
                        print(f"  [Heatmap] {len(peaks)} hotspot(s) — launching quads for close inspection")

                    mission_state = SQUAD_GO
                    orange_idx    = 0
                    banner(SQUAD_GO, f"Camera -> Blue quad | {len(BLUE_WPS)} site(s) to inspect")

            # ════════════════════════════════════════════════════════
            elif mission_state == SQUAD_GO:

                if len(BLUE_WPS) == 0:
                    mission_state = SQUAD_BACK
                    banner(SQUAD_BACK, "No sites - returning immediately")
                elif orange_idx < len(BLUE_WPS):
                    blue_ldr,  blue_reached = move_leader(blue_ldr,  BLUE_WPS[orange_idx],  dt)
                    green_ldr, _            = move_leader(green_ldr, GREEN_WPS[orange_idx], dt)

                    if blue_reached:
                        wx, wy = detected_sites[orange_idx]
                        # Quad takes a close-up photo at the site
                        _ = get_fw_camera_rgb(
                            np.array([wx, wy, QUAD_ALT]), np.array([1.0, 0.0, 0.0]), env.CLIENT)
                        print(f"  [Squad] Site {orange_idx+1}/{len(BLUE_WPS)} inspected "
                              f"at ({wx:.1f}, {wy:.1f}) — photo captured")
                        orange_idx += 1
                else:
                    mission_state = SQUAD_BACK
                    banner(SQUAD_BACK, "All sites inspected - returning to base")

                target_pos[0] = blue_ldr.copy()
                target_pos[1] = green_ldr.copy()

            # ════════════════════════════════════════════════════════
            elif mission_state == SQUAD_BACK:

                blue_ldr,  blue_home  = move_leader(blue_ldr,  QUAD_HOVER[0], dt)
                green_ldr, green_home = move_leader(green_ldr, QUAD_HOVER[1], dt)
                target_pos[0] = blue_ldr.copy()
                target_pos[1] = green_ldr.copy()

                if blue_home and green_home:
                    mission_state  = QUAD_LAND
                    quad_land_step = 0
                    banner(QUAD_LAND, "Both quads descending next to fixed-wing")

            # ════════════════════════════════════════════════════════
            elif mission_state == QUAD_LAND:

                alpha = min(1.0, quad_land_step / max(1, quad_land_max))
                z_now = QUAD_ALT * (1.0 - alpha) + 0.2 * alpha
                for i in range(NUM_DRONES):
                    target_pos[i] = np.array([QUAD_HOVER[i][0], QUAD_HOVER[i][1], z_now])
                quad_land_step += 1

                if quad_land_step >= quad_land_max:
                    # Print final report then exit
                    peaks = heatmap.find_peaks()
                    print_report(peaks, heatmap, total_frames, total_px)
                    mission_state = DONE
                    break   # <-- simulation exits here

            # ════════════════════════════════════════════════════════
            elif mission_state == DONE:
                break

            # ── Always refresh FW visual ──────────────────────────
            _place_fw(fw_ids, fw_pos, fw_dir, env.CLIENT)

            # ── Quad collision avoidance ─────────────────────────
            for i in range(NUM_DRONES):
                push = np.zeros(2)
                for j in range(NUM_DRONES):
                    if j == i: continue
                    dx = obs[i][0] - obs[j][0]; dy = obs[i][1] - obs[j][1]
                    d  = math.hypot(dx, dy)
                    if 1e-6 < d < MIN_SEP:
                        push += AVOID_GAIN * (MIN_SEP-d) / d * np.array([dx, dy])
                n_ = np.linalg.norm(push)
                if n_ > MAX_PUSH: push *= MAX_PUSH / n_
                target_pos[i, 0:2] = np.clip(target_pos[i, 0:2]+push, -WORLD_LIMIT, WORLD_LIMIT)
                target_pos[i, 2]   = np.clip(target_pos[i, 2], Z_MIN, Z_MAX)

            # ── PID ──────────────────────────────────────────────
            for i, ctrl in enumerate(ctrls):
                raw = ctrl.computeControlFromState(
                    control_timestep = dt, state = obs[i],
                    target_pos       = target_pos[i],
                    target_rpy       = np.zeros(3),
                    target_vel       = np.zeros(3),
                    target_rpy_rates = np.zeros(3),
                )
                action[i, :] = raw[0]
                logger.log(drone=i, timestamp=step*dt, state=obs[i])

            obs, _, _, _, _ = env.step(action)

            update_cam(
                fw_pos.tolist() if mission_state in (FW_SCANNING, FW_RETURN, FW_LAND)
                else list(obs[0][0:3]),
                keys)
            env.render()
            if gui: sync(step, t0, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        print("\n[Mission] Interrupted - partial report:")
        _peaks = heatmap.find_peaks()
        heatmap.save_png(HEATMAP_PNG)
        print_report(_peaks, heatmap, total_frames, total_px)

    finally:
        env.close()
        logger.save()
        logger.save_as_csv("mixed_fleet_mission")
        if plot: logger.plot()
        print("[Mission] Logs saved.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",  default=DEFAULT_GUI,  type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    args = parser.parse_args()
    run(**vars(args))