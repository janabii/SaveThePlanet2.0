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
    """
    Full-coverage 2-D heatmap.

    Two grids are maintained in parallel:
      self.grid     — detection heat (white-pixel accumulation)
      self.coverage — how many camera frames have seen each cell
                      (updated every time the FW takes a photo)

    Colour legend for ground tiles and PNG:
      Dark grey         — never scanned (no coverage)
      Teal / blue-green — scanned, nothing found (clear)
      Yellow            — weak detection signal (low heat, watch area)
      Orange → Red      — strong detection signal (probable waste)
      White ring marker — confirmed hotspot peak
    """

    def __init__(self):
        self.grid     = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self.coverage = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self.viz_ids  = []

    # ── coordinate helpers ────────────────────────────────────────────
    def world_to_cell(self, wx, wy):
        cx = int(np.clip((wx - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        cy = int(np.clip((wy - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        return cx, cy

    def cell_to_world(self, cx, cy):
        return (GRID_MIN + (cx + 0.5) * GRID_CELL,
                GRID_MIN + (cy + 0.5) * GRID_CELL)

    # ── mark which cells the FW camera footprint covered ─────────────
    def mark_coverage(self, fw_pos, fw_dir):
        """
        Stamp the ground footprint of the FW camera into self.coverage.
        Called every time a camera frame is captured (whether or not
        any white pixels were found).
        """
        cam_h    = fw_pos[2] + FW_CAM_OFFSET
        fov_rad  = math.radians(FW_CAM_FOV)
        half_fwd = cam_h * math.tan(fov_rad / 2)
        half_rgt = half_fwd * (FW_CAM_W / FW_CAM_H)

        yaw = math.atan2(fw_dir[1], fw_dir[0])
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        rgt = np.array([math.sin(yaw), -math.cos(yaw)])

        # Walk a coarse grid of sample points across the footprint
        for nu in np.linspace(-1.0, 1.0, 8):
            for nr in np.linspace(-1.0, 1.0, 8):
                wx = fw_pos[0] + nu * half_fwd * fwd[0] + nr * half_rgt * rgt[0]
                wy = fw_pos[1] + nu * half_fwd * fwd[1] + nr * half_rgt * rgt[1]
                if GRID_MIN <= wx <= GRID_MAX and GRID_MIN <= wy <= GRID_MAX:
                    cx, cy = self.world_to_cell(wx, wy)
                    self.coverage[cx, cy] += 1.0

    # ── accumulate detection heat ─────────────────────────────────────
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

    # ── peak extraction ───────────────────────────────────────────────
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

    # ── semantic cell colour ──────────────────────────────────────────
    @staticmethod
    def _cell_colour(heat_norm, cov):
        """
        Return RGBA for a single cell based on coverage and heat level.

        heat_norm : float in [0, 1]  — normalised detection heat
        cov       : float            — raw coverage count (0 = never seen)

        Colour bands:
          cov == 0              → dark grey (0.25, 0.25, 0.25)  no data
          cov > 0, heat < 0.05  → teal (0.10, 0.65, 0.60)       clear
          0.05 <= heat < 0.30   → yellow (0.95, 0.90, 0.10)     weak signal
          0.30 <= heat < 0.65   → orange (1.00, 0.45, 0.00)     medium signal
          heat >= 0.65          → red→white hot ramp             strong signal
        """
        if cov == 0:
            return [0.25, 0.25, 0.25, 0.45]   # dark grey — no data
        if heat_norm < 0.05:
            return [0.10, 0.65, 0.60, 0.40]   # teal — scanned clear
        if heat_norm < 0.30:
            return [0.95, 0.90, 0.10, 0.55]   # yellow — weak
        if heat_norm < 0.65:
            return [1.00, 0.45, 0.00, 0.65]   # orange — medium
        # Hot ramp: red → white
        t = (heat_norm - 0.65) / 0.35         # 0..1 within the hot zone
        r = 1.0
        g = float(np.clip(t * 1.5,       0.0, 1.0))
        b = float(np.clip(t * 3.0 - 1.0, 0.0, 1.0))
        return [r, g, b, 0.75]

    # ── PyBullet full-ground visualisation ────────────────────────────
    def visualize(self, cid):
        """Render every grid cell as a flat coloured tile on the ground."""
        if self.viz_ids:
            return
        gmax = float(self.grid.max()) if self.grid.max() > 1e-6 else 1.0
        count = 0
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                heat_n = float(self.grid[cx, cy]) / gmax
                cov    = float(self.coverage[cx, cy])
                colour = self._cell_colour(heat_n, cov)
                wx, wy = self.cell_to_world(cx, cy)
                vs  = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[GRID_CELL * 0.48, GRID_CELL * 0.48, 0.010],
                    rgbaColor=colour,
                    physicsClientId=cid)
                bid = p.createMultiBody(0, -1, vs, [wx, wy, 0.05], physicsClientId=cid)
                self.viz_ids.append(bid)
                count += 1
        print(f"  [Heatmap] Full-ground render: {count} tiles "
              f"(dark-grey=unseen, teal=clear, yellow=weak, orange=medium, red/white=hot)")

    def mark_peaks(self, peaks, cid):
        """Yellow ring around each confirmed hotspot peak."""
        for wx, wy in peaks:
            vs = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.8, length=0.06,
                rgbaColor=[1.0, 1.0, 0.0, 0.95],
                physicsClientId=cid)
            p.createMultiBody(0, -1, vs, [wx, wy, 0.20], physicsClientId=cid)

    # ── save annotated PNG ────────────────────────────────────────────
    def save_png(self, path=HEATMAP_PNG):
        if not _HAS_CV2:
            csv = path.replace(".png", ".csv")
            np.savetxt(csv, self.grid, fmt="%.3f")
            print(f"  [Heatmap] CSV saved -> {csv}  (install opencv-python for PNG)")
            return

        H, W = GRID_N, GRID_N
        # Build an RGB image pixel-by-pixel using the same colour logic
        img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        gmax = float(self.grid.max()) if self.grid.max() > 1e-6 else 1.0

        for cx in range(W):
            for cy in range(H):
                heat_n = float(self.grid[cx, cy]) / gmax
                cov    = float(self.coverage[cx, cy])
                rgba   = self._cell_colour(heat_n, cov)
                r8 = int(rgba[0] * 255)
                g8 = int(rgba[1] * 255)
                b8 = int(rgba[2] * 255)
                # PNG coords: row = (H-1-cy) so world-Y increases upward
                img_rgb[H - 1 - cy, cx] = [b8, g8, r8]   # OpenCV BGR

        # Scale up for readability
        scale  = max(1, 800 // H)
        img_up = cv2.resize(img_rgb, (W * scale, H * scale),
                            interpolation=cv2.INTER_NEAREST)
        H2, W2 = img_up.shape[:2]

        # Grid lines every 5 cells
        for i in range(0, GRID_N + 1, 5):
            xi = i * scale
            yi = i * scale
            cv2.line(img_up, (xi, 0),  (xi, H2), (60, 60, 60), 1)
            cv2.line(img_up, (0, yi),  (W2, yi), (60, 60, 60), 1)
            wv = int(GRID_MIN + i * GRID_CELL)
            cv2.putText(img_up, str(wv), (xi + 2, H2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Legend
        legend = [
            ([64,  64,  64],  "No data"),
            ([153, 166, 26],  "Clear (scanned)"),
            ([26,  229, 242], "Weak signal"),
            ([0,   115, 255], "Medium signal"),
            ([0,   0,   255], "Hotspot"),
            ([0,   255, 255], "Peak marker"),
        ]
        lx, ly = 6, 16
        for bgr, label in legend:
            cv2.rectangle(img_up, (lx, ly - 10), (lx + 14, ly + 2), bgr, -1)
            cv2.putText(img_up, label, (lx + 18, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            ly += 18

        cv2.imwrite(path, img_up)
        print(f"  [Heatmap] PNG saved -> {path}")


# ======================================================================
# Camera utilities — fixed-wing and quadcopter
# ======================================================================

CAMERA_FRAMES_DIR = "camera_frames"

def _ensure_cam_dir():
    os.makedirs(CAMERA_FRAMES_DIR, exist_ok=True)


def get_fw_camera_rgb(fw_pos, fw_dir, cid):
    """Render a nadir (straight-down) frame from the fixed-wing."""
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


def save_fw_frame(rgb, frame_idx):
    """Save a fixed-wing surveillance frame to camera_frames/fw_NNNN.png."""
    if not _HAS_CV2:
        return
    _ensure_cam_dir()
    path = os.path.join(CAMERA_FRAMES_DIR, f"fw_{frame_idx:04d}.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def get_quad_camera_rgb(obs_state, cid, cam_w=320, cam_h=240):
    """
    Render a forward-facing inspection photo from a quadcopter, using its
    actual position and orientation from the physics state vector.

    obs_state : full drone state vector — pos[0:3], quat[3:7], ...
    Returns   : np.ndarray (H, W, 3) uint8 RGB
    """
    pos  = np.array(obs_state[0:3])
    quat = np.array(obs_state[3:7])

    rm  = p.getMatrixFromQuaternion(quat)
    fwd = np.array([rm[0], rm[3], rm[6]])   # body X = forward
    up  = np.array([rm[2], rm[5], rm[8]])   # body Z = up

    # Place the camera slightly above and ahead of the drone centroid
    eye     = pos + 0.05 * up + 0.03 * fwd
    look_at = eye + fwd

    view = p.computeViewMatrix(eye.tolist(), look_at.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60.0, cam_w / cam_h, 0.05, 50.0)
    raw  = p.getCameraImage(cam_w, cam_h, view, proj,
                            renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
    return np.reshape(raw[2], (cam_h, cam_w, 4))[:, :, :3].astype(np.uint8)


def save_quad_frame(rgb, drone_name, site_idx):
    """Save a quad inspection photo to camera_frames/site_NN_<name>.png."""
    if not _HAS_CV2:
        return
    _ensure_cam_dir()
    path = os.path.join(CAMERA_FRAMES_DIR, f"site_{site_idx:02d}_{drone_name}.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"    [Camera] Saved {path}")


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
    fw_frame_idx = 0
    total_frames = 0
    total_px     = 0
    _ensure_cam_dir()

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
                    # Always mark coverage regardless of detections
                    heatmap.mark_coverage(fw_pos, fw_dir)
                    # Save every frame to disk
                    save_fw_frame(rgb, fw_frame_idx)
                    fw_frame_idx += 1
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
                        site_n = orange_idx + 1
                        # Blue quad — forward-facing photo using real drone orientation
                        blue_rgb = get_quad_camera_rgb(obs[0], env.CLIENT)
                        save_quad_frame(blue_rgb, "blue", site_n)
                        # Green quad — same site, offset position
                        green_rgb = get_quad_camera_rgb(obs[1], env.CLIENT)
                        save_quad_frame(green_rgb, "green", site_n)
                        print(f"  [Squad] Site {site_n}/{len(BLUE_WPS)} inspected "
                              f"at ({wx:.1f}, {wy:.1f}) — photos saved to {CAMERA_FRAMES_DIR}/")
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