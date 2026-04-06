"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission
4-QUAD SWARM + CAMERA HEATMAP + NEAREST-SITE ASSIGNMENT

MISSION FLOW:
  1. FW_SCANNING  Fixed-wing lawnmower grid. Nadir camera detects WHITE
                  cubes via HSV. Detections accumulated into 34x34 heatmap.
                  All 4 quads hover at their start positions.

  2. FW_RETURN    Fixed-wing returns home.

  3. FW_LAND      FW descends. Peaks extracted from heatmap.
                  Each peak is assigned to the quad whose START position
                  is closest to it (greedy nearest-neighbour).
                  Heatmap tiles + yellow rings rendered on ground.

  4. SQUAD_GO     All 4 quads fly INDEPENDENTLY to their assigned sites:
                    - cruise to site at QUAD_CRUISE_ALT
                    - stop directly overhead (within 0.25 m)
                    - hover QUAD_DWELL_SECS for PID to settle
                    - capture 1280x960 nadir photo → quad_frames/
                    - move to next assigned site
                  Quads work in parallel; phase ends when ALL are done.
                  Viewport follows Quad-0 (blue).

  5. SQUAD_BACK   All quads return to their home hover spots.

  6. QUAD_LAND    All quads descend and land.

  7. DONE         Precision/recall report printed. Simulation exits.

Start positions (shared base at FW home corner HOME_XY = [-15, -15]):
  Quad 0 (Blue)   [-17, -13]   NW pad  (+3 m separation between all pads)
  Quad 1 (Green)  [-13, -13]   NE pad
  Quad 2 (Red)    [-17, -17]   SW pad
  Quad 3 (Purple) [-13, -17]   SE pad
  All quads deploy from and return to the same corner as the fixed-wing.

Controls:
  Arrows: rotate   Z/X: zoom   F: free-cam   R: reset   Ctrl+C: abort
"""

import os
import time
import math
import argparse
import numpy as np
import pybullet as p

# All asset paths are resolved relative to this script file, so the sim
# works regardless of which directory you launch it from.
_HERE = os.path.dirname(os.path.abspath(__file__))

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

# ======================================================================
# ODM terrain builder  (replaces desert_utils.create_desert_terrain)
# Uses PyBullet's native GEOM_HEIGHTFIELD — no external dependency,
# correct UV texture mapping, renders fully in the tiny renderer.
# ======================================================================
def build_odm_terrain(heightmap_path, texture_path, arena_m, max_height_m, cid):
    """
    Load an ODM heightmap + ortho-texture into PyBullet as a heightfield.

    Parameters
    ----------
    heightmap_path : str   — grayscale PNG, 128×128 recommended
    texture_path   : str   — square RGB PNG for the terrain texture
    arena_m        : float — world side length in metres (terrain is centred at origin)
    max_height_m   : float — height in metres that pixel value 255 maps to
    cid            : int   — PyBullet physics client ID
    """
    from PIL import Image as _PIL
    import numpy as _np

    # ── heightmap → flat float array ─────────────────────────────────
    hm_img  = _PIL.open(heightmap_path).convert('L')
    rows    = hm_img.height
    cols    = hm_img.width
    hm_arr  = _np.array(hm_img, dtype=_np.float32) / 255.0 * max_height_m

    # PyBullet GEOM_HEIGHTFIELD expects the data in row-major order
    # with (0,0) at the bottom-left corner → flip vertically
    hm_flat = _np.flipud(hm_arr).flatten().tolist()

    # meshScale: metres per cell in X and Y, Z kept at 1 (data already in m)
    cell = arena_m / max(rows, cols)
    shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD,
        meshScale        = [cell, cell, 1.0],
        heightfieldData  = hm_flat,
        numHeightfieldRows    = rows,
        numHeightfieldColumns = cols,
        physicsClientId  = cid,
    )

    # Centre the heightfield at the world origin.
    # PyBullet places the geometric centre of the heightfield at basePosition.
    # We want the minimum of the terrain to sit at z=0, so shift down by
    # half the height range (hm_arr.max() / 2 when min≈0).
    half_h = float(hm_arr.max()) / 2.0
    body = p.createMultiBody(
        baseMass        = 0,
        baseCollisionShapeIndex = shape,
        baseVisualShapeIndex    = shape,
        basePosition    = [0.0, 0.0, half_h],
        physicsClientId = cid,
    )

    # ── texture ──────────────────────────────────────────────────────
    try:
        tex_id = p.loadTexture(texture_path, physicsClientId=cid)
        p.changeVisualShape(body, -1,
                            textureUniqueId = tex_id,
                            rgbaColor       = [1, 1, 1, 1],
                            physicsClientId = cid)
        print(f"[Terrain] ODM map loaded — "
              f"{cols}×{rows}px  cell={cell:.3f}m  "
              f"h=0..{max_height_m}m  tex={tex_id}")
    except Exception as e:
        print(f"[Terrain] Texture load failed ({e}) — terrain will be grey")

    return body


def spawn_random_obstacles(n, min_r, arena_m, exclusions, min_sep, cid):
    """Scatter small rock/shrub stand-ins that avoid waste sites and base."""
    import random, math as _math
    lim = arena_m / 2.0
    placed = list(exclusions)
    for _ in range(n * 10):   # attempts
        if len(placed) - len(exclusions) >= n:
            break
        x = random.uniform(-lim + 1, lim - 1)
        y = random.uniform(-lim + 1, lim - 1)
        if any(_math.hypot(x-ex[0], y-ex[1]) < min_sep for ex in placed):
            continue
        h = random.uniform(0.2, 0.6)
        r = random.uniform(0.15, 0.35)
        vs = p.createVisualShape(p.GEOM_CYLINDER,
                                 radius=r, length=h,
                                 rgbaColor=[0.25, 0.20, 0.10, 1.0],
                                 physicsClientId=cid)
        p.createMultiBody(0, -1, vs,
                          basePosition=[x, y, h/2],
                          physicsClientId=cid)
        placed.append((x, y, 0.0))

# ======================================================================
# Mission states
# ======================================================================
FW_SCANNING   = 0
FW_RETURN     = 1
FW_LAND       = 2
SQUAD_GO      = 3
SQUAD_BACK    = 4
QUAD_LAND     = 5
DONE          = 6
STATE_NAMES = {
    0: "FW_SCANNING",     1: "FW_RETURNING",
    2: "FW_LANDING",      3: "SQUAD_INSPECTION",
    4: "SQUAD_RETURNING", 5: "QUAD_LANDING",
    6: "DONE",
}

# ======================================================================
# World geometry
# ======================================================================
HOME_XY = np.array([-15.0, -15.0])
FW_ALT  = 5.0
HOME_FW = np.array([HOME_XY[0], HOME_XY[1], FW_ALT])

QUAD_CRUISE_ALT = 4.0      # transit altitude
WORLD_LIMIT     = 19.0
Z_MIN, Z_MAX    = 0.2, 8.0

# ── Shared quad launch base  (matches FW home: HOME_XY) ────────────
#   All 4 quads deploy from pads around the same corner as the FW.
#   Pads are spread 3 m apart (> MIN_SEP=2.0) so collision avoidance
#   never fires while quads are sitting on the ground.
BASE_XY = HOME_XY   # alias — planners use BASE_XY for energy estimates

QUAD_START_XY = np.array([
    [HOME_XY[0] - 2.0, HOME_XY[1] + 2.0],   # 0 Blue   — NW
    [HOME_XY[0] + 2.0, HOME_XY[1] + 2.0],   # 1 Green  — NE
    [HOME_XY[0] - 2.0, HOME_XY[1] - 2.0],   # 2 Red    — SW
    [HOME_XY[0] + 2.0, HOME_XY[1] - 2.0],   # 3 Purple — SE
])
QUAD_HOVER = np.array(
    [[x, y, QUAD_CRUISE_ALT] for x, y in QUAD_START_XY], dtype=float)

# Seconds between each quad departing toward its first waypoint.
# Quads hold at QUAD_HOVER until their slot opens — simple, physics-safe.
TAKEOFF_STAGGER_SECS = 4.0

# Fraction of battery kept as planning reserve so every assigned quad
# is guaranteed enough energy to reach its last site AND fly home.
BATTERY_SAFETY_FRAC = 0.20   # keep 20 % as reserve (never planned away)

# ======================================================================
# Lawnmower survey waypoints
# ======================================================================
_ys  = [-16.0, -10.0, -4.0, 2.0, 8.0, 14.0]
_wps = []
for _i, _y in enumerate(_ys):
    _wps += [[-17.0, _y, FW_ALT], [17.0, _y, FW_ALT]] if _i % 2 == 0 \
        else [[17.0, _y, FW_ALT], [-17.0, _y, FW_ALT]]
FW_SCAN_WPS  = np.array(_wps, dtype=float)
FW_SPEED     = 4.5
FW_WP_RADIUS = 2.0
FW_LAND_SECS = 4.0

# ======================================================================
# Quad navigation
# ======================================================================
SQUAD_SPD   = 2.5
LAND_SECS   = 4.0
MIN_SEP     = 2.0     # collision avoidance trigger distance (m)
AVOID_GAIN  = 0.7
MAX_PUSH    = 0.8
ARRIVE_DIST = 0.25    # stop threshold for "overhead" arrival (m)

# ======================================================================
# Waste cubes  (WHITE — high contrast against tan/orange sand)
# ======================================================================
WASTE_POSITIONS = [
    (-15.0, -12.0, 0.4), (-15.0,  12.0, 0.4),
    (  0.0, -15.0, 0.4), (  0.0,  15.0, 0.4),
    ( 15.0, -12.0, 0.4), ( 15.0,  12.0, 0.4),
    ( -8.0,   0.0, 0.4), (  8.0,   0.0, 0.4),
]
CUBE_COLOR  = [1.0, 1.0, 1.0, 1.0]
NUM_TARGETS = len(WASTE_POSITIONS)

# ======================================================================
# 4 Drones
# ======================================================================
QUAD_COLORS = [
    [0.20, 0.20, 1.00, 1.0],   # Blue
    [0.20, 0.80, 0.20, 1.0],   # Green
    [1.00, 0.20, 0.20, 1.0],   # Red
    [0.70, 0.20, 0.90, 1.0],   # Purple
]
QUAD_NAMES  = ["blue", "green", "red", "purple"]
NUM_DRONES  = 4

# ======================================================================
# Simulation defaults
# ======================================================================
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI     = True
DEFAULT_PLOT    = False
DEFAULT_SIM_HZ  = 240
DEFAULT_CTRL_HZ = 30
DEFAULT_OUT_DIR = os.path.join(_HERE, "results")

# ======================================================================
# Viewport camera
# ======================================================================
_cam = dict(dist=25.0, yaw=45.0, pitch=-40.0, tx=0.0, ty=0.0, tz=0.0, free=False)
CAM_PITCH_MIN, CAM_PITCH_MAX = -89.0, -10.0

# ======================================================================
# Fixed-wing downward camera
# ======================================================================
FW_CAM_W      = 320
FW_CAM_H      = 240
FW_CAM_FOV    = 90.0
FW_CAM_OFFSET = 0.35
FW_CAM_SAMPLE  = 4
FW_CAM_FRAMES_DIR = os.path.join(_HERE, "fw_frames")  # set to "" to disable

WHITE_LO      = np.array([  0,  0, 200], dtype=np.uint8)   # V≥200
WHITE_HI      = np.array([179, 60, 255], dtype=np.uint8)   # S≤60 (slightly looser for real ortho bg)
MIN_WHITE_PX  = 4
MAX_PROJ_SAMP = 200

# ======================================================================
# Quad inspection camera
# ======================================================================
QUAD_CAM_W      = 800           # 800×600 — good quality, fast enough for tiny renderer
QUAD_CAM_H      = 600
QUAD_CAM_FOV    = 75.0          # ~9 m footprint at 5 m alt
QUAD_CAM_ALT    = 5.0           # overhead hover altitude (m)
QUAD_DWELL_SECS = 2.5           # hover duration before shooting
QUAD_FRAMES_DIR = os.path.join(_HERE, "quad_frames")

# ======================================================================
# Battery & Propulsion Model  (Zeng et al., IEEE 2018)
#
#   P(t) = P_hover + k_v * ||v||^2 + k_a * ||a||
#   E(t+dt) = E(t) - P(t) * dt
#
# CF2X (Crazyflie 2.X) — same parameters as reference swa5c.py.
# Capacity scaled to 4.44 Wh (15 984 J) for extended mission endurance.
# ======================================================================
BATTERY_CAPACITY_WH = 4.44
BATTERY_CAPACITY_J  = BATTERY_CAPACITY_WH * 3600   # 15 984 J

P_HOVER    = 4.0    # W        — constant hover baseline
K_VELOCITY = 0.05   # W/(m/s)^2 — aerodynamic drag term
K_ACCEL    = 0.02   # W/(m/s^2) — manoeuvre / motor-workload term

LOW_BATTERY_RTB      = 0.10   # 10% -> abort remaining sites, return to base
LOW_BATTERY_CRITICAL = 0.05   # 5%  -> emergency land in place

# ======================================================================
# Heatmap
# ======================================================================
GRID_MIN      = -17.0
GRID_MAX      =  17.0
GRID_CELL     =  1.0
GRID_N        = int((GRID_MAX - GRID_MIN) / GRID_CELL)   # 34

HEAT_SPREAD   = 1.2
PEAK_FRAC     = 0.28
PEAK_MIN_HEAT = 5.0
PEAK_MIN_SEP  = 4.5
HEATMAP_PNG   = "heatmap_detection.png"
MATCH_RADIUS  = 3.5    # m — peak within this distance of GT = true positive


# ======================================================================
# Battery system
# ======================================================================
class BatterySystem:
    """
    Per-drone battery tracker.

    Each drone starts at BATTERY_CAPACITY_J.  Every control step call
    update() with the current and previous velocity vectors; it computes
    instantaneous power (Zeng et al. 2018), drains the energy, and
    returns the new battery fractions.

    State vectors from CtrlAviary._getDroneStateVector:
        [0:3]  pos   [3:7]  quat   [7:10] rpy
        [10:13] vel  [13:16] ang_vel  [16:20] rpm
    """

    def __init__(self, n):
        self.n          = n
        self.energy_j   = np.full(n, BATTERY_CAPACITY_J, dtype=float)
        self.rtb_active = [False] * n      # set True when RTB triggered
        self.emergency  = [False] * n      # set True on critical threshold
        self._text_ids  = [-1] * n

    # ── core physics ─────────────────────────────────────────────────

    @staticmethod
    def _power(vel, accel):
        """Instantaneous power in Watts for one drone."""
        return (P_HOVER
                + K_VELOCITY * float(np.dot(vel, vel))
                + K_ACCEL    * float(np.linalg.norm(accel)))

    def update(self, obs, prev_vel, dt):
        """Update ALL drone batteries (including ground-sitting drones)."""
        power = np.zeros(self.n)
        for i in range(self.n):
            vel   = np.array(obs[i][10:13], dtype=float)
            accel = (vel - prev_vel[i]) / dt if dt > 0 else np.zeros(3)
            p     = self._power(vel, accel)
            power[i] = p
            self.energy_j[i] = max(0.0, self.energy_j[i] - p * dt)
            prev_vel[i] = vel
        levels = self.energy_j / BATTERY_CAPACITY_J
        return levels, power

    def update_airborne(self, obs, prev_vel, dt):
        """
        Update batteries only for quads that are airborne (z > 0.5 m).
        Grounded quads still have prev_vel updated but their energy is
        not drained — they are not yet using flight power.
        """
        power = np.zeros(self.n)
        for i in range(self.n):
            vel   = np.array(obs[i][10:13], dtype=float)
            accel = (vel - prev_vel[i]) / dt if dt > 0 else np.zeros(3)
            if obs[i][2] > 0.5:   # only drain if actually flying
                p = self._power(vel, accel)
                power[i] = p
                self.energy_j[i] = max(0.0, self.energy_j[i] - p * dt)
            prev_vel[i] = vel
        levels = self.energy_j / BATTERY_CAPACITY_J
        return levels, power

    # ── trip-energy estimator  (used by assignment planner) ──────────

    @staticmethod
    def estimate_trip_j(from_xy, site_xy, to_xy=None):
        """
        Estimate Joules consumed for one leg of an inspection trip.

        Models steady-state cruise (no acceleration impulse) because we
        are planning, not measuring.  Vertical climbs to QUAD_CAM_ALT
        are short and already covered by the hover term.

            P_cruise = P_hover + k_v * v²   (at SQUAD_SPD, a≈0)
            t_leg    = distance / SQUAD_SPD
            E_leg    = P_cruise * t_leg

        The full "from → site → base" round trip is:
            E = P_cruise*(d_from_site + d_site_to_base)/SQUAD_SPD
              + P_hover * QUAD_DWELL_SECS

        Parameters
        ----------
        from_xy  : (x, y) current or last-assigned position of the quad
        site_xy  : (x, y) candidate inspection site
        to_xy    : (x, y) return destination (defaults to BASE_XY)
        """
        if to_xy is None:
            to_xy = BASE_XY
        p_cruise = P_HOVER + K_VELOCITY * SQUAD_SPD ** 2
        d_out  = math.hypot(site_xy[0] - from_xy[0], site_xy[1] - from_xy[1])
        d_back = math.hypot(to_xy[0]   - site_xy[0], to_xy[1]   - site_xy[1])
        t_fly  = (d_out + d_back) / max(SQUAD_SPD, 1e-6)
        return p_cruise * t_fly + P_HOVER * QUAD_DWELL_SECS

    # ── RTB / emergency checks ────────────────────────────────────────

    def check_rtb(self, i, level, quad_done, leaders, obs):
        """
        Check drone i's battery and act if below threshold.
        Modifies quad_done[i] and leaders[i] in place.
        Returns a status string for logging.
        """
        if self.emergency[i]:
            leaders[i][2] = max(0.15, leaders[i][2] - 0.05)
            return "EMERGENCY"

        if level < LOW_BATTERY_CRITICAL and not self.emergency[i]:
            self.emergency[i]  = True
            self.rtb_active[i] = True
            quad_done[i]       = True
            print(f"\n  [BATTERY] Quad-{i} ({QUAD_NAMES[i]}) "
                  f"CRITICAL {level*100:.1f}% — emergency landing!")
            return "EMERGENCY"

        if level < LOW_BATTERY_RTB and not self.rtb_active[i]:
            self.rtb_active[i] = True
            quad_done[i]       = True
            print(f"\n  [BATTERY] Quad-{i} ({QUAD_NAMES[i]}) "
                  f"LOW {level*100:.1f}% — aborting inspection, RTB")

        if self.rtb_active[i] and not self.emergency[i]:
            # Steer leader back to this quad's pad at the shared base
            home = QUAD_HOVER[i]
            diff = home[:2] - leaders[i][:2]
            dist = np.linalg.norm(diff)
            if dist > 0.3:
                nd             = diff / dist
                leaders[i][:2] = leaders[i][:2] + nd * min(SQUAD_SPD * 0.033, dist)
            leaders[i][2] = QUAD_CRUISE_ALT
            return "RTB"

        return "OK"

    # ── PyBullet HUD ──────────────────────────────────────────────────

    def update_hud(self, obs, levels, cid):
        """Render / refresh battery % text floating above each drone."""
        for i in range(self.n):
            pct = levels[i] * 100.0

            if self.emergency[i]:
                label = f"Q{i} {pct:.0f}% EMRG"
                col   = [1.0, 0.0, 0.0]
            elif self.rtb_active[i]:
                label = f"Q{i} {pct:.0f}% RTB"
                col   = [1.0, 0.4, 0.0]
            elif pct > 60:
                label = f"Q{i} {pct:.0f}%"
                col   = [0.0, 1.0, 0.0]
            elif pct > 30:
                label = f"Q{i} {pct:.0f}%"
                col   = [1.0, 0.85, 0.0]
            else:
                label = f"Q{i} {pct:.0f}%"
                col   = [1.0, 0.0, 0.0]

            pos = list(obs[i][0:3])
            pos[2] += 0.55

            if self._text_ids[i] == -1:
                # First frame — create the item
                self._text_ids[i] = p.addUserDebugText(
                    text=label, textPosition=pos,
                    textColorRGB=col, textSize=1.1,
                    lifeTime=0, physicsClientId=cid)
            else:
                # Subsequent frames — replace in-place (no remove needed,
                # no risk of a ghost label being left behind)
                self._text_ids[i] = p.addUserDebugText(
                    text=label, textPosition=pos,
                    textColorRGB=col, textSize=1.1,
                    lifeTime=0,
                    replaceItemUniqueId=self._text_ids[i],
                    physicsClientId=cid)

    def cleanup_hud(self, cid):
        for tid in self._text_ids:
            if tid != -1:
                try:
                    p.removeUserDebugItem(tid, physicsClientId=cid)
                except Exception:
                    pass

    # ── report helper ─────────────────────────────────────────────────

    def summary(self):
        """Return list of per-drone summary strings for the final report."""
        lines = []
        for i in range(self.n):
            rem_j   = self.energy_j[i]
            rem_pct = rem_j / BATTERY_CAPACITY_J * 100.0
            used_j  = BATTERY_CAPACITY_J - rem_j
            status  = ("EMERGENCY" if self.emergency[i]
                       else "RTB" if self.rtb_active[i]
                       else "OK")
            lines.append(
                f"    Quad-{i} ({QUAD_NAMES[i]:6s})  "
                f"remaining={rem_pct:5.1f}%  "
                f"used={used_j/3600:.4f} Wh  status={status}")
        return lines


# ======================================================================
# Heatmap class
# ======================================================================
class WasteHeatmap:

    def __init__(self):
        self.grid    = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self.viz_ids = []

    def _wc(self, wx, wy):
        cx = int(np.clip((wx - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        cy = int(np.clip((wy - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        return cx, cy

    def _cw(self, cx, cy):
        return (GRID_MIN + (cx + 0.5) * GRID_CELL,
                GRID_MIN + (cy + 0.5) * GRID_CELL)

    def accumulate(self, world_pts, weight=1.0):
        spread  = int(math.ceil(HEAT_SPREAD * 2)) + 1
        two_s2  = 2.0 * HEAT_SPREAD ** 2
        for wx, wy in world_pts:
            if not (GRID_MIN <= wx <= GRID_MAX and GRID_MIN <= wy <= GRID_MAX):
                continue
            cx, cy = self._wc(wx, wy)
            for dx in range(-spread, spread + 1):
                for dy in range(-spread, spread + 1):
                    ncx, ncy = cx + dx, cy + dy
                    if 0 <= ncx < GRID_N and 0 <= ncy < GRID_N:
                        self.grid[ncx, ncy] += weight * math.exp(
                            -(dx*dx + dy*dy) / two_s2)

    def find_peaks(self):
        gmax = float(self.grid.max())
        if gmax < PEAK_MIN_HEAT:
            return []
        thresh = max(PEAK_FRAC * gmax, PEAK_MIN_HEAT)
        cands  = []
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                v = float(self.grid[cx, cy])
                if v >= thresh:
                    wx, wy = self._cw(cx, cy)
                    cands.append((v, wx, wy))
        cands.sort(reverse=True)
        peaks = []
        for _, wx, wy in cands:
            if all(math.hypot(wx - px, wy - py) >= PEAK_MIN_SEP
                   for px, py in peaks):
                peaks.append((wx, wy))
        return peaks

    def visualize(self, cid):
        if self.viz_ids:
            return
        gmax  = float(self.grid.max()) or 1.0
        norm  = self.grid / gmax
        count = 0
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                v = float(norm[cx, cy])
                if v < 0.04:
                    continue
                wx, wy = self._cw(cx, cy)
                # Heat-colour: blue→red→yellow→white as value rises
                r = float(np.clip(v * 2.0,       0.0, 1.0))
                g = float(np.clip(v * 2.0 - 1.0, 0.0, 1.0))
                b = float(np.clip(1.0 - v * 2.0, 0.0, 1.0))
                # Tile height scales with heat value so hot spots are taller
                tile_h = 0.05 + v * 0.30
                vs  = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[GRID_CELL * 0.46, GRID_CELL * 0.46, tile_h / 2],
                    rgbaColor=[r, g, b, 1.0],   # fully opaque
                    physicsClientId=cid)
                # Place tiles at z=0.4 so they sit above max terrain (0.25m)
                p.createMultiBody(0, -1, vs,
                                  [wx, wy, 0.4 + tile_h / 2],
                                  physicsClientId=cid)
                count += 1
        self.viz_ids = [count]
        print(f"  [Heatmap] {count} tiles rendered")

    def mark_peaks(self, peaks, cid):
        for wx, wy in peaks:
            vs = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.75, length=0.08,
                rgbaColor=[1.0, 1.0, 0.0, 1.0], physicsClientId=cid)
            p.createMultiBody(0, -1, vs, [wx, wy, 0.8], physicsClientId=cid)

    def mark_assignment(self, assignments, cid):
        """Draw a small coloured dot above each peak showing which quad owns it."""
        dot_colors = [
            [0.2, 0.2, 1.0, 1.0],   # blue
            [0.2, 0.8, 0.2, 1.0],   # green
            [1.0, 0.2, 0.2, 1.0],   # red
            [0.7, 0.2, 0.9, 1.0],   # purple
        ]
        for qi, sites in enumerate(assignments):
            for wx, wy in sites:
                vs = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.35,
                    rgbaColor=dot_colors[qi], physicsClientId=cid)
                p.createMultiBody(0, -1, vs, [wx, wy, 1.1], physicsClientId=cid)

    def save_png(self, path=HEATMAP_PNG):
        gmax = float(self.grid.max())
        if gmax < 1e-6:
            print("  [Heatmap] Grid empty — nothing saved.")
            return
        if not _HAS_CV2:
            np.savetxt(path.replace(".png", ".csv"), self.grid, fmt="%.3f")
            print("  [Heatmap] CSV saved (install opencv-python for PNG)")
            return
        norm = (self.grid / gmax * 255.0).astype(np.uint8)
        img  = cv2.applyColorMap(norm.T[::-1, :], cv2.COLORMAP_HOT)
        h, w = img.shape[:2]
        for i in range(0, GRID_N + 1, 5):
            xi = int(i * w / GRID_N); yi = int(i * h / GRID_N)
            cv2.line(img, (xi, 0),  (xi, h), (50, 50, 50), 1)
            cv2.line(img, (0,  yi), (w, yi), (50, 50, 50), 1)
            wv = int(GRID_MIN + i * GRID_CELL)
            cv2.putText(img, str(wv), (xi+2, h-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        cv2.imwrite(path, img)
        print(f"  [Heatmap] PNG saved -> {path}")


# ======================================================================
# Hotspot → quad assignment  (greedy nearest-neighbour)
# ======================================================================

def assign_peaks_to_quads(peaks, battery):
    """
    Energy-constrained site allocation.

    For each candidate site the planner estimates the Joule cost of the
    round trip  base → [prior sites] → site → base  using steady-state
    cruise power (Zeng et al. 2018).  A quad may only be assigned a site
    if its remaining PLANNING BUDGET (current energy minus the 20 % safety
    reserve already held back for the return leg) covers ALL committed
    costs PLUS the new trip.

    Algorithm
    ---------
    1. Sort sites nearest-to-base first so short trips get first pick,
       maximising total coverage.
    2. For each site test every quad:
         feasible  ←  committed_j[qi] + estimate_trip_j(last_xy, site, BASE_XY)
                       ≤  energy_j[qi] * (1 − BATTERY_SAFETY_FRAC)
    3. Among feasible quads, assign to the one with the MOST spare budget
       after this commitment (best-fit greedy — preserves optionality for
       future sites).
    4. Sites that no quad can reach are logged as infeasible and skipped.

    Returns
    -------
    assignments : list[list[(wx,wy)]]  — per-quad site lists
    skipped     : list[(wx,wy)]        — sites no quad could reach
    """
    if not peaks:
        return [[] for _ in range(NUM_DRONES)], []

    assignments  = [[] for _ in range(NUM_DRONES)]
    committed_j  = [0.0] * NUM_DRONES
    last_xy      = [tuple(BASE_XY)] * NUM_DRONES   # each quad starts at base

    # Effective planning budget (energy available for new trips)
    budget_j = [battery.energy_j[qi] * (1.0 - BATTERY_SAFETY_FRAC)
                for qi in range(NUM_DRONES)]

    # Sort nearest-to-base first for greedy coverage
    sorted_peaks = sorted(
        peaks,
        key=lambda s: math.hypot(s[0] - BASE_XY[0], s[1] - BASE_XY[1]))

    skipped = []

    for wx, wy in sorted_peaks:
        best_qi      = -1
        best_spare_j = -1.0

        for qi in range(NUM_DRONES):
            if battery.emergency[qi] or battery.rtb_active[qi]:
                continue   # drone already out of action

            trip_j = BatterySystem.estimate_trip_j(
                last_xy[qi], (wx, wy))   # to_xy defaults to BASE_XY
            total_j = committed_j[qi] + trip_j

            if total_j <= budget_j[qi]:
                spare = budget_j[qi] - total_j
                if spare > best_spare_j:
                    best_spare_j = spare
                    best_qi      = qi

        if best_qi >= 0:
            assignments[best_qi].append((wx, wy))
            trip_j             = BatterySystem.estimate_trip_j(
                last_xy[best_qi], (wx, wy))
            committed_j[best_qi] += trip_j
            last_xy[best_qi]     = (wx, wy)
        else:
            skipped.append((wx, wy))

    # ── print allocation table ────────────────────────────────────────
    print("\n  [Assignment] Energy-constrained allocation "
          f"(reserve={int(BATTERY_SAFETY_FRAC*100)}%):")
    for qi in range(NUM_DRONES):
        e_pct  = battery.energy_j[qi] / BATTERY_CAPACITY_J * 100.0
        c_pct  = committed_j[qi]      / BATTERY_CAPACITY_J * 100.0
        sites  = ", ".join(f"({x:.1f},{y:.1f})" for x, y in assignments[qi])
        print(f"    Quad-{qi} ({QUAD_NAMES[qi]:6s}) "
              f"batt={e_pct:5.1f}%  planned={c_pct:4.1f}%  "
              f"{len(assignments[qi])} site(s)  {sites or '(none)'}")
    if skipped:
        sk = ", ".join(f"({x:.1f},{y:.1f})" for x, y in skipped)
        print(f"    *** SKIPPED (no feasible quad): {sk}")
    print()
    return assignments, skipped


# ======================================================================
# Fixed-wing camera + detection
# ======================================================================

def fw_camera_rgb(fw_pos, fw_dir, cid):
    eye    = [fw_pos[0], fw_pos[1], fw_pos[2] + FW_CAM_OFFSET]
    target = [fw_pos[0], fw_pos[1], 0.0]
    fwd    = np.array([fw_dir[0], fw_dir[1], 0.0])
    n      = np.linalg.norm(fwd)
    fwd    = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    view   = p.computeViewMatrix(eye, target, fwd.tolist())
    proj   = p.computeProjectionMatrixFOV(FW_CAM_FOV, FW_CAM_W/FW_CAM_H, 0.1, 60.0)
    raw    = p.getCameraImage(FW_CAM_W, FW_CAM_H, view, proj,
                              renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
    return np.reshape(raw[2], (FW_CAM_H, FW_CAM_W, 4))[:, :, :3].astype(np.uint8)


def detect_white_pts(rgb, fw_pos, fw_dir):
    if _HAS_CV2:
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    else:
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        mask = ((r > 210) & (g > 210) & (b > 210)
                & (np.abs(r-g) < 30) & (np.abs(g-b) < 30)
                & (np.abs(r-b) < 30)).astype(np.uint8) * 255

    n_px = int(np.count_nonzero(mask))
    if n_px < MIN_WHITE_PX:
        return [], n_px

    ys, xs = np.where(mask > 0)
    ns     = min(len(xs), MAX_PROJ_SAMP)
    idx    = np.random.choice(len(xs), ns, replace=False)

    cam_h    = fw_pos[2] + FW_CAM_OFFSET
    half_fwd = cam_h * math.tan(math.radians(FW_CAM_FOV) / 2)
    half_rgt = half_fwd * (FW_CAM_W / FW_CAM_H)
    yaw      = math.atan2(fw_dir[1], fw_dir[0])
    fwd      = np.array([ math.cos(yaw),  math.sin(yaw)])
    rgt      = np.array([ math.sin(yaw), -math.cos(yaw)])

    pts = []
    for px, py in zip(xs[idx], ys[idx]):
        nr =  (px / FW_CAM_W - 0.5) * 2.0
        nu = -(py / FW_CAM_H - 0.5) * 2.0
        wx = fw_pos[0] + nu * half_fwd * fwd[0] + nr * half_rgt * rgt[0]
        wy = fw_pos[1] + nu * half_fwd * fwd[1] + nr * half_rgt * rgt[1]
        pts.append((wx, wy))
    return pts, n_px


# ======================================================================
# Quad inspection camera
# ======================================================================

def quad_camera_rgb(quad_pos, cid):
    """1280x960 nadir shot from directly above a site."""
    eye    = [quad_pos[0], quad_pos[1], float(quad_pos[2])]
    target = [quad_pos[0], quad_pos[1], 0.0]
    up     = [1.0, 0.0, 0.0]
    view   = p.computeViewMatrix(eye, target, up)
    proj   = p.computeProjectionMatrixFOV(
        QUAD_CAM_FOV, QUAD_CAM_W / QUAD_CAM_H, 0.1, 30.0)
    raw    = p.getCameraImage(QUAD_CAM_W, QUAD_CAM_H, view, proj,
                              renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
    return np.reshape(raw[2], (QUAD_CAM_H, QUAD_CAM_W, 4))[:, :, :3].astype(np.uint8)


def save_quad_photo(rgb, quad_name, site_idx):
    if not _HAS_CV2:
        return
    os.makedirs(QUAD_FRAMES_DIR, exist_ok=True)
    path = os.path.join(QUAD_FRAMES_DIR, f"site_{site_idx:02d}_{quad_name}.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"    [Cam-{quad_name}] {QUAD_CAM_W}x{QUAD_CAM_H} -> {path}")


# ======================================================================
# Fixed-wing visual model
# ======================================================================

def _box(half, color, cid):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    v = p.createVisualShape( p.GEOM_BOX, halfExtents=half, rgbaColor=color,
                             physicsClientId=cid)
    return p.createMultiBody(0, c, v, [0, 0, -500], physicsClientId=cid)

def spawn_fw(cid):
    Y = [1.0, 0.85, 0.0, 1.0]; D = [0.8, 0.60, 0.0, 1.0]
    return [_box([0.55, 0.055, 0.055], Y, cid),
            _box([0.07, 0.85,  0.018], Y, cid),
            _box([0.06, 0.28,  0.012], D, cid),
            _box([0.05, 0.012, 0.200], D, cid)]

def place_fw(ids, pos, dirn, cid):
    yaw = math.atan2(dirn[1], dirn[0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    q = p.getQuaternionFromEuler([0, 0, yaw])
    for bid, (lx, ly, lz) in zip(ids,
            [(0,0,0),(0,0,0),(-0.52,0,0),(-0.52,0,0.16)]):
        p.resetBasePositionAndOrientation(
            bid,
            [pos[0]+cy*lx-sy*ly, pos[1]+sy*lx+cy*ly, pos[2]+lz],
            q, physicsClientId=cid)

def step_fw(ids, pos, dirn, target, dt, cid):
    diff = target - pos
    dist = math.hypot(diff[0], diff[1])
    if dist < FW_WP_RADIUS:
        place_fw(ids, pos, dirn, cid)
        return pos.copy(), dirn.copy(), True
    nd  = diff / (np.linalg.norm(diff) + 1e-9)
    np_ = pos + nd * FW_SPEED * dt
    np_[0] = np.clip(np_[0], -WORLD_LIMIT, WORLD_LIMIT)
    np_[1] = np.clip(np_[1], -WORLD_LIMIT, WORLD_LIMIT)
    np_[2] = target[2]
    place_fw(ids, np_, nd, cid)
    return np_, nd, False


# ======================================================================
# Quad software-leader navigation
# ======================================================================

def step_leader(ldr, wp, dt, speed=SQUAD_SPD):
    diff = wp[:2] - ldr[:2]
    dist = np.linalg.norm(diff)
    out  = ldr.copy()
    # Blend z smoothly toward target — never snap it in one step.
    # This prevents sudden altitude commands when the leader z is
    # seeded from a different altitude (e.g. QUAD_CAM_ALT → QUAD_CRUISE_ALT).
    dz        = wp[2] - ldr[2]
    out[2]    = ldr[2] + np.clip(dz, -speed * dt, speed * dt)
    if dist < ARRIVE_DIST:
        out[2] = wp[2]   # snap z only once we have arrived in XY
        return out, True
    nd      = diff / dist
    out[:2] = ldr[:2] + nd * min(speed * dt, dist)
    out[0]  = np.clip(out[0], -WORLD_LIMIT, WORLD_LIMIT)
    out[1]  = np.clip(out[1], -WORLD_LIMIT, WORLD_LIMIT)
    return out, False


# ======================================================================
# Viewport camera
# ======================================================================

def update_viewport(target, keys):
    c = _cam
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        c['free'] = not c['free']
        if c['free']:
            c['tx'], c['ty'], c['tz'] = target
    for k, attr, delta in [
        (p.B3G_LEFT_ARROW,  'yaw',   -2.0),
        (p.B3G_RIGHT_ARROW, 'yaw',    2.0),
        (p.B3G_UP_ARROW,    'pitch',  1.5),
        (p.B3G_DOWN_ARROW,  'pitch', -1.5),
        (ord('z'),          'dist',  -0.5),
        (ord('x'),          'dist',   0.5),
    ]:
        if k in keys and keys[k] & p.KEY_IS_DOWN:
            if attr == 'pitch':
                c[attr] = np.clip(c[attr]+delta, CAM_PITCH_MIN, CAM_PITCH_MAX)
            else:
                c[attr] += delta
    if c['free']:
        cy_ = math.cos(math.radians(c['yaw']))
        sy_ = math.sin(math.radians(c['yaw']))
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            c['tx'] += cy_; c['ty'] += sy_
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            c['tx'] -= cy_; c['ty'] -= sy_
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            c['tz'] += 0.5
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            c['tz'] -= 0.5
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        c.update(dist=25.0, yaw=45.0, pitch=-40.0, free=False)
    tgt = [c['tx'], c['ty'], c['tz']] if c['free'] else list(target)
    p.resetDebugVisualizerCamera(c['dist'], c['yaw'], c['pitch'], tgt)


# ======================================================================
# Report
# ======================================================================

def print_report(peaks, heatmap, total_frames, total_px,
                 photos_taken, assignments, battery=None, skipped=None):
    sep = "=" * 66
    print(f"\n{sep}")
    print("  MISSION COMPLETE — DETECTION REPORT")
    print(sep)
    print(f"  FW detection frames           : {total_frames}")
    print(f"  Cumulative white pixels       : {total_px}")
    print(f"  Heatmap peak cell value       : {heatmap.grid.max():.2f}")
    print(f"  Ground-truth cube sites       : {NUM_TARGETS}")
    print(f"  Heatmap hotspots detected     : {len(peaks)}")
    print(f"  Close-up photos saved         : {photos_taken}")
    print()

    # Per-quad assignment summary
    print("  Quad assignments:")
    for qi in range(NUM_DRONES):
        sites = assignments[qi] if assignments else []
        print(f"    Quad-{qi} ({QUAD_NAMES[qi]:6s}) — "
              f"{len(sites)} site(s) assigned")

    # Precision / recall
    true_pos   = 0
    false_pos  = 0
    matched_gt = set()

    if peaks:
        print("\n  Per-hotspot breakdown:")
        for i, (px, py) in enumerate(peaks):
            dists  = [math.hypot(px - gx, py - gy)
                      for gx, gy, _ in WASTE_POSITIONS]
            near_d = min(dists)
            near_i = int(np.argmin(dists))
            heat   = float(heatmap.grid[heatmap._wc(px, py)])
            # find which quad was assigned this peak
            owner  = next((qi for qi, ss in enumerate(assignments)
                           if (px, py) in ss), -1)
            owner_str = (f"Quad-{owner} ({QUAD_NAMES[owner]})"
                         if owner >= 0 else "unassigned")
            if near_d <= MATCH_RADIUS and near_i not in matched_gt:
                matched_gt.add(near_i)
                true_pos += 1
                verdict = f"TRUE  POS  (GT#{near_i+1} dist={near_d:.1f}m)"
            else:
                false_pos += 1
                verdict = f"FALSE POS  (nearest GT dist={near_d:.1f}m)"
            print(f"    Peak {i+1:2d}  ({px:6.1f},{py:6.1f})  "
                  f"heat={heat:6.1f}  owner={owner_str}  {verdict}")
    else:
        print("  No hotspots found.")

    missed    = NUM_TARGETS - len(matched_gt)
    precision = true_pos / max(len(peaks), 1) * 100
    recall    = true_pos / NUM_TARGETS * 100

    print()
    print(f"  True Positives  : {true_pos} / {NUM_TARGETS}")
    print(f"  False Positives : {false_pos}")
    print(f"  Missed (FN)     : {missed}")
    print(f"  Precision       : {precision:.1f}%")
    print(f"  Recall          : {recall:.1f}%")
    print(f"\n  Heatmap PNG     : {HEATMAP_PNG}")
    print(f"  Quad photos dir : {QUAD_FRAMES_DIR}/")

    if battery is not None:
        print()
        print("  Battery summary (Zeng et al. 2018 model):")
        for line in battery.summary():
            print(line)

    if skipped:
        print()
        sk = ", ".join(f"({x:.1f},{y:.1f})" for x, y in skipped)
        print(f"  Sites skipped (energy infeasible): {sk}")

    print(sep + "\n")


def banner(state, msg=""):
    print(f"\n{'='*60}\n  [{STATE_NAMES[state]}]  {msg}\n{'='*60}\n")


# ======================================================================
# MAIN
# ======================================================================

def run(
    physics  = DEFAULT_PHYSICS,
    gui      = DEFAULT_GUI,
    plot     = DEFAULT_PLOT,
    sim_hz   = DEFAULT_SIM_HZ,
    ctrl_hz  = DEFAULT_CTRL_HZ,
    out_dir  = DEFAULT_OUT_DIR,
):
    dt = 1.0 / ctrl_hz

    # Quads spawn at cruise altitude — stable physics from the start.
    # Battery is NOT drained during FW phases (update skipped below).
    init_xyzs = QUAD_HOVER.copy()

    env = CtrlAviary(
        drone_model    = DroneModel.CF2X,
        num_drones     = NUM_DRONES,
        initial_xyzs   = init_xyzs,
        initial_rpys   = np.zeros((NUM_DRONES, 3)),
        physics        = physics,
        pyb_freq       = sim_hz,
        ctrl_freq      = ctrl_hz,
        gui            = gui,
        user_debug_gui = False,
    )
    cid = env.CLIENT

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,          0, physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1, physicsClientId=cid)

    # Hide the default PyBullet ground plane — the ODM heightfield replaces it
    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0, 0, 0, 0],
                        physicsClientId=cid)

    # Drone colours — must recolor ALL links, not just base (linkIndex=-1).
    # The CF2X URDF has separate links for arms, motors and propellers which
    # default to light-grey and will flood the FW nadir camera with false
    # white-pixel detections when the quads hover below the FW.
    for i in range(NUM_DRONES):
        num_joints = p.getNumJoints(env.DRONE_IDS[i], physicsClientId=cid)
        for link_idx in range(-1, num_joints):
            try:
                p.changeVisualShape(env.DRONE_IDS[i], link_idx,
                                    rgbaColor=QUAD_COLORS[i],
                                    physicsClientId=cid)
            except Exception:
                pass

    # Terrain + obstacles + waste
    # ODM map: 128×128 heightfield + 512×512 ortho-texture.
    print("[World] Loading ODM terrain...")
    build_odm_terrain(
        heightmap_path = os.path.join(_HERE, "assets", "odm_heightmap_128.png"),
        texture_path   = os.path.join(_HERE, "assets", "odm_texture_512.png"),
        arena_m        = 34.0,
        max_height_m   = 0.25,   # subtle dune relief — keeps terrain below cube bases
        cid            = cid,
    )
    excl = list(WASTE_POSITIONS) + [
        (HOME_XY[0], HOME_XY[1], 0.0)] + [
        (x, y, 0.0) for x, y in QUAD_START_XY]
    spawn_random_obstacles(12, 0.3, 34.0, excl, 3.0, cid)

    for pos in WASTE_POSITIONS:
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.30]*3,
                                    physicsClientId=cid)
        vs = p.createVisualShape(   p.GEOM_BOX, halfExtents=[0.30]*3,
                                    rgbaColor=CUBE_COLOR, physicsClientId=cid)
        p.createMultiBody(0, cs, vs, basePosition=pos, physicsClientId=cid)
    print(f"[World] {NUM_TARGETS} WHITE waste cubes | {NUM_DRONES} quads")

    # Fixed-wing
    fw_ids = spawn_fw(cid)
    fw_pos = HOME_FW.copy().astype(float)
    fw_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    place_fw(fw_ids, fw_pos, fw_dir, cid)

    # Heatmap + detection counters
    heatmap      = WasteHeatmap()
    fw_cam_ctr   = 0
    total_frames = 0
    total_px     = 0

    # Mission variables
    mission_state  = FW_SCANNING
    scan_idx       = 0
    fw_land_step   = 0
    fw_land_max    = int(FW_LAND_SECS * ctrl_hz)
    quad_land_step = 0
    quad_land_max  = int(LAND_SECS * ctrl_hz)

    # Per-quad inspection state  (populated after FW lands)
    assignments    = [[] for _ in range(NUM_DRONES)]   # (wx,wy) lists
    skipped        = []                                 # energy-infeasible sites
    # Per-quad waypoints as numpy arrays [N,3]
    quad_wps       = [np.empty((0,3), dtype=float) for _ in range(NUM_DRONES)]
    quad_wp_idx    = [0] * NUM_DRONES        # current WP index per quad
    quad_arrived   = [False] * NUM_DRONES    # at current WP?
    quad_dwell     = [0]     * NUM_DRONES    # dwell counter per quad
    quad_done      = [False] * NUM_DRONES    # finished all assigned WPs?
    quad_photo_taken = [False] * NUM_DRONES  # photo fired for current site?
    dwell_max      = int(QUAD_DWELL_SECS * ctrl_hz)
    photos_taken   = 0

    # Software leaders (3-D position targets fed to PID)
    leaders = QUAD_HOVER.copy().astype(float)

    # Per-quad departure stagger: quad i waits i*TAKEOFF_STAGGER_SECS
    # before flying toward its first waypoint.  Until then it holds hover.
    stagger_steps = [int(i * TAKEOFF_STAGGER_SECS * ctrl_hz)
                     for i in range(NUM_DRONES)]
    squad_go_step = 0   # counts steps inside SQUAD_GO

    # Battery system (Zeng et al. 2018)
    battery    = BatterySystem(NUM_DRONES)
    prev_vel   = np.zeros((NUM_DRONES, 3), dtype=float)
    bat_levels = np.ones(NUM_DRONES, dtype=float)   # start at 100%

    # PID + logging
    ctrls      = [DSLPIDControl(drone_model=DroneModel.CF2X)
                  for _ in range(NUM_DRONES)]
    action     = np.zeros((NUM_DRONES, 4))
    logger     = Logger(logging_freq_hz=ctrl_hz, num_drones=NUM_DRONES,
                        output_folder=out_dir)
    target_pos = QUAD_HOVER.copy().astype(float)

    t0   = time.time()
    step = 0

    banner(FW_SCANNING,
           f"White-cube detection | {len(FW_SCAN_WPS)} lines | "
           f"{NUM_DRONES} quads hovering at base — battery paused until FW returns")
    print(f"[Fleet] Yellow FW=scout | "
          + " | ".join(f"Quad-{i}={QUAD_NAMES[i]}" for i in range(NUM_DRONES)))
    print("[Controls] Arrows:rotate | Z/X:zoom | F:free | R:reset | Ctrl+C:abort\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=cid)
            obs  = [env._getDroneStateVector(i) for i in range(NUM_DRONES)]

            # ── FW_SCANNING ──────────────────────────────────────────
            if mission_state == FW_SCANNING:

                fw_pos, fw_dir, reached = step_fw(
                    fw_ids, fw_pos, fw_dir, FW_SCAN_WPS[scan_idx], dt, cid)

                fw_cam_ctr += 1
                if fw_cam_ctr >= FW_CAM_SAMPLE:
                    fw_cam_ctr = 0
                    rgb = fw_camera_rgb(fw_pos, fw_dir, cid)
                    pts, n_px = detect_white_pts(rgb, fw_pos, fw_dir)

                    # Save frame so you can inspect what the FW actually sees
                    if _HAS_CV2 and FW_CAM_FRAMES_DIR:
                        os.makedirs(FW_CAM_FRAMES_DIR, exist_ok=True)
                        frame_path = os.path.join(
                            FW_CAM_FRAMES_DIR, f"fw_{total_frames:05d}.png")
                        cv2.imwrite(frame_path,
                                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                    if n_px >= MIN_WHITE_PX:
                        total_frames += 1
                        total_px     += n_px
                        heatmap.accumulate(pts)
                        if total_frames % 10 == 1:
                            print(f"  [FW-cam] ({fw_pos[0]:5.1f},{fw_pos[1]:5.1f}) "
                                  f"white_px={n_px:4d}  "
                                  f"heat_max={heatmap.grid.max():.1f}")
                    else:
                        # Always show a low-rate heartbeat so silence is obvious
                        if step % (FW_CAM_SAMPLE * 75) == 0:
                            print(f"  [FW-cam] ({fw_pos[0]:5.1f},{fw_pos[1]:5.1f}) "
                                  f"white_px={n_px} (below threshold)")

                if reached:
                    scan_idx += 1
                    print(f"  [FW] WP {scan_idx}/{len(FW_SCAN_WPS)} | "
                          f"heat_max={heatmap.grid.max():.1f}")
                    if scan_idx >= len(FW_SCAN_WPS):
                        mission_state = FW_RETURN
                        banner(FW_RETURN, "Scan complete — returning home")

                # Quads hold at base hover — battery not drained yet
                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

            # ── FW_RETURN ────────────────────────────────────────────
            elif mission_state == FW_RETURN:

                fw_pos, fw_dir, reached = step_fw(
                    fw_ids, fw_pos, fw_dir, HOME_FW, dt, cid)
                if reached:
                    mission_state = FW_LAND
                    fw_land_step  = 0
                    banner(FW_LAND, "Fixed-wing landing — analysing heatmap...")

                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

            # ── FW_LAND ──────────────────────────────────────────────
            elif mission_state == FW_LAND:

                alpha     = min(1.0, fw_land_step / max(1, fw_land_max))
                fw_pos[2] = FW_ALT * (1.0 - alpha) + 0.15 * alpha
                place_fw(fw_ids, fw_pos, fw_dir, cid)
                fw_land_step += 1

                for i in range(NUM_DRONES):
                    target_pos[i] = QUAD_HOVER[i].copy()

                if fw_land_step >= fw_land_max:
                    peaks = heatmap.find_peaks()
                    heatmap.visualize(cid)
                    heatmap.mark_peaks(peaks, cid)
                    heatmap.save_png()

                    assignments, skipped = assign_peaks_to_quads(peaks, battery)
                    heatmap.mark_assignment(assignments, cid)

                    # Build per-quad WP arrays
                    for qi in range(NUM_DRONES):
                        sites = assignments[qi]
                        if sites:
                            quad_wps[qi] = np.array(
                                [[x, y, QUAD_CAM_ALT] for x, y in sites],
                                dtype=float)
                        else:
                            quad_wps[qi] = np.empty((0, 3), dtype=float)
                        quad_wp_idx[qi]      = 0
                        quad_arrived[qi]     = False
                        quad_dwell[qi]       = 0
                        quad_photo_taken[qi] = False
                        quad_done[qi]        = len(quad_wps[qi]) == 0

                    # Leaders at current hover positions
                    for i in range(NUM_DRONES):
                        leaders[i] = QUAD_HOVER[i].copy()

                    squad_go_step = 0
                    mission_state = SQUAD_GO
                    total_assigned = sum(len(a) for a in assignments)
                    banner(SQUAD_GO,
                           f"{NUM_DRONES} quads | {total_assigned} site(s) | "
                           f"staggered departure every {TAKEOFF_STAGGER_SECS:.0f}s")

            # ── SQUAD_GO ─────────────────────────────────────────────
            elif mission_state == SQUAD_GO:

                squad_go_step += 1

                for qi in range(NUM_DRONES):

                    # ── stagger: hold at hover until this quad's slot opens
                    if squad_go_step < stagger_steps[qi]:
                        leaders[qi]    = QUAD_HOVER[qi].copy()
                        target_pos[qi] = QUAD_HOVER[qi].copy()
                        continue

                    # ── already done or RTB — hold at current leader pos
                    if quad_done[qi]:
                        pass   # leader unchanged; PID keeps drone in place

                    # ── finished all assigned waypoints
                    elif quad_wp_idx[qi] >= len(quad_wps[qi]):
                        quad_done[qi] = True
                        print(f"  [Quad-{qi}/{QUAD_NAMES[qi]}] All sites done")

                    # ── flying toward current waypoint
                    elif not quad_arrived[qi]:
                        wp = quad_wps[qi][quad_wp_idx[qi]]
                        leaders[qi], here = step_leader(leaders[qi], wp, dt)
                        if here:
                            quad_arrived[qi]    = True
                            quad_dwell[qi]      = 0
                            quad_photo_taken[qi] = False
                            wx, wy = assignments[qi][quad_wp_idx[qi]]
                            print(f"  [Quad-{qi}/{QUAD_NAMES[qi]}] Overhead "
                                  f"({wx:.1f},{wy:.1f}) — hovering…")

                    # ── dwelling overhead: hold XY, count steps, shoot once
                    else:
                        leaders[qi][2]  = QUAD_CAM_ALT   # lock altitude
                        quad_dwell[qi] += 1

                        # Fire photo exactly once at the midpoint of the dwell.
                        # We reset t0 right after the capture so the sync()
                        # call doesn't try to race to catch up lost real-time.
                        if (not quad_photo_taken[qi]
                                and quad_dwell[qi] >= dwell_max // 2):
                            wx, wy = assignments[qi][quad_wp_idx[qi]]
                            site_n = quad_wp_idx[qi] + 1
                            rgb    = quad_camera_rgb(obs[qi][0:3], cid)
                            save_quad_photo(rgb, QUAD_NAMES[qi],
                                            qi * 100 + site_n)
                            photos_taken        += 1
                            quad_photo_taken[qi] = True
                            t0 = time.time() - step * dt   # re-anchor sync clock
                            print(f"  [Quad-{qi}/{QUAD_NAMES[qi]}] Photo @ "
                                  f"({wx:.1f},{wy:.1f}) site {site_n}")

                        # Advance to next site once dwell is complete
                        if quad_dwell[qi] >= dwell_max:
                            quad_wp_idx[qi]     += 1
                            quad_arrived[qi]     = False
                            quad_dwell[qi]       = 0
                            quad_photo_taken[qi] = False

                # Push updated leader positions to PID targets
                for i in range(NUM_DRONES):
                    target_pos[i] = leaders[i].copy()

                # Transition when ALL quads are done
                if all(quad_done):
                    mission_state = SQUAD_BACK
                    banner(SQUAD_BACK, "All quads finished — returning home")
                    for i in range(NUM_DRONES):
                        # Seed XY from actual drone position but force z to
                        # cruise altitude — never trust obs[z] which could be
                        # anywhere between QUAD_CAM_ALT and hover due to PID lag.
                        leaders[i] = np.array(
                            [obs[i][0], obs[i][1], QUAD_CRUISE_ALT])

            # ── SQUAD_BACK ───────────────────────────────────────────
            elif mission_state == SQUAD_BACK:

                all_home = True
                for i in range(NUM_DRONES):
                    leaders[i], home = step_leader(
                        leaders[i], QUAD_HOVER[i], dt)
                    if not home:
                        all_home = False
                    target_pos[i] = leaders[i].copy()

                if all_home:
                    mission_state  = QUAD_LAND
                    quad_land_step = 0
                    banner(QUAD_LAND, "All quads descending")

            # ── QUAD_LAND ────────────────────────────────────────────
            elif mission_state == QUAD_LAND:

                alpha = min(1.0, quad_land_step / max(1, quad_land_max))
                z_now = QUAD_CRUISE_ALT * (1.0 - alpha) + 0.15 * alpha
                for i in range(NUM_DRONES):
                    target_pos[i] = np.array(
                        [QUAD_HOVER[i][0], QUAD_HOVER[i][1], z_now])
                quad_land_step += 1

                if quad_land_step >= quad_land_max:
                    peaks = heatmap.find_peaks()
                    print_report(peaks, heatmap, total_frames,
                                 total_px, photos_taken, assignments, battery, skipped)
                    mission_state = DONE
                    break

            elif mission_state == DONE:
                break

            # ── Always refresh FW visual ──────────────────────────────
            place_fw(fw_ids, fw_pos, fw_dir, cid)

            # ── Quad-to-quad collision avoidance ─────────────────────
            for i in range(NUM_DRONES):
                push = np.zeros(2)
                for j in range(NUM_DRONES):
                    if j == i: continue
                    dx = obs[i][0] - obs[j][0]
                    dy = obs[i][1] - obs[j][1]
                    d  = math.hypot(dx, dy)
                    if 1e-6 < d < MIN_SEP:
                        push += AVOID_GAIN * (MIN_SEP-d) / d * np.array([dx, dy])
                n_ = np.linalg.norm(push)
                if n_ > MAX_PUSH: push *= MAX_PUSH / n_
                target_pos[i, 0:2] = np.clip(
                    target_pos[i, 0:2] + push, -WORLD_LIMIT, WORLD_LIMIT)
                target_pos[i, 2]   = np.clip(
                    target_pos[i, 2], Z_MIN, Z_MAX)

            # ── Battery: drain every step (hovering at base costs energy too) ─
            # The energy planner reads battery.energy_j after FW lands, so it
            # automatically works with whatever is left after base hovering.
            bat_levels, _power = battery.update(obs, prev_vel, dt)
            battery.update_hud(obs, bat_levels, cid)

            # RTB checks only during active inspection
            if mission_state == SQUAD_GO:
                for i in range(NUM_DRONES):
                    battery.check_rtb(i, bat_levels[i], quad_done, leaders, obs)

            # ── PID ───────────────────────────────────────────────────
            for i, ctrl in enumerate(ctrls):
                raw = ctrl.computeControlFromState(
                    control_timestep = dt,
                    state            = obs[i],
                    target_pos       = target_pos[i],
                    target_rpy       = np.zeros(3),
                    target_vel       = np.zeros(3),
                    target_rpy_rates = np.zeros(3),
                )
                action[i, :] = raw[0]
                logger.log(drone=i, timestamp=step*dt, state=obs[i])

            env.step(action)

            # ── Viewport: FW during scan, Quad-0 from takeoff onwards ──
            cam_tgt = (fw_pos.tolist()
                       if mission_state in (FW_SCANNING, FW_RETURN, FW_LAND)
                       else list(obs[0][0:3]))
            update_viewport(cam_tgt, keys)
            env.render()
            if gui: sync(step, t0, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        print("\n[Abort] Ctrl+C — partial report:")
        peaks = heatmap.find_peaks()
        heatmap.save_png()
        print_report(peaks, heatmap, total_frames, total_px,
                     photos_taken, assignments, battery, skipped)

    finally:
        battery.cleanup_hud(cid)
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
    run(gui=args.gui, plot=args.plot)