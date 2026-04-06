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

Start positions (arc along west edge, facing the field):
  Quad 0 (Blue)   [-18, -18]   south end of arc
  Quad 1 (Green)  [-18,  -8]   south-centre
  Quad 2 (Red)    [-18,  +2]   north-centre
  Quad 3 (Purple) [-18, +12]   north end of arc

Controls:
  Arrows: rotate   Z/X: zoom   F: free-cam   R: reset   Ctrl+C: abort
"""

import os
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
# Mission states
# ======================================================================
FW_SCANNING = 0
FW_RETURN   = 1
FW_LAND     = 2
SQUAD_GO    = 3
SQUAD_BACK  = 4
QUAD_LAND   = 5
DONE        = 6
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

# ── 4 quad start positions — arc along the south-west baseline ─────
#   Spread evenly from y=-18 to y=+2 along x=-18, facing into the field.
#   This gives each quad a different "sector" of the arena to look across.
QUAD_START_XY = np.array([
    [-18.0, -18.0],   # 0 Blue   — far-south corner
    [-18.0,  -8.0],   # 1 Green  — south-centre
    [-18.0,   2.0],   # 2 Red    — north-centre
    [-18.0,  12.0],   # 3 Purple — far-north
])
QUAD_HOVER = np.array(
    [[x, y, QUAD_CRUISE_ALT] for x, y in QUAD_START_XY], dtype=float)

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
    (-15.0, -12.0, 0.0), (-15.0,  12.0, 0.0),
    (  0.0, -15.0, 0.0), (  0.0,  15.0, 0.0),
    ( 15.0, -12.0, 0.0), ( 15.0,  12.0, 0.0),
    ( -8.0,   0.0, 0.0), (  8.0,   0.0, 0.0),
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
DEFAULT_OUT_DIR = "results"

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
FW_CAM_SAMPLE = 4

WHITE_LO      = np.array([  0,  0, 210], dtype=np.uint8)
WHITE_HI      = np.array([179, 55, 255], dtype=np.uint8)
MIN_WHITE_PX  = 4
MAX_PROJ_SAMP = 200

# ======================================================================
# Quad inspection camera
# ======================================================================
QUAD_CAM_W      = 1280
QUAD_CAM_H      = 960
QUAD_CAM_FOV    = 75.0       # ~9 m footprint at 5 m alt
QUAD_CAM_ALT    = 5.0        # overhead hover altitude (m)
QUAD_DWELL_SECS = 2.5        # hover duration before shooting
QUAD_FRAMES_DIR = "quad_frames"

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
        """
        Update all drone batteries for one control step.

        Parameters
        ----------
        obs      : list of state vectors (length n)
        prev_vel : np.ndarray (n, 3) — velocities from the previous step
        dt       : float — control timestep in seconds

        Returns
        -------
        levels : np.ndarray (n,) — battery fractions 0..1
        power  : np.ndarray (n,) — instantaneous power in Watts
        """
        power = np.zeros(self.n)
        for i in range(self.n):
            vel   = np.array(obs[i][10:13], dtype=float)
            accel = (vel - prev_vel[i]) / dt if dt > 0 else np.zeros(3)
            p     = self._power(vel, accel)
            power[i] = p
            self.energy_j[i] = max(0.0, self.energy_j[i] - p * dt)
            prev_vel[i] = vel   # in-place update for caller

        levels = self.energy_j / BATTERY_CAPACITY_J
        return levels, power

    # ── RTB / emergency checks ────────────────────────────────────────

    def check_rtb(self, i, level, quad_done, leaders, obs):
        """
        Check drone i's battery and act if below threshold.
        Modifies quad_done[i] and leaders[i] in place.
        Returns a status string for logging.
        """
        if self.emergency[i]:
            # Already in emergency land — keep descending
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
            # Steer leader back to home hover position
            home   = QUAD_HOVER[i]
            diff   = home[:2] - leaders[i][:2]
            dist   = np.linalg.norm(diff)
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

            if self._text_ids[i] != -1:
                try:
                    p.removeUserDebugItem(self._text_ids[i], physicsClientId=cid)
                except Exception:
                    pass

            self._text_ids[i] = p.addUserDebugText(
                text=label, textPosition=pos,
                textColorRGB=col, textSize=1.1,
                lifeTime=0, physicsClientId=cid)

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
                r = float(np.clip(v * 2.0,       0.0, 1.0))
                g = float(np.clip(v * 2.0 - 1.0, 0.0, 1.0))
                b = float(np.clip(v * 4.0 - 3.0, 0.0, 1.0))
                vs  = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[GRID_CELL*0.46, GRID_CELL*0.46, 0.01],
                    rgbaColor=[r, g, b, 0.6], physicsClientId=cid)
                p.createMultiBody(0, -1, vs, [wx, wy, 0.05], physicsClientId=cid)
                count += 1
        self.viz_ids = [count]
        print(f"  [Heatmap] {count} tiles rendered")

    def mark_peaks(self, peaks, cid):
        for wx, wy in peaks:
            vs = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.75, length=0.06,
                rgbaColor=[1.0, 1.0, 0.0, 1.0], physicsClientId=cid)
            p.createMultiBody(0, -1, vs, [wx, wy, 0.18], physicsClientId=cid)

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
                p.createMultiBody(0, -1, vs, [wx, wy, 0.55], physicsClientId=cid)

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

def assign_peaks_to_quads(peaks):
    """
    Distribute hotspots across ALL 4 quads so every drone gets work.

    Strategy — balanced round-robin by travel distance:
      1. Sort peaks by their angle from the arc baseline so sites are
         handed out left-to-right (south-to-north), matching the quad
         layout along the arc.
      2. Assign peak[0] -> Quad-0, peak[1] -> Quad-1, ... cycling mod 4.
         This guarantees every quad gets floor(N/4) or ceil(N/4) sites
         regardless of how many peaks were found.
      3. Within each quad the sites are ordered nearest-first so the
         quad takes the shortest path through its assigned sites.
    """
    if not peaks:
        return [[] for _ in range(NUM_DRONES)]

    # Sort peaks south-to-north (ascending Y) so the round-robin
    # naturally hands adjacent sites to different quads rather than
    # clustering all southern sites on one quad.
    sorted_peaks = sorted(peaks, key=lambda p: p[1])

    assignments = [[] for _ in range(NUM_DRONES)]
    for rank, (wx, wy) in enumerate(sorted_peaks):
        qi = rank % NUM_DRONES
        assignments[qi].append((wx, wy))

    # Within each quad, sort assigned sites by distance from its start
    # position so it travels the shortest sequential path.
    for qi in range(NUM_DRONES):
        if len(assignments[qi]) <= 1:
            continue
        sx, sy = QUAD_START_XY[qi]
        ordered, remaining = [], list(assignments[qi])
        cx, cy = sx, sy
        while remaining:
            nearest_i = int(np.argmin(
                [math.hypot(wx - cx, wy - cy) for wx, wy in remaining]))
            ordered.append(remaining.pop(nearest_i))
            cx, cy = ordered[-1]
        assignments[qi] = ordered

    print("\n  [Assignment] Round-robin hotspot allocation:")
    for qi in range(NUM_DRONES):
        sites_str = ", ".join(f"({x:.1f},{y:.1f})" for x, y in assignments[qi])
        print(f"    Quad-{qi} ({QUAD_NAMES[qi]:6s}) <- "
              f"{len(assignments[qi])} site(s)  {sites_str or '(none)'}")
    print()
    return assignments


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
    if dist < ARRIVE_DIST:
        out    = ldr.copy()
        out[2] = wp[2]
        return out, True
    nd  = diff / dist
    out = ldr.copy()
    out[:2] = ldr[:2] + nd * min(speed * dt, dist)
    out[0]  = np.clip(out[0], -WORLD_LIMIT, WORLD_LIMIT)
    out[1]  = np.clip(out[1], -WORLD_LIMIT, WORLD_LIMIT)
    out[2]  = wp[2]
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
                 photos_taken, assignments, battery=None):
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

    # Quads spawn at their diamond hover spots
    init_xyzs = np.array(
        [[x, y, QUAD_CRUISE_ALT] for x, y in QUAD_START_XY], dtype=float)

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

    # Sand floor
    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0,0,0,0], physicsClientId=cid)
    fv = p.createVisualShape(p.GEOM_BOX, halfExtents=[25,25,0.01],
                             rgbaColor=[0.87,0.72,0.53,1.0], physicsClientId=cid)
    p.createMultiBody(0, -1, fv, [0,0,-0.05], physicsClientId=cid)

    # Drone colours
    for i in range(NUM_DRONES):
        try:
            p.changeVisualShape(env.DRONE_IDS[i], -1,
                                rgbaColor=QUAD_COLORS[i], physicsClientId=cid)
        except Exception:
            pass

    # Terrain + obstacles + waste
    print("[World] Building terrain...")
    create_desert_terrain(
        "assets/terrain_desert_dunes.png",
        "assets/desert_sand.png",
        (0.15, 0.15, 1.5))
    excl = list(WASTE_POSITIONS) + [
        (HOME_XY[0], HOME_XY[1], 0.0)] + [
        (x, y, 0.0) for x, y in QUAD_START_XY]
    spawn_desert_obstacles(12, 6, 35.0, excl, 5.0)

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
    # Per-quad waypoints as numpy arrays [N,3]
    quad_wps       = [np.empty((0,3), dtype=float) for _ in range(NUM_DRONES)]
    quad_wp_idx    = [0] * NUM_DRONES        # current WP index per quad
    quad_arrived   = [False] * NUM_DRONES    # at current WP?
    quad_dwell     = [0]     * NUM_DRONES    # dwell counter per quad
    quad_done      = [False] * NUM_DRONES    # finished all assigned WPs?
    dwell_max      = int(QUAD_DWELL_SECS * ctrl_hz)
    photos_taken   = 0

    # Software leaders (3-D position targets fed to PID)
    leaders = init_xyzs.copy()

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
    target_pos = init_xyzs.copy()

    t0   = time.time()
    step = 0

    banner(FW_SCANNING,
           f"White-cube detection | {len(FW_SCAN_WPS)} lines | "
           f"{NUM_DRONES} quads hover at diamond positions")
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
                    if n_px >= MIN_WHITE_PX:
                        total_frames += 1
                        total_px     += n_px
                        heatmap.accumulate(pts)
                        if total_frames % 10 == 1:
                            print(f"  [FW-cam] ({fw_pos[0]:5.1f},{fw_pos[1]:5.1f}) "
                                  f"white_px={n_px:4d}  "
                                  f"heat_max={heatmap.grid.max():.1f}")

                if reached:
                    scan_idx += 1
                    print(f"  [FW] WP {scan_idx}/{len(FW_SCAN_WPS)} | "
                          f"heat_max={heatmap.grid.max():.1f}")
                    if scan_idx >= len(FW_SCAN_WPS):
                        mission_state = FW_RETURN
                        banner(FW_RETURN, "Scan complete — returning home")

                # All quads hold at their hover spots
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

                    assignments = assign_peaks_to_quads(peaks)
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
                        quad_wp_idx[qi]  = 0
                        quad_arrived[qi] = False
                        quad_dwell[qi]   = 0
                        quad_done[qi]    = len(quad_wps[qi]) == 0

                    # Reset leaders to current hover positions
                    for i in range(NUM_DRONES):
                        leaders[i] = QUAD_HOVER[i].copy()

                    mission_state = SQUAD_GO
                    total_assigned = sum(len(a) for a in assignments)
                    banner(SQUAD_GO,
                           f"{NUM_DRONES} quads | {total_assigned} site(s) total | "
                           f"working in parallel")

            # ── SQUAD_GO ─────────────────────────────────────────────
            elif mission_state == SQUAD_GO:

                for qi in range(NUM_DRONES):
                    if quad_done[qi]:
                        # Finished — hold at last position or hover
                        pass

                    elif quad_wp_idx[qi] >= len(quad_wps[qi]):
                        quad_done[qi] = True
                        print(f"  [Quad-{qi}/{QUAD_NAMES[qi]}] All sites done")

                    elif not quad_arrived[qi]:
                        # ── fly to current WP ──
                        wp = quad_wps[qi][quad_wp_idx[qi]]
                        leaders[qi], here = step_leader(leaders[qi], wp, dt)
                        if here:
                            quad_arrived[qi] = True
                            quad_dwell[qi]   = 0
                            wx, wy = assignments[qi][quad_wp_idx[qi]]
                            print(f"  [Quad-{qi}/{QUAD_NAMES[qi]}] Overhead "
                                  f"({wx:.1f},{wy:.1f}) — hovering…")

                    else:
                        # ── dwell: lock altitude, count ──
                        leaders[qi][2] = QUAD_CAM_ALT
                        quad_dwell[qi] += 1

                        # Shoot at midpoint of dwell window
                        if quad_dwell[qi] == dwell_max // 2:
                            wx, wy = assignments[qi][quad_wp_idx[qi]]
                            site_n = quad_wp_idx[qi] + 1
                            rgb    = quad_camera_rgb(obs[qi][0:3], cid)
                            save_quad_photo(rgb, QUAD_NAMES[qi],
                                            # unique filename: qi_site
                                            int(f"{qi}{site_n}"))
                            photos_taken += 1
                            print(f"  [Quad-{qi}/{QUAD_NAMES[qi]}] Photo @ "
                                  f"({wx:.1f},{wy:.1f}) site {site_n}")

                        if quad_dwell[qi] >= dwell_max:
                            quad_wp_idx[qi] += 1
                            quad_arrived[qi] = False
                            quad_dwell[qi]   = 0

                # Update target positions for all quads
                for i in range(NUM_DRONES):
                    target_pos[i] = leaders[i].copy()

                # Transition when ALL quads are done
                if all(quad_done):
                    mission_state = SQUAD_BACK
                    banner(SQUAD_BACK, "All quads finished — returning home")
                    for i in range(NUM_DRONES):
                        leaders[i] = np.array(
                            [obs[i][0], obs[i][1], obs[i][2]])

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
                                 total_px, photos_taken, assignments, battery)
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

            # ── Battery: update energy, HUD, check RTB ───────────────
            bat_levels, _power = battery.update(obs, prev_vel, dt)
            battery.update_hud(obs, bat_levels, cid)

            # Check RTB / emergency per quad (only during SQUAD_GO so
            # the mission state machine stays consistent in other phases)
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

            # ── Viewport: FW during scan, Quad-0 (blue) after ─────────
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
                     photos_taken, assignments, battery)

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