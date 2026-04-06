"""
Save the Planet: Mixed UAV Fleet Waste Detection Mission
4-QUAD SWARM + CAMERA HEATMAP + NEAREST-SITE ASSIGNMENT

MISSION FLOW:
  1. FW_SCANNING  Fixed-wing lawnmower grid. Nadir camera detects plastic
                  waste via Roboflow when available (else HSV cyan fallback).
                  Optional: synthetic URDF props on terrain (visible in 3D) and/or
                  synthetic_overlay on FW bitmap (2D). Detections → 34x34 heatmap.
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
import random
import numpy as np
import pybullet as p
import matplotlib
import json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linear_sum_assignment

# All asset paths are resolved relative to this script file, so the sim
# works regardless of which directory you launch it from.
_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

# Roboflow for plastic-waste detection (optional; falls back to HSV cyan)
_ROBOFLOW_MODEL = None
try:
    from roboflow import Roboflow

    _rf_api_key = os.environ.get("ROBOFLOW_API_KEY", "BUjYN0bQO9kXyQIaWk88")
    if not _rf_api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set")
    rf = Roboflow(api_key=_rf_api_key)
    _ROBOFLOW_MODEL = (
        rf.workspace("myworkspace-08tvw")
        .project("save-the-planet_duplicate")
        .version(7)
        .model
    )
    print("[Roboflow] Plastic waste model loaded (save-the-planet_duplicate v4)")
except Exception as e:
    print(f"[Roboflow] Not available ({e}) — using HSV cyan detection for waste")
    _ROBOFLOW_MODEL = None

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Synthetic 2D overlay — guarantees waste appears in FW camera frames even
# when 3D URDFs are too small for PyBullet's tiny renderer at high altitude.
from synthetic_overlay import SyntheticOverlay
from drone_comms import DroneCommsLayer

synth_3d = True

TERRAIN_NO_DATA_RGB = np.array([168, 156, 142], dtype=np.float32)
TERRAIN_NO_DATA_THRESH = 20.0


def prepare_terrain_texture(texture_path):
    """Fill no-data side patches so terrain edges blend with the map."""
    if not _HAS_CV2 or not os.path.isfile(texture_path):
        return texture_path

    img = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    if img is None:
        return texture_path

    nd_bgr = TERRAIN_NO_DATA_RGB[::-1]
    dist = np.linalg.norm(img.astype(np.float32) - nd_bgr.reshape(1, 1, 3), axis=2)
    mask = (dist < TERRAIN_NO_DATA_THRESH).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return texture_path

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    fixed = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    out_path = os.path.join(
        os.path.dirname(texture_path), "terrain_texture_autofill.png"
    )
    try:
        cv2.imwrite(out_path, fixed)
        print("[Terrain] Auto-filled side no-data patches in texture.")
        return out_path
    except Exception:
        return texture_path


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
    hm_img = _PIL.open(heightmap_path).convert("L")
    rows = hm_img.height
    cols = hm_img.width
    hm_arr = _np.array(hm_img, dtype=_np.float32) / 255.0 * max_height_m

    # PyBullet GEOM_HEIGHTFIELD expects the data in row-major order
    # with (0,0) at the bottom-left corner → flip vertically
    hm_flat = _np.flipud(hm_arr).flatten().tolist()

    # meshScale: metres per cell in X and Y, Z kept at 1 (data already in m)
    cell = arena_m / max(rows, cols)
    shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD,
        meshScale=[cell, cell, 1.0],
        heightfieldData=hm_flat,
        numHeightfieldRows=rows,
        numHeightfieldColumns=cols,
        physicsClientId=cid,
    )

    # Centre the heightfield at the world origin.
    # PyBullet places the geometric centre of the heightfield at basePosition.
    # We want the minimum of the terrain to sit at z=0, so shift down by
    # half the height range (hm_arr.max() / 2 when min≈0).
    half_h = float(hm_arr.max()) / 2.0
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=shape,
        baseVisualShapeIndex=shape,
        basePosition=[0.0, 0.0, half_h],
        physicsClientId=cid,
    )

    # ── texture ──────────────────────────────────────────────────────
    try:
        tex_path = prepare_terrain_texture(texture_path)
        tex_id = p.loadTexture(tex_path, physicsClientId=cid)
        p.changeVisualShape(
            body,
            -1,
            textureUniqueId=tex_id,
            rgbaColor=[1, 1, 1, 1],
            physicsClientId=cid,
        )
        print(
            f"[Terrain] ODM map loaded — "
            f"{cols}×{rows}px  cell={cell:.3f}m  "
            f"h=0..{max_height_m}m  tex={tex_id}"
        )
    except Exception as e:
        print(f"[Terrain] Texture load failed ({e}) — terrain will be grey")

    return body


def spawn_random_obstacles(n, min_r, arena_m, exclusions, min_sep, cid):
    """Scatter small rock/shrub stand-ins that avoid waste sites and base."""
    import random, math as _math

    lim = arena_m / 2.0
    placed = list(exclusions)
    for _ in range(n * 10):  # attempts
        if len(placed) - len(exclusions) >= n:
            break
        x = random.uniform(-lim + 1, lim - 1)
        y = random.uniform(-lim + 1, lim - 1)
        if any(_math.hypot(x - ex[0], y - ex[1]) < min_sep for ex in placed):
            continue
        h = random.uniform(0.2, 0.6)
        r = random.uniform(0.15, 0.35)
        vs = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=r,
            length=h,
            rgbaColor=[0.25, 0.20, 0.10, 1.0],
            physicsClientId=cid,
        )
        p.createMultiBody(0, -1, vs, basePosition=[x, y, h / 2], physicsClientId=cid)
        placed.append((x, y, 0.0))


# ======================================================================
# Synthetic waste URDF spawning (3D only, no capture/overlay pipeline)
# Mirrors synthetic_waste_capture.py placement style.
# ======================================================================
SYNTH_WASTE_COUNT = 10
NO_DATA_RGB = np.array([168, 156, 142], dtype=np.float32)
NO_DATA_THRESH = 18.0
MAX_SPAWN_TRIES = 50
MIN_WASTE_GAP = 3.5

WASTE_URDFS = [
    {
        "name": "bottle",
        "path": os.path.join(_HERE, "bottle.urdf"),
        "half_height": 0.085,
        "footprint": 0.04,
        "burial_bias": ["surface", "half_buried", "mostly_buried", "side_exposed"],
        "scale_range": (0.85, 1.15),
    },
    {
        "name": "cardboard_bag",
        "path": os.path.join(_HERE, "cardboard_bag.urdf"),
        "half_height": 0.10,
        "footprint": 0.22,
        "burial_bias": ["surface", "surface", "half_buried", "top_only"],
        "scale_range": (1.35, 1.75),
    },
    {
        "name": "cardboard_box",
        "path": os.path.join(_HERE, "cardboard_box.urdf"),
        "half_height": 0.09,
        "footprint": 0.24,
        "burial_bias": ["surface", "surface", "half_buried", "corner_peek", "top_only"],
        "scale_range": (1.35, 1.75),
    },
    {
        "name": "garbage_bag",
        "path": os.path.join(_HERE, "garbage_bag.urdf"),
        "half_height": 0.18,
        "footprint": 0.24,
        "burial_bias": ["surface", "surface", "half_buried"],
        "scale_range": (1.00, 1.30),
    },
]


def world_to_tex_uv(x, y, aabb_min, aabb_max, tex_w, tex_h):
    u = (x - aabb_min[0]) / (aabb_max[0] - aabb_min[0])
    v = (y - aabb_min[1]) / (aabb_max[1] - aabb_min[1])
    px = int(np.clip(u * (tex_w - 1), 0, tex_w - 1))
    py = int(np.clip((1.0 - v) * (tex_h - 1), 0, tex_h - 1))
    return px, py


def texture_ok_for_spawn(x, y, tex_rgb, aabb_min, aabb_max):
    if tex_rgb is None:
        return True
    th, tw = tex_rgb.shape[:2]
    px, py = world_to_tex_uv(x, y, aabb_min, aabb_max, tw, th)
    rgb = tex_rgb[py, px].astype(np.float32)
    if np.linalg.norm(rgb - NO_DATA_RGB) < NO_DATA_THRESH:
        return False
    x0 = max(0, px - 8)
    x1 = min(tw, px + 9)
    y0 = max(0, py - 8)
    y1 = min(th, py + 9)
    if tex_rgb[y0:y1, x0:x1].astype(np.float32).std() < 3.0:
        return False
    return True


def terrain_hit_z(x, y, terrain_id, cid):
    hit = p.rayTest([x, y, 200], [x, y, -200], physicsClientId=cid)[0]
    return hit[3][2] if hit[0] == terrain_id else None


def far_from_existing(x, y, existing_xy, min_gap):
    for ex, ey in existing_xy:
        if (x - ex) ** 2 + (y - ey) ** 2 < min_gap**2:
            return False
    return True


def placement_pose(kind, mode, ground_z):
    yaw = random.uniform(-math.pi, math.pi)
    roll = 0.0
    pitch = 0.0
    hh = kind["half_height"]
    if mode == "surface":
        z = ground_z + hh + random.uniform(0.005, 0.02)
    elif mode == "half_buried":
        z = ground_z + hh * random.uniform(0.35, 0.60)
        if kind["name"] in ["bottle", "cardboard_box"]:
            roll = random.uniform(-0.45, 0.45)
            pitch = random.uniform(-0.45, 0.45)
        else:
            roll = random.uniform(-0.20, 0.20)
            pitch = random.uniform(-0.20, 0.20)
    elif mode == "mostly_buried":
        z = ground_z + hh * random.uniform(0.08, 0.25)
        roll = random.uniform(-0.30, 0.30)
        pitch = random.uniform(-0.30, 0.30)
    elif mode == "top_only":
        z = ground_z + hh * random.uniform(0.05, 0.15)
        roll = random.uniform(-0.20, 0.20)
        pitch = random.uniform(-0.20, 0.20)
    elif mode == "corner_peek":
        z = ground_z + hh * random.uniform(0.15, 0.30)
        roll = random.choice([-1, 1]) * random.uniform(0.6, 1.1)
        pitch = random.choice([-1, 1]) * random.uniform(0.2, 0.7)
    elif mode == "side_exposed":
        z = ground_z + hh * random.uniform(0.15, 0.35)
        roll = random.choice([-1, 1]) * random.uniform(1.15, 1.45)
        pitch = random.uniform(-0.25, 0.25)
    else:
        z = ground_z + hh
    quat = p.getQuaternionFromEuler([roll, pitch, yaw])
    return z, quat


def load_texture_rgb(texture_path):
    if not os.path.isfile(texture_path):
        return None
    try:
        from PIL import Image as _PIL

        return np.array(_PIL.open(texture_path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def spawn_synthetic_waste_urdfs(terrain_id, tex_rgb, aabb_min, aabb_max, cid):
    xmin, ymin = aabb_min[0] + 2.5, aabb_min[1] + 2.5
    xmax, ymax = aabb_max[0] - 2.5, aabb_max[1] - 2.5
    cluster_seed = None
    existing_xy = []
    waste_items = []
    failed = 0

    valid_urdfs = []
    for kind in WASTE_URDFS:
        if os.path.isfile(kind["path"]):
            valid_urdfs.append(kind)
        else:
            print(f"[Waste] Missing URDF: {kind['path']}")
    if not valid_urdfs:
        print("[Waste] No valid URDF files found. Skipping synthetic waste spawn.")
        return []

    planned_kinds = []
    if len(valid_urdfs) <= SYNTH_WASTE_COUNT:
        planned_kinds.extend(valid_urdfs)  # guarantee at least one of each class
    while len(planned_kinds) < SYNTH_WASTE_COUNT:
        planned_kinds.append(random.choice(valid_urdfs))
    random.shuffle(planned_kinds)

    for planned_kind in planned_kinds:
        spawned = False
        for _try in range(MAX_SPAWN_TRIES):
            kind = planned_kind
            if cluster_seed and random.random() < 0.30:
                x = cluster_seed[0] + random.uniform(-6.0, 6.0)
                y = cluster_seed[1] + random.uniform(-6.0, 6.0)
            else:
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)

            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue

            min_gap = max(MIN_WASTE_GAP, kind["footprint"] * 1.6)
            if not far_from_existing(x, y, existing_xy, min_gap):
                continue
            if not texture_ok_for_spawn(x, y, tex_rgb, aabb_min, aabb_max):
                continue

            ground_z = terrain_hit_z(x, y, terrain_id, cid)
            if ground_z is None:
                continue

            mode = random.choice(kind["burial_bias"])
            scale = random.uniform(kind["scale_range"][0], kind["scale_range"][1])
            z, quat = placement_pose(kind, mode, ground_z)

            try:
                body_id = p.loadURDF(
                    kind["path"],
                    basePosition=[x, y, z],
                    baseOrientation=quat,
                    useFixedBase=True,
                    globalScaling=scale,
                    physicsClientId=cid,
                )
            except Exception as e:
                print(f"[Waste] URDF load failed: {kind['path']} ({e})")
                continue

            waste_items.append(
                {
                    "name": kind["name"],
                    "body_id": body_id,
                    "world_x": x,
                    "world_y": y,
                    "world_z": z,
                    "burial_mode": mode,
                    "obj_scale": scale,
                }
            )
            existing_xy.append((x, y))
            if cluster_seed is None or random.random() < 0.18:
                cluster_seed = (x, y)
            spawned = True
            break

        if not spawned:
            failed += 1

    if failed:
        print(f"[Waste] Could not place {failed} item(s) after retries.")
    if waste_items:
        per_class = {}
        for item in waste_items:
            per_class[item["name"]] = per_class.get(item["name"], 0) + 1
        cls_summary = ", ".join(f"{k}={v}" for k, v in sorted(per_class.items()))
        print(f"[Waste] Class counts: {cls_summary}")
    return waste_items


# ======================================================================
# Mission states
# ======================================================================
FW_SCANNING = 0
FW_RETURN = 1
FW_LAND = 2
SQUAD_GO = 3
SQUAD_BACK = 4
QUAD_LAND = 5
DONE = 6
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
# World geometry
# ======================================================================
HOME_XY = np.array([-15.0, -15.0])
# Keep FW intentionally higher than quad to make detection harder/noisier.
FW_ALT = 11.0
HOME_FW = np.array([HOME_XY[0], HOME_XY[1], FW_ALT])

QUAD_CRUISE_ALT = 3.0  # all drones transit at this single altitude
# (no per-drone layers — preserves uniform mAP conditions)

WORLD_LIMIT = 19.0
Z_MIN, Z_MAX = 0.2, 8.0

# ======================================================================
# Path planning — conflict-aware corridor detours
# ======================================================================
# When two drones have assigned sites that are close together, a corridor
# detour waypoint is inserted so they approach from different angular lanes
# rather than converging on the same region simultaneously.
#
#   Outbound : pad → [corridor_wp?] → site_1 → site_2 → …
#   Return   : last_site → [corridor_wp?] → home_hover
#
# Angles fan 30° apart from BASE_XY — all corridor WPs stay well inside
# the ±WORLD_LIMIT arena.
DEPARTURE_ANGLES_DEG = [0, 30, 60, 90]  # E / ENE / NNE / N
DEPARTURE_RADIUS = 10.0  # m from BASE_XY to corridor waypoint
PATH_VIZ_ALT = 0.6  # z for ground-plane debug-line drawing
PATH_CONFLICT_THRESHOLD = 3.5  # m — site proximity that triggers detour

# ── Shared quad launch base  (matches FW home: HOME_XY) ────────────
#   All 4 quads deploy from pads around the same corner as the FW.
#   Pads are spread 3 m apart (> MIN_SEP=2.0) so collision avoidance
#   never fires while quads are sitting on the ground.
BASE_XY = HOME_XY  # alias — planners use BASE_XY for energy estimates

QUAD_START_XY = np.array(
    [
        [HOME_XY[0] - 2.0, HOME_XY[1] + 2.0],  # 0 Blue   — NW
        [HOME_XY[0] + 2.0, HOME_XY[1] + 2.0],  # 1 Green  — NE
        [HOME_XY[0] - 2.0, HOME_XY[1] - 2.0],  # 2 Red    — SW
        [HOME_XY[0] + 2.0, HOME_XY[1] - 2.0],  # 3 Purple — SE
    ]
)
QUAD_HOVER = np.array([[x, y, QUAD_CRUISE_ALT] for x, y in QUAD_START_XY], dtype=float)

# Seconds between each quad departing toward its first waypoint.
# Quads hold at QUAD_HOVER until their slot opens — simple, physics-safe.
TAKEOFF_STAGGER_SECS = 6.0

# Fraction of battery kept as planning reserve so every assigned quad
# is guaranteed enough energy to reach its last site AND fly home.
BATTERY_SAFETY_FRAC = 0.20  # keep 20 % as reserve (never planned away)

# ======================================================================
# Lawnmower survey waypoints
# ======================================================================
_ys = [-16.0, -10.0, -4.0, 2.0, 8.0, 14.0]
_wps = []
for _i, _y in enumerate(_ys):
    _wps += (
        [[-17.0, _y, FW_ALT], [17.0, _y, FW_ALT]]
        if _i % 2 == 0
        else [[17.0, _y, FW_ALT], [-17.0, _y, FW_ALT]]
    )
FW_SCAN_WPS = np.array(_wps, dtype=float)
FW_SPEED = 4.5
FW_WP_RADIUS = 2.0
FW_LAND_SECS = 4.0

# ======================================================================
# Fixed-wing battery model (aerodynamic + payload power)
# Modelled as a 2 kg, 0.45 m^2 wing-area fixed-wing scout with a 40 Wh
# LiPo pack.  Energy drain uses the parabolic drag-power formula:
#   P_prop = (D * v) / eta,  where D = q * S * (CD0 + k * CL^2)
# Payload power draws are added on top of propulsive power each step.
# ======================================================================
FW_BATTERY_CAPACITY_WH = 40.0
FW_BATTERY_CAPACITY_J = FW_BATTERY_CAPACITY_WH * 3600.0  # 144 000 J

FW_MASS_KG = 2.0  # total aircraft mass including battery (kg)
FW_WING_AREA_M2 = 0.45  # reference wing planform area (m^2)
FW_CD0 = 0.035  # zero-lift parasitic drag coefficient
FW_K_INDUCED = 0.060  # induced-drag factor k = 1/(pi * AR * e)
FW_PROP_EFF = 0.70  # propeller/motor system efficiency

FW_P_AVIONICS_W = 10.0  # flight controller + IMU + GPS (W)
FW_P_COMM_W = 2.0  # radio transceiver (W)
FW_P_AI_W = 0.0  # onboard inference (W) — offloaded to GS in this sim
FW_P_PAYLOAD_W = 0.0  # additional static payload power (W)
FW_P_CAMERA_W = 4.0  # nadir camera + ISP (W) — active only during scan

# Set True to drain FW battery even when quads are operating (SQUAD_GO/BACK).
# False means FW battery is only drained during FW_SCANNING, FW_RETURN, FW_LAND.
DRAIN_FW_OUTSIDE_FW_PHASES = False

# ======================================================================
# Mission communication log
# ======================================================================
DEFAULT_COMM_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mission_comm_log.json"
)

# ======================================================================
# Quad navigation
# ======================================================================
SQUAD_SPD = 2.0
LAND_SECS = 4.0
MIN_SEP = 2.5  # collision avoidance trigger distance (m)
AVOID_GAIN = 0.5
MAX_PUSH = 0.8
ARRIVE_DIST = 0.4  # stop threshold for "overhead" arrival (m)
WP_TIMEOUT_SECS = 45.0
SQUAD_GO_TIMEOUT_SECS = 420.0
SQUAD_BACK_TIMEOUT_SECS = 240.0
HOME_REACHED_DIST = 2.0

# ======================================================================
# Waste cubes
# Bright CYAN [0,255,255] — maximum contrast against the sandy/brown ODM
# ortho texture.  White was too close to light-beige terrain pixels in
# PyBullet's tiny renderer, causing missed detections.
# ======================================================================
WASTE_POSITIONS = [
    (-15.0, -12.0, 0.4),
    (-15.0, 12.0, 0.4),
    (0.0, -15.0, 0.4),
    (0.0, 15.0, 0.4),
    (15.0, -12.0, 0.4),
    (15.0, 12.0, 0.4),
    (-8.0, 0.0, 0.4),
    (8.0, 0.0, 0.4),
]
CUBE_COLOR = [0.0, 1.0, 1.0, 1.0]  # bright cyan
NUM_TARGETS = len(WASTE_POSITIONS)

# ======================================================================
# 4 Drones
# ======================================================================
QUAD_COLORS = [
    [0.20, 0.20, 1.00, 1.0],  # Blue
    [0.20, 0.80, 0.20, 1.0],  # Green
    [1.00, 0.20, 0.20, 1.0],  # Red
    [0.70, 0.20, 0.90, 1.0],  # Purple
]
QUAD_NAMES = ["blue", "green", "red", "purple"]
NUM_DRONES = 4

# ======================================================================
# Simulation defaults
# ======================================================================
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_SIM_HZ = 240
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
FW_CAM_W = 320
FW_CAM_H = 240
FW_CAM_FOV = 105.0
# Camera sits 0.065 m BELOW the FW centre (fuselage half-height=0.055 + 0.01 gap).
# Negative offset means the eye is underneath the airframe so the fuselage
# and wing rectangles are above the camera and never enter the nadir view.
FW_CAM_OFFSET = -0.065
FW_CAM_SAMPLE = 4
FW_CAM_FRAMES_DIR = os.path.join(_HERE, "fw_frames")  # set to "" to disable

# Cyan detection: Hue ≈ 90 (HSV), high saturation, high value.
# This is completely absent from sandy/brown ODM terrain (hue ~15-30).
CYAN_LO = np.array([80, 180, 180], dtype=np.uint8)  # H:80-100 S:180+ V:180+
CYAN_HI = np.array([100, 255, 255], dtype=np.uint8)
MIN_CYAN_PX = 4
MAX_PROJ_SAMP = 200

# ======================================================================
# Quad inspection camera
# ======================================================================
# Quad camera tuned closer to training-style top-down captures.
QUAD_CAM_W = 1024
QUAD_CAM_H = 1024
QUAD_CAM_FOV = 65.0
QUAD_CAM_ALT = 4.0  # overhead hover altitude (m) — 1 m above transit layer
QUAD_DWELL_SECS = 3.5  # hover duration — extra time ensures PID fully settles
# at cam alt before the first Roboflow sample fires
QUAD_FRAMES_DIR = os.path.join(_HERE, "quad_frames")
# Sample only in the second half of the dwell (after PID has settled at cam alt)
QUAD_RF_SAMPLE_FRACS = (0.55, 0.70, 0.85)

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
BATTERY_CAPACITY_J = BATTERY_CAPACITY_WH * 3600  # 15 984 J

P_HOVER = 4.0  # W        — constant hover baseline
K_VELOCITY = 0.05  # W/(m/s)^2 — aerodynamic drag term
K_ACCEL = 0.02  # W/(m/s^2) — manoeuvre / motor-workload term

LOW_BATTERY_RTB = 0.10  # 10% -> abort remaining sites, return to base
LOW_BATTERY_CRITICAL = 0.05  # 5%  -> emergency land in place

# ======================================================================
# Heatmap
# ======================================================================
GRID_MIN = -17.0
GRID_MAX = 17.0
GRID_CELL = 1.0
GRID_N = int((GRID_MAX - GRID_MIN) / GRID_CELL)  # 34

HEAT_SPREAD = 1.2
PEAK_FRAC = 0.20
PEAK_MIN_HEAT = 0.8
PEAK_MIN_SEP = 4.5
HEATMAP_PNG = os.path.join(_HERE, "heatmap_detection.png")
MATCH_RADIUS = 3.5  # m — peak within this distance of GT = true positive
QUAD_MATCH_RADIUS = 4.0
GT_COORDS_CSV = os.path.join(_HERE, "ground_truth_waste.csv")


# ======================================================================
# Battery system
# ======================================================================
# ======================================================================
# CommLog — JSON mission communication log
# Records every inter-agent message (telemetry uplinks, command downlinks,
# task bids, claims) as a flat JSON array written on mission close.
# Encrypted messages carry an extra "encrypted" field so the log clearly
# identifies which records were ChaCha20-Poly1305 protected.
# ======================================================================


def _json_norm(obj):
    """Recursively convert values to JSON-serialisable int/float (incl. numpy)."""
    if isinstance(obj, dict):
        return {k: _json_norm(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_norm(v) for v in obj]
    if isinstance(obj, np.generic):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    return obj


def _comm_header_dict(step, sim_dt, phase, sender, receiver, msg_type):
    """Build the envelope header for one comm-log record."""
    h = {
        "sim_step": int(step),
        "phase": phase,
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
    }
    if sim_dt is not None:
        h["sim_time_s"] = float(step) * float(sim_dt)
    return h


class CommLog:
    """
    Collect messages in memory; write one valid JSON array file on close.

    Each record is an envelope dict:
        sim_step, sim_time_s, phase, sender, receiver, type,
        encrypted (bool), data (payload dict)

    Usage:
        comm_log = CommLog("mission_comm_log.json")
        log_msg(comm_log, step, dt, "FW_SCANNING", "fw", "ground_station",
                "telemetry_uplink", payload_dict, encrypted=True)
        comm_log.close()  # writes the file
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._records: list = []
        self._closed = False

    def _append(self, record: dict):
        if self._closed:
            return
        self._records.append(_json_norm(record))

    def close(self):
        """Write all queued records to disk as a JSON array."""
        if self._closed:
            return
        self._closed = True
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump(self._records, fp, ensure_ascii=False, indent=2)
        self._records = []


def log_msg(
    comm_log,
    step,
    sim_dt,
    phase,
    sender,
    receiver,
    msg_type,
    data,
    *,
    encrypted: bool = False,
):
    """
    Queue one envelope record in comm_log (no-op if comm_log is None).

    Parameters
    ----------
    comm_log  : CommLog | None
    step      : current simulation step counter
    sim_dt    : control timestep in seconds (used to compute sim_time_s)
    phase     : mission-state string, e.g. "FW_SCANNING"
    sender    : e.g. "quad_0", "fw", "ground_station"
    receiver  : e.g. "ground_station", "quad_1"
    msg_type  : e.g. "telemetry_uplink", "command_downlink", "task_claim"
    data      : arbitrary payload dict
    encrypted : True when the message was ChaCha20-Poly1305 encrypted
    """
    if comm_log is None:
        return
    header = _comm_header_dict(step, sim_dt, phase, sender, receiver, msg_type)
    env = {**header, "encrypted": bool(encrypted), "data": data}
    comm_log._append(env)


# ======================================================================
# Fixed-wing battery model
# ======================================================================


class FixedWingBattery:
    """
    Energy state for the fixed-wing scout, drained by aerodynamic propulsive
    power plus fixed payload draws.

    Physics (level flight):
        q   = 0.5 * rho * v^2              (dynamic pressure, Pa)
        CL  = W / (q * S)                  (lift balance: L = W)
        CD  = CD0 + k * CL^2               (parabolic drag polar)
        D   = q * S * CD                   (drag force, N)
        P_prop = (D * v) / eta             (shaft power required, W)
        P_total = P_prop + P_avionics + P_comm + P_camera

    Energy update each control step:
        E[k+1] = max(0, E[k] - P_total * dt)
        SOC    = E / E_max

    Parameters map directly to the FW_* module-level constants so they
    can be overridden per-instance for Monte-Carlo sweeps.
    """

    def __init__(
        self,
        capacity_j: float = FW_BATTERY_CAPACITY_J,
        mass_kg: float = FW_MASS_KG,
        wing_area_m2: float = FW_WING_AREA_M2,
        cd0: float = FW_CD0,
        k_induced: float = FW_K_INDUCED,
        prop_eff: float = FW_PROP_EFF,
        rho_mode: str = "simple",
    ):
        self.capacity_j = float(capacity_j)
        self.energy_j = float(capacity_j)  # start fully charged

        self.mass_kg = float(mass_kg)
        self.wing_area_m2 = float(wing_area_m2)
        self.cd0 = float(cd0)
        self.k_induced = float(k_induced)
        self.prop_eff = float(prop_eff)
        self.rho_mode = str(rho_mode)

    # -- atmospheric model ------------------------------------------------

    @staticmethod
    def rho_from_alt(h_m: float, mode: str = "simple") -> float:
        """
        Air density at altitude h_m.

        mode="simple"  : constant sea-level ISA (1.225 kg/m^3) -- fast.
        mode="isa"     : International Standard Atmosphere troposphere
                         lapse-rate model (valid to 11 km).
        """
        if mode == "isa":
            h = float(max(0.0, h_m))
            T0 = 288.15  # K
            p0 = 101325.0  # Pa
            L = 0.0065  # K/m lapse rate
            g0 = 9.80665  # m/s^2
            R = 287.05  # J/(kg K)
            T = max(1.0, T0 - L * h)
            p = p0 * (T / T0) ** (g0 / (R * L))
            return float(p / (R * T))
        return 1.225  # sea-level ISA (default)

    # -- power model ------------------------------------------------------

    def power_from_speed_alt(self, v_mps: float, h_m: float) -> tuple:
        """
        Compute propulsive and total required power for level flight.

        Returns
        -------
        (p_prop_w, drag_n) : propulsive power (W) and drag force (N)
        """
        v = float(max(0.0, v_mps))
        rho = self.rho_from_alt(h_m, self.rho_mode)

        if v < 1e-6:
            return 0.0, 0.0

        g = 9.80665
        w_n = self.mass_kg * g
        q = 0.5 * rho * v * v
        cl = w_n / max(1e-9, q * self.wing_area_m2)
        cd = self.cd0 + self.k_induced * (cl * cl)
        d_n = q * self.wing_area_m2 * cd
        eta = max(1e-6, self.prop_eff)
        p_prop_w = (d_n * v) / eta
        return float(p_prop_w), float(d_n)

    # -- step update ------------------------------------------------------

    def update(
        self, fw_pos, fw_speed_mps: float, dt: float, camera_active: bool = False
    ) -> tuple:
        """
        Drain energy for one control step and return new SOC.

        Parameters
        ----------
        fw_pos        : (x, y, z) position array -- z used for ISA rho
        fw_speed_mps  : current airspeed (m/s)
        dt            : timestep length (s)
        camera_active : True when the nadir camera ISP is running

        Returns
        -------
        (soc, p_total_w) : state-of-charge in [0, 1] and total power draw
        """
        h_m = float(fw_pos[2]) if fw_pos is not None and len(fw_pos) >= 3 else 0.0
        p_prop_w, _ = self.power_from_speed_alt(fw_speed_mps, h_m)

        p_payload = (
            FW_P_AVIONICS_W
            + FW_P_COMM_W
            + FW_P_AI_W
            + FW_P_PAYLOAD_W
            + (FW_P_CAMERA_W if camera_active else 0.0)
        )
        p_total_w = float(p_prop_w + p_payload)

        self.energy_j = max(0.0, self.energy_j - p_total_w * float(max(0.0, dt)))
        soc = 0.0 if self.capacity_j <= 0 else float(self.energy_j / self.capacity_j)
        return float(max(0.0, min(1.0, soc))), p_total_w


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
        self.n = n
        self.energy_j = np.full(n, BATTERY_CAPACITY_J, dtype=float)
        self.rtb_active = [False] * n  # set True when RTB triggered
        self.emergency = [False] * n  # set True on critical threshold
        self._text_ids = [-1] * n

    # ── core physics ─────────────────────────────────────────────────

    @staticmethod
    def _power(vel, accel):
        """Instantaneous power in Watts for one drone."""
        return (
            P_HOVER
            + K_VELOCITY * float(np.dot(vel, vel))
            + K_ACCEL * float(np.linalg.norm(accel))
        )

    def update(self, obs, prev_vel, dt):
        """Update ALL drone batteries (including ground-sitting drones)."""
        power = np.zeros(self.n)
        for i in range(self.n):
            vel = np.array(obs[i][10:13], dtype=float)
            accel = (vel - prev_vel[i]) / dt if dt > 0 else np.zeros(3)
            p = self._power(vel, accel)
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
            vel = np.array(obs[i][10:13], dtype=float)
            accel = (vel - prev_vel[i]) / dt if dt > 0 else np.zeros(3)
            if obs[i][2] > 0.5:  # only drain if actually flying
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
        p_cruise = P_HOVER + K_VELOCITY * SQUAD_SPD**2
        d_out = math.hypot(site_xy[0] - from_xy[0], site_xy[1] - from_xy[1])
        d_back = math.hypot(to_xy[0] - site_xy[0], to_xy[1] - site_xy[1])
        t_fly = (d_out + d_back) / max(SQUAD_SPD, 1e-6)
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
            self.emergency[i] = True
            self.rtb_active[i] = True
            quad_done[i] = True
            print(
                f"\n  [BATTERY] Quad-{i} ({QUAD_NAMES[i]}) "
                f"CRITICAL {level*100:.1f}% — emergency landing!"
            )
            return "EMERGENCY"

        if level < LOW_BATTERY_RTB and not self.rtb_active[i]:
            self.rtb_active[i] = True
            quad_done[i] = True
            print(
                f"\n  [BATTERY] Quad-{i} ({QUAD_NAMES[i]}) "
                f"LOW {level*100:.1f}% — aborting inspection, RTB"
            )

        if self.rtb_active[i] and not self.emergency[i]:
            # Steer leader back to this quad's pad at the shared base
            home = QUAD_HOVER[i]
            diff = home[:2] - leaders[i][:2]
            dist = np.linalg.norm(diff)
            if dist > 0.3:
                nd = diff / dist
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
                col = [1.0, 0.0, 0.0]
            elif self.rtb_active[i]:
                label = f"Q{i} {pct:.0f}% RTB"
                col = [1.0, 0.4, 0.0]
            elif pct > 60:
                label = f"Q{i} {pct:.0f}%"
                col = [0.0, 1.0, 0.0]
            elif pct > 30:
                label = f"Q{i} {pct:.0f}%"
                col = [1.0, 0.85, 0.0]
            else:
                label = f"Q{i} {pct:.0f}%"
                col = [1.0, 0.0, 0.0]

            pos = list(obs[i][0:3])
            pos[2] += 0.55

            if self._text_ids[i] == -1:
                # First frame — create the item
                self._text_ids[i] = p.addUserDebugText(
                    text=label,
                    textPosition=pos,
                    textColorRGB=col,
                    textSize=1.1,
                    lifeTime=0,
                    physicsClientId=cid,
                )
            else:
                # Subsequent frames — replace in-place (no remove needed,
                # no risk of a ghost label being left behind)
                self._text_ids[i] = p.addUserDebugText(
                    text=label,
                    textPosition=pos,
                    textColorRGB=col,
                    textSize=1.1,
                    lifeTime=0,
                    replaceItemUniqueId=self._text_ids[i],
                    physicsClientId=cid,
                )

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
            rem_j = self.energy_j[i]
            rem_pct = rem_j / BATTERY_CAPACITY_J * 100.0
            used_j = BATTERY_CAPACITY_J - rem_j
            status = (
                "EMERGENCY"
                if self.emergency[i]
                else "RTB" if self.rtb_active[i] else "OK"
            )
            lines.append(
                f"    Quad-{i} ({QUAD_NAMES[i]:6s})  "
                f"remaining={rem_pct:5.1f}%  "
                f"used={used_j/3600:.4f} Wh  status={status}"
            )
        return lines


# ======================================================================
# Heatmap class
# ======================================================================
class WasteHeatmap:

    def __init__(self):
        self.grid = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self.viz_ids = []

    def _wc(self, wx, wy):
        cx = int(np.clip((wx - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        cy = int(np.clip((wy - GRID_MIN) / GRID_CELL, 0, GRID_N - 1))
        return cx, cy

    def _cw(self, cx, cy):
        return (GRID_MIN + (cx + 0.5) * GRID_CELL, GRID_MIN + (cy + 0.5) * GRID_CELL)

    def accumulate(self, world_pts, weight=1.0):
        spread = int(math.ceil(HEAT_SPREAD * 2)) + 1
        two_s2 = 2.0 * HEAT_SPREAD**2
        for wx, wy in world_pts:
            if not (GRID_MIN <= wx <= GRID_MAX and GRID_MIN <= wy <= GRID_MAX):
                continue
            cx, cy = self._wc(wx, wy)
            for dx in range(-spread, spread + 1):
                for dy in range(-spread, spread + 1):
                    ncx, ncy = cx + dx, cy + dy
                    if 0 <= ncx < GRID_N and 0 <= ncy < GRID_N:
                        self.grid[ncx, ncy] += weight * math.exp(
                            -(dx * dx + dy * dy) / two_s2
                        )

    def find_peaks(self):
        gmax = float(self.grid.max())
        if gmax < PEAK_MIN_HEAT:
            return []
        thresh = max(PEAK_FRAC * gmax, PEAK_MIN_HEAT)
        cands = []
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                v = float(self.grid[cx, cy])
                if v >= thresh:
                    wx, wy = self._cw(cx, cy)
                    cands.append((v, wx, wy))
        cands.sort(reverse=True)
        peaks = []
        for _, wx, wy in cands:
            if all(math.hypot(wx - px, wy - py) >= PEAK_MIN_SEP for px, py in peaks):
                peaks.append((wx, wy))
        return peaks

    def visualize(self, cid):
        if self.viz_ids:
            return
        gmax = float(self.grid.max()) or 1.0
        norm = self.grid / gmax
        count = 0
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                v = float(norm[cx, cy])
                if v < 0.04:
                    continue
                wx, wy = self._cw(cx, cy)
                # Heat-colour: blue→red→yellow→white as value rises
                r = float(np.clip(v * 2.0, 0.0, 1.0))
                g = float(np.clip(v * 2.0 - 1.0, 0.0, 1.0))
                b = float(np.clip(1.0 - v * 2.0, 0.0, 1.0))
                # Tile height scales with heat value so hot spots are taller
                tile_h = 0.05 + v * 0.30
                vs = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[GRID_CELL * 0.46, GRID_CELL * 0.46, tile_h / 2],
                    rgbaColor=[r, g, b, 1.0],  # fully opaque
                    physicsClientId=cid,
                )
                # Place tiles at z=0.4 so they sit above max terrain (0.25m)
                p.createMultiBody(
                    0, -1, vs, [wx, wy, 0.4 + tile_h / 2], physicsClientId=cid
                )
                count += 1
        self.viz_ids = [count]
        print(f"  [Heatmap] {count} tiles rendered")

    def mark_peaks(self, peaks, cid):
        for wx, wy in peaks:
            vs = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.75,
                length=0.08,
                rgbaColor=[1.0, 1.0, 0.0, 1.0],
                physicsClientId=cid,
            )
            p.createMultiBody(0, -1, vs, [wx, wy, 0.8], physicsClientId=cid)

    def mark_assignment(self, assignments, cid):
        """Draw a small coloured dot above each peak showing which quad owns it."""
        dot_colors = [
            [0.2, 0.2, 1.0, 1.0],  # blue
            [0.2, 0.8, 0.2, 1.0],  # green
            [1.0, 0.2, 0.2, 1.0],  # red
            [0.7, 0.2, 0.9, 1.0],  # purple
        ]
        for qi, sites in enumerate(assignments):
            for wx, wy in sites:
                vs = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.35,
                    rgbaColor=dot_colors[qi],
                    physicsClientId=cid,
                )
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
        img = cv2.applyColorMap(norm.T[::-1, :], cv2.COLORMAP_HOT)
        h, w = img.shape[:2]
        for i in range(0, GRID_N + 1, 5):
            xi = int(i * w / GRID_N)
            yi = int(i * h / GRID_N)
            cv2.line(img, (xi, 0), (xi, h), (50, 50, 50), 1)
            cv2.line(img, (0, yi), (w, yi), (50, 50, 50), 1)
            wv = int(GRID_MIN + i * GRID_CELL)
            cv2.putText(
                img,
                str(wv),
                (xi + 2, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (200, 200, 200),
                1,
            )
        cv2.imwrite(path, img)
        print(f"  [Heatmap] PNG saved -> {path}")


# ======================================================================
# Hotspot → quad assignment  (pluggable strategy)
# ======================================================================

# Available strategy names (pass via --strategy CLI argument)
STRATEGIES = ["worst_fit", "best_fit", "normalised_spare", "hungarian", "genetic"]


def _build_assignment_state(battery, num_drones):
    """Shared initialisation for all strategies."""
    budget_j = [
        battery.energy_j[qi] * (1.0 - BATTERY_SAFETY_FRAC) for qi in range(num_drones)
    ]
    committed_j = [0.0] * num_drones
    last_xy = [tuple(BASE_XY)] * num_drones
    return budget_j, committed_j, last_xy


def _is_available(qi, battery):
    return not battery.emergency[qi] and not battery.rtb_active[qi]


def assign_peaks_to_quads(peaks, battery, strategy="worst_fit", num_drones=None):
    """
    Energy-constrained site allocation with pluggable strategy.

    Strategies
    ----------
    worst_fit         Min proportional utilisation — load-balanced.
    best_fit          Max absolute spare Joules — bin-packing heuristic.
    normalised_spare  Max (budget - committed) / budget — equiv. worst_fit shown separately.
    hungarian         Optimal min-cost linear assignment per round (Kuhn 1955).
    genetic           GA over full assignment vector (coverage + balance).

    Parameters
    ----------
    peaks       : list of (wx, wy) detected hotspot coordinates
    battery     : BatterySystem instance
    strategy    : one of STRATEGIES
    num_drones  : int, number of active drones (defaults to global NUM_DRONES)

    Returns
    -------
    assignments : list[list[(wx,wy)]]  per-drone site lists
    skipped     : list[(wx,wy)]        sites no quad could reach
    """
    if num_drones is None:
        num_drones = NUM_DRONES
    if not peaks:
        return [[] for _ in range(num_drones)], []

    # Sort nearest-to-base first — used by all greedy strategies
    sorted_peaks = sorted(
        peaks, key=lambda s: math.hypot(s[0] - BASE_XY[0], s[1] - BASE_XY[1])
    )

    if strategy == "worst_fit":
        assignments, skipped, committed_j = _strategy_worst_fit(
            sorted_peaks, battery, num_drones
        )
    elif strategy == "best_fit":
        assignments, skipped, committed_j = _strategy_best_fit(
            sorted_peaks, battery, num_drones
        )
    elif strategy == "normalised_spare":
        assignments, skipped, committed_j = _strategy_normalised_spare(
            sorted_peaks, battery, num_drones
        )
    elif strategy == "hungarian":
        assignments, skipped, committed_j = _strategy_hungarian(
            sorted_peaks, battery, num_drones
        )
    elif strategy == "genetic":
        assignments, skipped, committed_j = _strategy_genetic(
            sorted_peaks, battery, num_drones
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}")

    # ── print allocation table ────────────────────────────────────────
    label_map = {
        "worst_fit": "Worst-Fit Greedy (min utilisation)",
        "best_fit": "Best-Fit Greedy (max spare J)",
        "normalised_spare": "Normalised-Spare Greedy (max spare %)",
        "hungarian": "Hungarian Algorithm (min-cost per round)",
        "genetic": "Genetic Algorithm (coverage + balance)",
    }
    print(
        f"\n  [Assignment] {label_map.get(strategy, strategy)} "
        f"(reserve={int(BATTERY_SAFETY_FRAC*100)}%):"
    )
    for qi in range(num_drones):
        e_pct = battery.energy_j[qi] / BATTERY_CAPACITY_J * 100.0
        c_pct = committed_j[qi] / BATTERY_CAPACITY_J * 100.0
        sites = ", ".join(f"({x:.1f},{y:.1f})" for x, y in assignments[qi])
        print(
            f"    Quad-{qi} ({QUAD_NAMES[qi]:6s}) "
            f"batt={e_pct:5.1f}%  planned={c_pct:4.1f}%  "
            f"{len(assignments[qi])} site(s)  {sites or '(none)'}"
        )
    if skipped:
        sk = ", ".join(f"({x:.1f},{y:.1f})" for x, y in skipped)
        print(f"    *** SKIPPED (no feasible quad): {sk}")
    print()
    return assignments, skipped


# ── Strategy implementations ──────────────────────────────────────────────


def _strategy_worst_fit(sorted_peaks, battery, num_drones):
    """Min proportional utilisation — load-balanced, heterogeneous-battery-safe."""
    budget_j, committed_j, last_xy = _build_assignment_state(battery, num_drones)
    assignments = [[] for _ in range(num_drones)]
    skipped = []
    for wx, wy in sorted_peaks:
        best_qi, best_score = -1, float("inf")
        for qi in range(num_drones):
            if not _is_available(qi, battery):
                continue
            trip_j = BatterySystem.estimate_trip_j(last_xy[qi], (wx, wy))
            total_j = committed_j[qi] + trip_j
            if total_j <= budget_j[qi]:
                score = total_j / max(budget_j[qi], 1e-6)
                if score < best_score:
                    best_score, best_qi = score, qi
        if best_qi >= 0:
            trip_j = BatterySystem.estimate_trip_j(last_xy[best_qi], (wx, wy))
            assignments[best_qi].append((wx, wy))
            committed_j[best_qi] += trip_j
            last_xy[best_qi] = (wx, wy)
        else:
            skipped.append((wx, wy))
    return assignments, skipped, committed_j


def _strategy_best_fit(sorted_peaks, battery, num_drones):
    """Max absolute spare Joules — bin-packing heuristic."""
    budget_j, committed_j, last_xy = _build_assignment_state(battery, num_drones)
    assignments = [[] for _ in range(num_drones)]
    skipped = []
    for wx, wy in sorted_peaks:
        best_qi, best_score = -1, -float("inf")
        for qi in range(num_drones):
            if not _is_available(qi, battery):
                continue
            trip_j = BatterySystem.estimate_trip_j(last_xy[qi], (wx, wy))
            total_j = committed_j[qi] + trip_j
            if total_j <= budget_j[qi]:
                score = budget_j[qi] - total_j  # spare Joules — maximise
                if score > best_score:
                    best_score, best_qi = score, qi
        if best_qi >= 0:
            trip_j = BatterySystem.estimate_trip_j(last_xy[best_qi], (wx, wy))
            assignments[best_qi].append((wx, wy))
            committed_j[best_qi] += trip_j
            last_xy[best_qi] = (wx, wy)
        else:
            skipped.append((wx, wy))
    return assignments, skipped, committed_j


def _strategy_normalised_spare(sorted_peaks, battery, num_drones):
    """Max (budget - committed) / budget — mathematically equivalent to worst_fit."""
    budget_j, committed_j, last_xy = _build_assignment_state(battery, num_drones)
    assignments = [[] for _ in range(num_drones)]
    skipped = []
    for wx, wy in sorted_peaks:
        best_qi, best_score = -1, -float("inf")
        for qi in range(num_drones):
            if not _is_available(qi, battery):
                continue
            trip_j = BatterySystem.estimate_trip_j(last_xy[qi], (wx, wy))
            total_j = committed_j[qi] + trip_j
            if total_j <= budget_j[qi]:
                score = (budget_j[qi] - total_j) / max(budget_j[qi], 1e-6)
                if score > best_score:
                    best_score, best_qi = score, qi
        if best_qi >= 0:
            trip_j = BatterySystem.estimate_trip_j(last_xy[best_qi], (wx, wy))
            assignments[best_qi].append((wx, wy))
            committed_j[best_qi] += trip_j
            last_xy[best_qi] = (wx, wy)
        else:
            skipped.append((wx, wy))
    return assignments, skipped, committed_j


def _strategy_hungarian(sorted_peaks, battery, num_drones):
    """
    Minimum-cost linear assignment solved with the Hungarian algorithm (Kuhn 1955).
    Multi-round: each round assigns at most one site per drone, then energy updated.
    Cost = utilisation fraction (committed + trip) / budget. Infeasible → INF.
    """
    budget_j, committed_j, last_xy = _build_assignment_state(battery, num_drones)
    assignments = [[] for _ in range(num_drones)]
    skipped = []
    remaining = list(sorted_peaks)
    INF = 1e9

    while remaining:
        n_s = len(remaining)
        cost = np.full((num_drones, n_s), INF)
        for qi in range(num_drones):
            if not _is_available(qi, battery):
                continue
            for si, (wx, wy) in enumerate(remaining):
                trip_j = BatterySystem.estimate_trip_j(last_xy[qi], (wx, wy))
                total_j = committed_j[qi] + trip_j
                if total_j <= budget_j[qi]:
                    cost[qi, si] = total_j / max(budget_j[qi], 1e-6)

        row_ind, col_ind = linear_sum_assignment(cost)

        newly_assigned_si = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= INF:
                continue
            wx, wy = remaining[ci]
            assignments[ri].append((wx, wy))
            trip_j = BatterySystem.estimate_trip_j(last_xy[ri], (wx, wy))
            committed_j[ri] += trip_j
            last_xy[ri] = (wx, wy)
            newly_assigned_si.add(ci)

        if not newly_assigned_si:
            skipped.extend(remaining)
            break

        remaining = [remaining[si] for si in range(n_s) if si not in newly_assigned_si]

    return assignments, skipped, committed_j


def _strategy_genetic(
    sorted_peaks,
    battery,
    num_drones,
    pop_size=60,
    generations=120,
    mutation_rate=0.12,
    elite_frac=0.3,
):
    """
    Genetic algorithm — global assignment optimisation.

    Chromosome  : list of length n_sites; gene[i] = drone index (0…n_drones-1).
    Fitness     : 10 × coverage_count  +  1 / (max_utilisation + 0.01).
    Operators   : tournament selection, single-point crossover, random mutation.
    """
    budget_j = [
        battery.energy_j[qi] * (1.0 - BATTERY_SAFETY_FRAC) for qi in range(num_drones)
    ]
    n_sites = len(sorted_peaks)

    def evaluate(chrom):
        asgn = [[] for _ in range(num_drones)]
        comm = [0.0] * num_drones
        lxy = [tuple(BASE_XY)] * num_drones
        valid = []
        for si, qi in enumerate(chrom):
            wx, wy = sorted_peaks[si]
            if not _is_available(qi, battery):
                valid.append(False)
                continue
            trip_j = BatterySystem.estimate_trip_j(lxy[qi], (wx, wy))
            total_j = comm[qi] + trip_j
            if total_j <= budget_j[qi]:
                asgn[qi].append((wx, wy))
                comm[qi] += trip_j
                lxy[qi] = (wx, wy)
                valid.append(True)
            else:
                valid.append(False)
        return asgn, comm, valid

    def fitness(chrom):
        _, comm, valid = evaluate(chrom)
        coverage = sum(valid)
        utils = [comm[qi] / max(budget_j[qi], 1e-6) for qi in range(num_drones)]
        balance = 1.0 / (max(utils) + 0.01)
        return coverage * 10.0 + balance

    population = [
        [random.randint(0, num_drones - 1) for _ in range(n_sites)]
        for _ in range(pop_size)
    ]

    n_elite = max(1, int(pop_size * elite_frac))
    for _gen in range(generations):
        scored = sorted([(fitness(c), c) for c in population], key=lambda x: -x[0])
        elites = [c for _, c in scored[:n_elite]]
        new_pop = list(elites)
        while len(new_pop) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            pt = random.randint(1, n_sites - 1) if n_sites > 1 else 1
            child = p1[:pt] + p2[pt:]
            child = [
                (
                    random.randint(0, num_drones - 1)
                    if random.random() < mutation_rate
                    else g
                )
                for g in child
            ]
            new_pop.append(child)
        population = new_pop

    best_chrom = max(population, key=fitness)
    asgn, committed_j, valid = evaluate(best_chrom)
    skipped = [sorted_peaks[si] for si, v in enumerate(valid) if not v]
    return asgn, skipped, committed_j


# ======================================================================
# Conflict-aware path planning
# ======================================================================


def _seg_min_dist_2d(p1, p2, p3, p4):
    """
    Minimum 2-D Euclidean distance between line segment p1→p2 and p3→p4.
    All inputs are 1-D numpy arrays of length 2.
    """

    def dot(u, v):
        return float(u[0] * v[0] + u[1] * v[1])

    d1, d2, r = p2 - p1, p4 - p3, p1 - p3
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, r)

    if a < 1e-10 and e < 1e-10:
        return float(np.linalg.norm(r))
    if a < 1e-10:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = dot(d1, r)
        if e < 1e-10:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = dot(d1, d2)
            denom = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
                if abs(denom) > 1e-10
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    return float(np.linalg.norm((p1 + s * d1) - (p3 + t * d2)))


def _path_min_sep(path_a, path_b):
    """
    Minimum 2-D separation between two polyline paths.
    Returns inf if either path has fewer than 2 nodes (no segments).
    """
    if len(path_a) < 2 or len(path_b) < 2:
        return float("inf")
    mn = float("inf")
    for i in range(len(path_a) - 1):
        a1 = np.asarray(path_a[i], dtype=float)
        a2 = np.asarray(path_a[i + 1], dtype=float)
        for j in range(len(path_b) - 1):
            b1 = np.asarray(path_b[j], dtype=float)
            b2 = np.asarray(path_b[j + 1], dtype=float)
            d = _seg_min_dist_2d(a1, a2, b1, b2)
            if d < mn:
                mn = d
    return mn


def _drones_conflict(sites_i, sites_j):
    """
    True if drone i and j have a genuine path conflict.

    Only site positions are checked — not departure or inter-site transit
    segments.  All drones fly at the same QUAD_CRUISE_ALT so any two sites
    within PATH_CONFLICT_THRESHOLD can cause both drones to be in the same
    XY region at the same altitude simultaneously.

    Returns (conflict: bool, reason: str).
    """
    for sx, sy in sites_i:
        for tx, ty in sites_j:
            d = math.hypot(sx - tx, sy - ty)
            if d < PATH_CONFLICT_THRESHOLD:
                return True, f"site proximity {d:.2f}m"
    return False, ""


def make_paths(assignments, num_drones=None):
    """
    Build conflict-aware outbound and return waypoint sequences for each drone.

    A corridor detour waypoint (at QUAD_CRUISE_ALT) is inserted ONLY for drones
    whose sites come within PATH_CONFLICT_THRESHOLD of another drone's sites.
    All other drones fly direct — no unnecessary detour.

    All transit waypoints use QUAD_CRUISE_ALT (no per-drone altitude layers).
    Site waypoints use QUAD_CAM_ALT so the ascend/dwell/descend logic fires
    correctly in SQUAD_GO.

    Parameters
    ----------
    assignments : list[list[(wx, wy)]]   per-drone site lists
    num_drones  : int                    active drone count (default NUM_DRONES)

    Returns
    -------
    out_wps    : list[np.ndarray[N,3]]   outbound waypoints (corridor? + sites)
    ret_wps    : list[np.ndarray[M,3]]   return waypoints  (corridor? + home)
    wp_is_site : list[list[bool]]        True for WPs that are inspection sites
    """
    if num_drones is None:
        num_drones = NUM_DRONES

    # ── Conflict detection (site-region only) ───────────────────────────
    needs_corridor = [False] * num_drones
    conflict_log = []

    for i in range(num_drones):
        if not assignments[i]:
            continue
        for j in range(i + 1, num_drones):
            if not assignments[j]:
                continue
            conflict, reason = _drones_conflict(assignments[i], assignments[j])
            if conflict:
                needs_corridor[i] = True
                needs_corridor[j] = True
                conflict_log.append(
                    f"    {QUAD_NAMES[i]}↔{QUAD_NAMES[j]}"
                    f"  {reason} < {PATH_CONFLICT_THRESHOLD}m"
                    f"  → corridor detour added for both"
                )

    # ── Build 3-D waypoint sequences ────────────────────────────────────
    out_wps = []
    ret_wps = []
    wp_is_site = []

    for qi in range(num_drones):
        sites = assignments[qi]
        home_wp = np.array([QUAD_HOVER[qi][0], QUAD_HOVER[qi][1], QUAD_CRUISE_ALT])

        if not sites:
            out_wps.append(np.empty((0, 3), dtype=float))
            ret_wps.append(home_wp.reshape(1, 3))
            wp_is_site.append([])
            continue

        # Site WPs at QUAD_CAM_ALT so SQUAD_GO triggers ascend/dwell
        site_wps = [np.array([x, y, QUAD_CAM_ALT]) for x, y in sites]

        if needs_corridor[qi]:
            angle_rad = math.radians(DEPARTURE_ANGLES_DEG[qi])
            cx = float(
                np.clip(
                    BASE_XY[0] + DEPARTURE_RADIUS * math.cos(angle_rad),
                    -WORLD_LIMIT + 1.0,
                    WORLD_LIMIT - 1.0,
                )
            )
            cy = float(
                np.clip(
                    BASE_XY[1] + DEPARTURE_RADIUS * math.sin(angle_rad),
                    -WORLD_LIMIT + 1.0,
                    WORLD_LIMIT - 1.0,
                )
            )
            # Corridor WP at cruise alt — just a pass-through, no inspection
            corr_wp = np.array([cx, cy, QUAD_CRUISE_ALT])

            out_seq = [corr_wp] + site_wps
            is_site_seq = [False] + [True] * len(sites)
            ret_seq = [corr_wp, home_wp]
        else:
            out_seq = site_wps
            is_site_seq = [True] * len(sites)
            ret_seq = [home_wp]

        out_wps.append(np.array(out_seq, dtype=float))
        ret_wps.append(np.array(ret_seq, dtype=float))
        wp_is_site.append(is_site_seq)

    # ── Summary ─────────────────────────────────────────────────────────
    print("  [PathMaker] Conflict-aware path planning (single cruise altitude):")
    if conflict_log:
        print("  Conflicts detected:")
        for line in conflict_log:
            print(line)
    else:
        print("    No site conflicts — all drones fly direct.")
    print("  Per-drone plan:")
    for qi in range(num_drones):
        sites = assignments[qi]
        if not sites:
            print(f"    Quad-{qi} ({QUAD_NAMES[qi]:6s}) no sites — holds at base")
        elif needs_corridor[qi]:
            cw = out_wps[qi][0]
            print(
                f"    Quad-{qi} ({QUAD_NAMES[qi]:6s})"
                f"  DETOUR via ({cw[0]:.1f},{cw[1]:.1f})"
                f"  → {len(sites)} site(s)"
            )
        else:
            print(
                f"    Quad-{qi} ({QUAD_NAMES[qi]:6s})"
                f"  DIRECT → {len(sites)} site(s)"
            )

    return out_wps, ret_wps, wp_is_site


def draw_drone_paths(out_wps, ret_wps, cid, num_drones=None):
    """
    Visualise each drone's outbound path (bright colour) and return path
    (darker shade) as PyBullet debug lines near ground level.
    Returns list of debug-item IDs for optional later cleanup.
    """
    if num_drones is None:
        num_drones = NUM_DRONES
    line_ids = []
    bright = [
        [0.30, 0.30, 1.00],  # Blue
        [0.20, 0.90, 0.20],  # Green
        [1.00, 0.30, 0.30],  # Red
        [0.80, 0.30, 1.00],  # Purple
    ]
    dim = [[c * 0.45 for c in col] for col in bright]

    def _add_line(ax, ay, bx, by, z, color, width):
        lid = p.addUserDebugLine(
            [ax, ay, z], [bx, by, z], color, lineWidth=width, physicsClientId=cid
        )
        line_ids.append(lid)

    for qi in range(num_drones):
        o = out_wps[qi]
        r = ret_wps[qi]
        if len(o) == 0:
            continue

        hx, hy = QUAD_HOVER[qi][0], QUAD_HOVER[qi][1]

        # Outbound: hover pad → [corridor?] → site(s)
        seq_out = [(hx, hy)] + [(w[0], w[1]) for w in o]
        for k in range(len(seq_out) - 1):
            _add_line(
                seq_out[k][0],
                seq_out[k][1],
                seq_out[k + 1][0],
                seq_out[k + 1][1],
                PATH_VIZ_ALT,
                bright[qi],
                2.0,
            )

        # Return: last site → [corridor?] → hover pad
        last_xy = (o[-1][0], o[-1][1])
        seq_ret = [last_xy] + [(w[0], w[1]) for w in r]
        for k in range(len(seq_ret) - 1):
            _add_line(
                seq_ret[k][0],
                seq_ret[k][1],
                seq_ret[k + 1][0],
                seq_ret[k + 1][1],
                PATH_VIZ_ALT + 0.12,
                dim[qi],
                1.5,
            )

    return line_ids


# ======================================================================
# Fixed-wing camera + detection
# ======================================================================


def fw_camera_rgb(fw_pos, fw_dir, cid):
    eye = [fw_pos[0], fw_pos[1], fw_pos[2] + FW_CAM_OFFSET]
    target = [fw_pos[0], fw_pos[1], 0.0]
    fwd = np.array([fw_dir[0], fw_dir[1], 0.0])
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    view = p.computeViewMatrix(eye, target, fwd.tolist())
    proj = p.computeProjectionMatrixFOV(FW_CAM_FOV, FW_CAM_W / FW_CAM_H, 0.1, 60.0)
    raw = p.getCameraImage(
        FW_CAM_W, FW_CAM_H, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid
    )
    return np.reshape(raw[2], (FW_CAM_H, FW_CAM_W, 4))[:, :, :3].astype(np.uint8)


def _pixel_to_world_pts(pixel_coords, fw_pos, fw_dir):
    """Project nadir camera pixel coords to world XY (same geometry as cyan detector)."""
    if not pixel_coords:
        return []
    cam_h = fw_pos[2] + FW_CAM_OFFSET
    half_fwd = cam_h * math.tan(math.radians(FW_CAM_FOV) / 2)
    half_rgt = half_fwd * (FW_CAM_W / FW_CAM_H)
    yaw = math.atan2(fw_dir[1], fw_dir[0])
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    rgt = np.array([math.sin(yaw), -math.cos(yaw)])

    pts = []
    for px, py in pixel_coords:
        nr = (px / FW_CAM_W - 0.5) * 2.0
        nu = -(py / FW_CAM_H - 0.5) * 2.0
        wx = fw_pos[0] + nu * half_fwd * fwd[0] + nr * half_rgt * rgt[0]
        wy = fw_pos[1] + nu * half_fwd * fwd[1] + nr * half_rgt * rgt[1]
        pts.append((wx, wy))
    return pts


def detect_roboflow_pts(
    rgb, fw_pos, fw_dir, *, confidence=60, overlap=30, allowed_classes=None
):
    """
    Detect waste via Roboflow model; project bbox centers to world XY.
    Returns (pts, n_detections, classes_seen).
    """
    pixel_centers, scores, classes_seen = _roboflow_pixel_centers(
        rgb,
        confidence=confidence,
        overlap=overlap,
        allowed_classes=allowed_classes,
    )
    pts = _pixel_to_world_pts(pixel_centers, fw_pos, fw_dir)
    return pts, len(pts), classes_seen, scores


def _roboflow_pixel_centers(rgb, *, confidence=60, overlap=30, allowed_classes=None):
    if _ROBOFLOW_MODEL is None or not _HAS_CV2:
        return [], [], set()
    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        cv2.imwrite(tmp_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        pred = _ROBOFLOW_MODEL.predict(
            tmp_path, confidence=int(confidence), overlap=int(overlap)
        )
        data = (
            pred.json()
            if hasattr(pred, "json")
            else (pred if isinstance(pred, dict) else {})
        )
        raw = data.get("predictions", []) or []

        if allowed_classes is None:
            allowed = {
                "waste",
                "plastic waste",
                "glass waste",
                "plastic-waste",
                "plastic_waste",
                "bottlewaste",
                "bottle-waste",
                "bottle_waste",
                "bottle waste",
            }
        else:
            allowed = {c.strip().lower() for c in allowed_classes if c.strip()}

        classes_seen = set()
        pixel_centers = []
        scores = []

        for det in raw:
            cls = (det.get("class") or det.get("predicted_class") or "").strip().lower()
            if cls:
                classes_seen.add(cls)
            if allowed and (not cls or cls not in allowed):
                continue

            x = (
                det.get("x")
                or det.get("x_center")
                or (det.get("width", 0) / 2 + det.get("x_min", 0))
            )
            y = (
                det.get("y")
                or det.get("y_center")
                or (det.get("height", 0) / 2 + det.get("y_min", 0))
            )
            conf = det.get("confidence", det.get("score", 0.0))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                pixel_centers.append((float(x), float(y)))
                scores.append(float(conf) if isinstance(conf, (int, float)) else 0.0)

        return pixel_centers, scores, classes_seen
    except Exception as e:
        if getattr(_roboflow_pixel_centers, "_warn_once", True):
            print(f"  [Roboflow] Detection error: {e}")
            _roboflow_pixel_centers._warn_once = False
        return [], [], set()
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def detect_cyan_pts(rgb, fw_pos, fw_dir):
    """Detect bright cyan waste cubes in the FW nadir image and project to world XY."""
    if _HAS_CV2:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, CYAN_LO, CYAN_HI)
    else:
        # Fallback: cyan = high G, high B, low R
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        mask = ((r < 80) & (g > 160) & (b > 160) & (np.abs(g - b) < 40)).astype(
            np.uint8
        ) * 255

    n_px = int(np.count_nonzero(mask))
    if n_px < MIN_CYAN_PX:
        return [], n_px

    ys, xs = np.where(mask > 0)
    ns = min(len(xs), MAX_PROJ_SAMP)
    idx = np.random.choice(len(xs), ns, replace=False)

    # Camera height above ground for projection geometry
    cam_h = fw_pos[2] + FW_CAM_OFFSET
    half_fwd = cam_h * math.tan(math.radians(FW_CAM_FOV) / 2)
    half_rgt = half_fwd * (FW_CAM_W / FW_CAM_H)
    yaw = math.atan2(fw_dir[1], fw_dir[0])
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    rgt = np.array([math.sin(yaw), -math.cos(yaw)])

    pts = []
    for px, py in zip(xs[idx], ys[idx]):
        nr = (px / FW_CAM_W - 0.5) * 2.0
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
    eye = [quad_pos[0], quad_pos[1], float(quad_pos[2])]
    target = [quad_pos[0], quad_pos[1], 0.0]
    up = [1.0, 0.0, 0.0]
    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(
        QUAD_CAM_FOV, QUAD_CAM_W / QUAD_CAM_H, 0.1, 30.0
    )
    raw = p.getCameraImage(
        QUAD_CAM_W,
        QUAD_CAM_H,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=cid,
    )
    return np.reshape(raw[2], (QUAD_CAM_H, QUAD_CAM_W, 4))[:, :, :3].astype(np.uint8)


def detect_roboflow_quad_pts(
    rgb, quad_pos, *, confidence=45, overlap=30, allowed_classes=None
):
    pixels, scores, classes_seen = _roboflow_pixel_centers(
        rgb,
        confidence=confidence,
        overlap=overlap,
        allowed_classes=allowed_classes,
    )
    if not pixels:
        return [], 0, classes_seen, []

    cam_h = max(0.1, float(quad_pos[2]))
    half_fwd = cam_h * math.tan(math.radians(QUAD_CAM_FOV) / 2.0)
    half_rgt = half_fwd * (QUAD_CAM_W / QUAD_CAM_H)
    pts = []
    for px, py in pixels:
        nr = (px / QUAD_CAM_W - 0.5) * 2.0
        nu = -(py / QUAD_CAM_H - 0.5) * 2.0
        wx = float(quad_pos[0]) + nu * half_fwd
        wy = float(quad_pos[1]) + nr * half_rgt
        pts.append((wx, wy))
    return pts, len(pts), classes_seen, scores


_QUAD_COLORS = {
    "bottle": [
        (255, 255, 255),
        (60, 200, 60),
        (200, 60, 60),
        (220, 200, 50),
        (60, 220, 220),
    ],
    "garbage_bag": [(12, 12, 12), (18, 26, 18), (28, 24, 24), (38, 38, 38)],
    "cardboard_box": [(75, 115, 165), (65, 100, 145), (90, 130, 178)],
    "cardboard_bag": [(80, 118, 162), (68, 102, 142), (95, 135, 180)],
}


def augment_quad_frame(canvas_bgr, quad_pos, overlay_items, display_scale=3.0):
    """Paint synthetic waste onto a quad nadir BGR frame using fast cv2
    primitives.  Coordinate mapping is the exact inverse of
    detect_roboflow_quad_pts so painted items round-trip correctly.
    """
    H, W = canvas_bgr.shape[:2]
    cam_h = max(0.1, float(quad_pos[2]))
    half_fwd = cam_h * math.tan(math.radians(QUAD_CAM_FOV) / 2.0)
    half_rgt = half_fwd * (QUAD_CAM_W / QUAD_CAM_H)
    margin = max(half_fwd, half_rgt) * 1.15

    for item in overlay_items:
        wx, wy = item["world_x"], item["world_y"]
        if abs(wx - quad_pos[0]) > margin or abs(wy - quad_pos[1]) > margin:
            continue
        dx = wx - float(quad_pos[0])
        dy = wy - float(quad_pos[1])
        px = int(((dy / half_rgt) / 2.0 + 0.5) * W)
        py = int((0.5 - (dx / half_fwd) / 2.0) * H)
        if not (4 <= px < W - 4 and 4 <= py < H - 4):
            continue

        name = item["name"]
        ang = -math.degrees(item["yaw"])
        oscale = float(item["obj_scale"])
        S = oscale * display_scale
        seed = abs(hash((round(wx, 2), round(wy, 2), name))) % (2**31)
        rng = random.Random(seed)
        palette = _QUAD_COLORS.get(name, [(180, 180, 180)])
        color = palette[seed % len(palette)]

        if name == "bottle":
            hl = max(4, int(7 * S))
            hr = max(2, int(3 * S))
            cv2.ellipse(canvas_bgr, (px, py), (hl, hr), ang, 0, 360, color, -1)
            cv2.ellipse(canvas_bgr, (px, py), (hl, hr), ang, 0, 360, (40, 40, 40), 1)
        elif name == "garbage_bag":
            r = max(3, int(6 * S))
            rx = max(3, int(r * rng.uniform(0.85, 1.1)))
            ry = max(3, int(r * rng.uniform(0.70, 0.90)))
            cv2.ellipse(canvas_bgr, (px, py), (rx, ry), ang, 0, 360, color, -1)
        elif name == "cardboard_box":
            hw = max(3, int(8 * S))
            hh = max(3, int(6 * S))
            box = cv2.boxPoints(((px, py), (hw * 2, hh * 2), ang)).astype(np.int32)
            cv2.fillConvexPoly(canvas_bgr, box, color)
            cv2.polylines(canvas_bgr, [box], True, (40, 60, 80), 1)
        elif name == "cardboard_bag":
            hw = max(3, int(7 * S))
            hh = max(3, int(7 * S))
            box = cv2.boxPoints(((px, py), (hw * 2, hh * 2), ang)).astype(np.int32)
            cv2.fillConvexPoly(canvas_bgr, box, color)
            cv2.polylines(canvas_bgr, [box], True, (40, 60, 80), 1)


def save_quad_photo(rgb, quad_name, site_idx):
    if not _HAS_CV2:
        return
    os.makedirs(QUAD_FRAMES_DIR, exist_ok=True)
    path = os.path.join(QUAD_FRAMES_DIR, f"site_{site_idx:02d}_{quad_name}.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"    [Cam-{quad_name}] {QUAD_CAM_W}x{QUAD_CAM_H} -> {path}")


def _match_predictions(preds_xy, gt_xy, radius):
    matched_gt = set()
    matched = []
    for pi, (px, py) in enumerate(preds_xy):
        best_gi = -1
        best_d = 1e9
        for gi, (gx, gy) in enumerate(gt_xy):
            if gi in matched_gt:
                continue
            d = math.hypot(px - gx, py - gy)
            if d <= radius and d < best_d:
                best_d = d
                best_gi = gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            matched.append((pi, best_gi, best_d))
    return matched


def _ap_from_scored_points(pred_records, gt_xy, radius):
    """
    Point-based AP proxy:
    sort predictions by score, greedy one-to-one match within radius,
    integrate precision-recall curve (single-class mAP proxy).
    """
    if not pred_records:
        return 0.0
    preds = sorted(pred_records, key=lambda x: x[2], reverse=True)
    matched_gt = set()
    tp = 0
    fp = 0
    pr_points = []
    for px, py, _score in preds:
        best_gi = -1
        best_d = 1e9
        for gi, (gx, gy) in enumerate(gt_xy):
            if gi in matched_gt:
                continue
            d = math.hypot(px - gx, py - gy)
            if d <= radius and d < best_d:
                best_d = d
                best_gi = gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(len(gt_xy), 1)
        pr_points.append((recall, precision))

    # Precision envelope + rectangle integration in recall space.
    recalls = [0.0] + [r for r, _ in pr_points] + [1.0]
    precisions = [1.0] + [p for _, p in pr_points] + [0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return float(ap)


def evaluate_detector(pred_records, gt_xy, radius):
    preds_xy = [(x, y) for x, y, _ in pred_records]
    matched = _match_predictions(preds_xy, gt_xy, radius)
    tp = len(matched)
    fp = max(0, len(pred_records) - tp)
    fn = max(0, len(gt_xy) - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(len(gt_xy), 1)
    mean_err = float(np.mean([d for _, _, d in matched])) if matched else float("nan")
    ap_proxy = _ap_from_scored_points(pred_records, gt_xy, radius)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "mean_err_m": mean_err,
        "ap_proxy": ap_proxy,
        "count_preds": len(pred_records),
    }


def update_best_site_detection(best_entry, q_pts, q_scores, site_xy):
    """Keep one strongest per-site quad detection with a light center-distance prior."""
    for di, (qx, qy) in enumerate(q_pts):
        qscore = float(q_scores[di]) if q_scores and di < len(q_scores) else 0.0
        d_site = math.hypot(qx - site_xy[0], qy - site_xy[1])
        rank = qscore - 0.06 * d_site
        if best_entry is None or rank > best_entry["rank"]:
            best_entry = {"x": qx, "y": qy, "score": qscore, "rank": rank}
    return best_entry


# ======================================================================
# Fixed-wing visual model
# ======================================================================


def _box(half, color, cid):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
    v = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid
    )
    return p.createMultiBody(0, c, v, [0, 0, -500], physicsClientId=cid)


def spawn_fw(cid):
    Y = [1.0, 0.85, 0.0, 1.0]
    D = [0.8, 0.60, 0.0, 1.0]
    return [
        _box([0.55, 0.055, 0.055], Y, cid),
        _box([0.07, 0.85, 0.018], Y, cid),
        _box([0.06, 0.28, 0.012], D, cid),
        _box([0.05, 0.012, 0.200], D, cid),
    ]


def place_fw(ids, pos, dirn, cid):
    yaw = math.atan2(dirn[1], dirn[0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    q = p.getQuaternionFromEuler([0, 0, yaw])
    for bid, (lx, ly, lz) in zip(
        ids, [(0, 0, 0), (0, 0, 0), (-0.52, 0, 0), (-0.52, 0, 0.16)]
    ):
        p.resetBasePositionAndOrientation(
            bid,
            [pos[0] + cy * lx - sy * ly, pos[1] + sy * lx + cy * ly, pos[2] + lz],
            q,
            physicsClientId=cid,
        )


def step_fw(ids, pos, dirn, target, dt, cid):
    diff = target - pos
    dist = math.hypot(diff[0], diff[1])
    if dist < FW_WP_RADIUS:
        place_fw(ids, pos, dirn, cid)
        return pos.copy(), dirn.copy(), True
    nd = diff / (np.linalg.norm(diff) + 1e-9)
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
    out = ldr.copy()
    # Blend z smoothly toward target — never snap it in one step.
    # This prevents sudden altitude commands when the leader z is
    # seeded from a different altitude (e.g. QUAD_CAM_ALT → QUAD_CRUISE_ALT).
    dz = wp[2] - ldr[2]
    out[2] = ldr[2] + np.clip(dz, -speed * dt, speed * dt)
    if dist < ARRIVE_DIST:
        out[2] = wp[2]  # snap z only once we have arrived in XY
        return out, True
    nd = diff / dist
    out[:2] = ldr[:2] + nd * min(speed * dt, dist)
    out[0] = np.clip(out[0], -WORLD_LIMIT, WORLD_LIMIT)
    out[1] = np.clip(out[1], -WORLD_LIMIT, WORLD_LIMIT)
    return out, False


# ======================================================================
# Viewport camera
# ======================================================================


def update_viewport(target, keys):
    c = _cam
    if ord("f") in keys and keys[ord("f")] & p.KEY_WAS_TRIGGERED:
        c["free"] = not c["free"]
        if c["free"]:
            c["tx"], c["ty"], c["tz"] = target
    for k, attr, delta in [
        (p.B3G_LEFT_ARROW, "yaw", -2.0),
        (p.B3G_RIGHT_ARROW, "yaw", 2.0),
        (p.B3G_UP_ARROW, "pitch", 1.5),
        (p.B3G_DOWN_ARROW, "pitch", -1.5),
        (ord("z"), "dist", -0.5),
        (ord("x"), "dist", 0.5),
    ]:
        if k in keys and keys[k] & p.KEY_IS_DOWN:
            if attr == "pitch":
                c[attr] = np.clip(c[attr] + delta, CAM_PITCH_MIN, CAM_PITCH_MAX)
            else:
                c[attr] += delta
    if c["free"]:
        cy_ = math.cos(math.radians(c["yaw"]))
        sy_ = math.sin(math.radians(c["yaw"]))
        if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
            c["tx"] += cy_
            c["ty"] += sy_
        if ord("s") in keys and keys[ord("s")] & p.KEY_IS_DOWN:
            c["tx"] -= cy_
            c["ty"] -= sy_
        if ord("q") in keys and keys[ord("q")] & p.KEY_IS_DOWN:
            c["tz"] += 0.5
        if ord("e") in keys and keys[ord("e")] & p.KEY_IS_DOWN:
            c["tz"] -= 0.5
    if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
        c.update(dist=25.0, yaw=45.0, pitch=-40.0, free=False)
    tgt = [c["tx"], c["ty"], c["tz"]] if c["free"] else list(target)
    p.resetDebugVisualizerCamera(c["dist"], c["yaw"], c["pitch"], tgt)


# ======================================================================
# Report
# ======================================================================


def print_report(
    peaks,
    heatmap,
    total_frames,
    total_px,
    photos_taken,
    assignments,
    fw_eval=None,
    quad_eval=None,
    battery=None,
    skipped=None,
    gt_xy=None,
):
    sep = "=" * 66
    print(f"\n{sep}")
    print("  MISSION COMPLETE — DETECTION REPORT")
    print(sep)
    backend = (
        "Roboflow plastic-waste detections"
        if _ROBOFLOW_MODEL is not None
        else "HSV cyan color thresholding"
    )
    print(f"  FW detection frames           : {total_frames}")
    print(f"  FW detection backend          : {backend}")
    print(f"  Cumulative FW detections/pixels: {total_px}")
    print(f"  Heatmap peak cell value       : {heatmap.grid.max():.2f}")
    n_gt_display = len(gt_xy) if gt_xy is not None else NUM_TARGETS
    print(
        f"  Ground-truth waste sites      : {n_gt_display} ({NUM_TARGETS} URDF + {n_gt_display - NUM_TARGETS} overlay)"
    )
    print(f"  Heatmap hotspots detected     : {len(peaks)}")
    print(f"  Close-up photos saved         : {photos_taken}")
    print()

    # Per-quad assignment summary
    print("  Quad assignments:")
    for qi in range(len(assignments) if assignments else 0):
        sites = assignments[qi] if assignments else []
        print(
            f"    Quad-{qi} ({QUAD_NAMES[qi]:6s}) — " f"{len(sites)} site(s) assigned"
        )

    # Precision / recall — use the full gt_xy (URDF + overlay) when available
    # so the heatmap breakdown is consistent with the mAP evaluation.
    report_gt = (
        [(gx, gy) for gx, gy in gt_xy]
        if gt_xy is not None
        else [(gx, gy) for gx, gy, _ in WASTE_POSITIONS]
    )
    n_gt = len(report_gt)
    true_pos = 0
    false_pos = 0
    matched_gt = set()

    if peaks:
        print("\n  Per-hotspot breakdown:")
        for i, (px, py) in enumerate(peaks):
            dists = [math.hypot(px - gx, py - gy) for gx, gy in report_gt]
            near_d = min(dists)
            near_i = int(np.argmin(dists))
            heat = float(heatmap.grid[heatmap._wc(px, py)])
            owner = next(
                (qi for qi, ss in enumerate(assignments) if (px, py) in ss), -1
            )
            owner_str = (
                f"Quad-{owner} ({QUAD_NAMES[owner]})" if owner >= 0 else "unassigned"
            )
            if near_d <= MATCH_RADIUS and near_i not in matched_gt:
                matched_gt.add(near_i)
                true_pos += 1
                verdict = f"TRUE  POS  (GT#{near_i+1} dist={near_d:.1f}m)"
            else:
                false_pos += 1
                verdict = f"FALSE POS  (nearest GT dist={near_d:.1f}m)"
            print(
                f"    Peak {i+1:2d}  ({px:6.1f},{py:6.1f})  "
                f"heat={heat:6.1f}  owner={owner_str}  {verdict}"
            )
    else:
        print("  No hotspots found.")

    missed = max(0, n_gt - len(matched_gt))
    precision = true_pos / max(len(peaks), 1) * 100
    recall = true_pos / max(n_gt, 1) * 100

    print()
    print(f"  True Positives  : {true_pos} / {n_gt}")
    print(f"  False Positives : {false_pos}")
    print(f"  Missed (FN)     : {missed}")
    print(f"  Precision       : {precision:.1f}%")
    print(f"  Recall          : {recall:.1f}%")
    print(f"\n  Heatmap PNG     : {HEATMAP_PNG}")
    print(f"  Quad photos dir : {QUAD_FRAMES_DIR}/")

    if fw_eval is not None or quad_eval is not None:
        print()
        print(
            "  Coordinate evaluation (single-class point mAP proxy; "
            "GT vs predicted XY):"
        )
    if fw_eval is not None:
        me = fw_eval["mean_err_m"]
        me_s = f"{me:.2f}m" if not np.isnan(me) else "n/a"
        print(
            "    Fixed-wing  "
            f"AP~mAP={fw_eval['ap_proxy']*100:.1f}%  "
            f"P={fw_eval['precision']*100:.1f}%  "
            f"R={fw_eval['recall']*100:.1f}%  "
            f"TP/FP/FN={fw_eval['tp']}/{fw_eval['fp']}/{fw_eval['fn']}  "
            f"mean_err={me_s}"
        )
    if quad_eval is not None:
        me = quad_eval["mean_err_m"]
        me_s = f"{me:.2f}m" if not np.isnan(me) else "n/a"
        print(
            "    Quadcopter  "
            f"AP~mAP={quad_eval['ap_proxy']*100:.1f}%  "
            f"P={quad_eval['precision']*100:.1f}%  "
            f"R={quad_eval['recall']*100:.1f}%  "
            f"TP/FP/FN={quad_eval['tp']}/{quad_eval['fp']}/{quad_eval['fn']}  "
            f"mean_err={me_s}"
        )
    if fw_eval is not None and quad_eval is not None:
        delta = (quad_eval["ap_proxy"] - fw_eval["ap_proxy"]) * 100.0
        print(f"    Delta (Quad - FW) AP~mAP: {delta:+.1f}%")

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


# ======================================================================
# Metrics visualisation
# ======================================================================


def generate_metrics_plots(
    num_drones,
    strategy,
    bat_history,  # list of (time_s, [level_pct per drone])
    init_bat_pct,  # list[float] — starting battery %
    final_bat_pct,  # list[float] — final battery %
    assignments,  # list[list[(wx,wy)]]
    true_pos,
    false_pos,
    missed,
    precision,
    recall,
    photos_taken,
    sim_time_s=0.0,
    wall_time_s=0.0,
    flight_times_s=None,  # list[float] — SQUAD_GO flight time per drone
    waste_positions=None,  # list of (wx,wy,wz) ground-truth positions
    peaks_for_plot=None,  # list of (wx,wy) detected heatmap peaks
    fw_eval=None,  # dict from evaluate_detector for FW scan
    quad_eval=None,  # dict from evaluate_detector for quad photos
    gt_xy=None,  # full GT list (URDF + overlay)
    detection_backend="HSV cyan",
    out_path=None,
):
    """
    Save a 6-panel (3×2) figure summarising the mission run.

    Panels
    ------
    [0,0] Battery level over time (line per drone)
    [0,1] Energy consumed vs remaining per drone (bar)
    [1,0] Detection accuracy — TP/FP/FN bars + precision/recall + FW vs quad AP
    [1,1] Assignment balance — sites per drone vs energy used (twin-axis bar)
    [2,0] Object distribution map — GT waste, detected peaks, drone pads, assignments
    [2,1] Flight time per drone + mission timing summary
    """
    if out_path is None:
        out_path = os.path.join(_HERE, "mission_metrics.png")
    if flight_times_s is None:
        flight_times_s = [0.0] * num_drones
    if waste_positions is None:
        waste_positions = []
    if peaks_for_plot is None:
        peaks_for_plot = []
    if gt_xy is None:
        gt_xy = [(x, y) for x, y, *_ in waste_positions]

    drone_labels = [QUAD_NAMES[i] for i in range(num_drones)]
    colors_mpl = ["#4466FF", "#22BB44", "#FF3333", "#AA33DD"][:num_drones]
    n_gt = len(gt_xy)

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    def _fmt(secs):
        m, s = divmod(int(secs), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    fig.suptitle(
        f"Mission Metrics  |  strategy={strategy}  |  drones={num_drones}"
        f"  |  backend={detection_backend}"
        f"  |  sim={_fmt(sim_time_s)}  wall={_fmt(wall_time_s)}",
        fontsize=11,
        fontweight="bold",
        y=0.998,
    )

    # ── [0,0] Battery level over time ─────────────────────────────────
    ax = axes[0, 0]
    if bat_history:
        times = [t for t, _ in bat_history]
        levels = np.array([lvl for _, lvl in bat_history])
        for i in range(num_drones):
            ax.plot(
                times,
                levels[:, i],
                color=colors_mpl[i],
                label=drone_labels[i],
                linewidth=1.8,
            )
        ax.axhline(
            LOW_BATTERY_RTB * 100,
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"RTB ({LOW_BATTERY_RTB*100:.0f}%)",
        )
        ax.axhline(
            LOW_BATTERY_CRITICAL * 100,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Critical ({LOW_BATTERY_CRITICAL*100:.0f}%)",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Battery level (%)")
        ax.set_ylim(-2, 105)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.35)
    else:
        ax.text(
            0.5,
            0.5,
            "No battery data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="grey",
        )
    ax.set_title("Battery Level Over Time")

    # ── [0,1] Energy consumed vs remaining ────────────────────────────
    ax = axes[0, 1]
    x = np.arange(num_drones)
    width = 0.35
    used_b = [init_bat_pct[i] - final_bat_pct[i] for i in range(num_drones)]
    rem_b = final_bat_pct[:num_drones]

    bars_u = ax.bar(
        x - width / 2,
        used_b,
        width,
        label="Consumed (%)",
        color=colors_mpl,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        rem_b,
        width,
        label="Remaining (%)",
        color=colors_mpl,
        alpha=0.35,
        edgecolor="black",
        linewidth=0.6,
        hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(drone_labels)
    ax.set_ylabel("Battery (%)")
    ax.set_title("Energy Consumed vs Remaining per Drone")
    ax.legend(fontsize=8)
    init_b_max = max(init_bat_pct[:num_drones]) if init_bat_pct else 100
    ax.set_ylim(0, init_b_max * 1.2 + 5)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars_u:
        h = bar.get_height()
        if h > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # ── [1,0] Detection accuracy ───────────────────────────────────────
    ax = axes[1, 0]
    bar_cols = ["#2ECC71", "#E74C3C", "#E67E22"]
    bars = ax.bar(
        ["True Pos", "False Pos", "Missed (FN)"],
        [true_pos, false_pos, missed],
        color=bar_cols,
        edgecolor="black",
        linewidth=0.7,
        width=0.45,
    )
    for bar, val in zip(bars, [true_pos, false_pos, missed]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Count")
    ax.set_title("Detection Accuracy (Heatmap Peaks vs GT)")
    ax.set_ylim(0, max(n_gt + 2, true_pos + false_pos + missed + 2))
    ax.axhline(
        n_gt, color="grey", linestyle=":", linewidth=1, label=f"GT sites = {n_gt}"
    )
    ax.legend(fontsize=8)
    # Text box: heatmap P/R + Roboflow mAP if available
    txt_lines = [
        f"Heatmap Precision: {precision:.1f}%",
        f"Heatmap Recall:    {recall:.1f}%",
        f"Photos taken:      {photos_taken}",
    ]
    if fw_eval is not None:
        txt_lines.append(
            f"FW  AP~mAP: {fw_eval['ap_proxy']*100:.1f}%  "
            f"P={fw_eval['precision']*100:.0f}%  R={fw_eval['recall']*100:.0f}%"
        )
    if quad_eval is not None:
        txt_lines.append(
            f"Quad AP~mAP: {quad_eval['ap_proxy']*100:.1f}%  "
            f"P={quad_eval['precision']*100:.0f}%  R={quad_eval['recall']*100:.0f}%"
        )
    ax.text(
        0.98,
        0.97,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )
    ax.grid(True, alpha=0.3, axis="y")

    # ── [1,1] Assignment balance ───────────────────────────────────────
    ax = axes[1, 1]
    sites_per_drone = [len(assignments[i]) for i in range(num_drones)]
    energy_used_pct = [init_bat_pct[i] - final_bat_pct[i] for i in range(num_drones)]
    x = np.arange(num_drones)
    width = 0.35
    ax2 = ax.twinx()
    b1 = ax.bar(
        x - width / 2,
        sites_per_drone,
        width,
        label="Sites assigned",
        color=colors_mpl,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.bar(
        x + width / 2,
        energy_used_pct,
        width,
        label="Energy used (%)",
        color=colors_mpl,
        alpha=0.40,
        edgecolor="black",
        linewidth=0.6,
        hatch="xx",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(drone_labels)
    ax.set_ylabel("Sites assigned")
    ax2.set_ylabel("Energy used (%)", color="grey")
    ax.set_title(f"Assignment Balance  [{strategy}]")
    ax.set_ylim(0, max(sites_per_drone + [1]) * 1.6)
    ax2.set_ylim(0, max(energy_used_pct + [1]) * 1.6)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # ── [2,0] Object distribution map ─────────────────────────────────
    ax = axes[2, 0]
    ax.set_aspect("equal")
    ax.set_xlim(GRID_MIN - 1, GRID_MAX + 1)
    ax.set_ylim(GRID_MIN - 1, GRID_MAX + 1)
    ax.set_facecolor("#F5F0E8")
    rect = mpatches.Rectangle(
        (GRID_MIN, GRID_MIN),
        GRID_MAX - GRID_MIN,
        GRID_MAX - GRID_MIN,
        linewidth=1.5,
        edgecolor="#555555",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    # Ground-truth waste (all GT including overlay)
    if gt_xy:
        gx_ = [g[0] for g in gt_xy]
        gy_ = [g[1] for g in gt_xy]
        ax.scatter(
            gx_,
            gy_,
            s=70,
            marker="s",
            color="cyan",
            edgecolors="#008888",
            linewidths=1.0,
            zorder=3,
            label=f"GT waste ({len(gt_xy)})",
        )
    # Home base marker
    ax.plot(
        HOME_XY[0],
        HOME_XY[1],
        marker="H",
        markersize=10,
        color="gold",
        markeredgecolor="#888800",
        zorder=5,
        label="Base",
    )
    # Quad pads (triangles coloured per drone)
    for qi in range(num_drones):
        ax.plot(
            QUAD_START_XY[qi, 0],
            QUAD_START_XY[qi, 1],
            marker="^",
            markersize=7,
            color=colors_mpl[qi],
            markeredgecolor="black",
            zorder=4,
            label=f"{drone_labels[qi]} pad",
        )
    # Detected peaks + match-radius rings
    if peaks_for_plot:
        px_ = [p_[0] for p_ in peaks_for_plot]
        py_ = [p_[1] for p_ in peaks_for_plot]
        ax.scatter(
            px_,
            py_,
            s=200,
            marker="*",
            color="#FF6600",
            edgecolors="#AA3300",
            linewidths=0.8,
            zorder=6,
            label=f"Detected peaks ({len(peaks_for_plot)})",
        )
        for p_ in peaks_for_plot:
            circ = plt.Circle(
                (p_[0], p_[1]),
                MATCH_RADIUS,
                color="#FF6600",
                fill=False,
                alpha=0.22,
                linewidth=0.8,
            )
            ax.add_patch(circ)
    # Coloured lines from each drone pad to its assigned sites
    for qi in range(num_drones):
        hx, hy = QUAD_START_XY[qi, 0], QUAD_START_XY[qi, 1]
        for sx, sy in assignments[qi]:
            ax.plot(
                [hx, sx],
                [hy, sy],
                color=colors_mpl[qi],
                linewidth=0.8,
                alpha=0.50,
                zorder=2,
            )
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title("Arena Map: Waste Sites, Peaks & Assignments")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2, linestyle=":")

    # ── [2,1] Flight time per drone + timing summary ───────────────────
    ax = axes[2, 1]
    bar_colors = [
        colors_mpl[i] if flight_times_s[i] > 0 else "#CCCCCC" for i in range(num_drones)
    ]
    ft_bars = ax.bar(
        np.arange(num_drones),
        flight_times_s,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    for bar, val in zip(ft_bars, flight_times_s):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.0f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    valid_ft = [t for t in flight_times_s if t > 0]
    avg_ft = float(np.mean(valid_ft)) if valid_ft else 0.0
    if avg_ft > 0:
        ax.axhline(
            avg_ft,
            color="navy",
            linestyle="--",
            linewidth=1.2,
            label=f"Avg: {avg_ft:.0f}s",
        )
        ax.legend(fontsize=8)
    ax.set_xticks(np.arange(num_drones))
    ax.set_xticklabels(drone_labels)
    ax.set_ylabel("Flight time (s)")
    ax.set_title("Inspection Flight Time per Drone (SQUAD_GO)")
    ax.grid(True, alpha=0.3, axis="y")

    def _fmtlong(secs):
        m, s = divmod(int(secs), 60)
        return f"{m} min {s:02d} sec" if m else f"{s} sec"

    timing_text = (
        f"Sim time  : {_fmtlong(sim_time_s)}\n"
        f"Wall time : {_fmtlong(wall_time_s)}\n"
        f"Avg drone : {_fmtlong(avg_ft)}"
    )
    ax.text(
        0.98,
        0.02,
        timing_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#E8F4FF", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Metrics] Report saved → {out_path}")
    print(
        f"[Metrics] Sim={_fmtlong(sim_time_s)} | Wall={_fmtlong(wall_time_s)} "
        f"| Avg flight={_fmtlong(avg_ft)}"
    )


def banner(state, msg=""):
    print(f"\n{'='*60}\n  [{STATE_NAMES[state]}]  {msg}\n{'='*60}\n")


# ======================================================================
# MAIN
# ======================================================================


def run(
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    plot=DEFAULT_PLOT,
    sim_hz=DEFAULT_SIM_HZ,
    ctrl_hz=DEFAULT_CTRL_HZ,
    out_dir=DEFAULT_OUT_DIR,
    rf_conf=60,
    rf_overlap=30,
    rf_classes="",
    init_battery=None,
    num_drones=None,
    strategy="worst_fit",
    comm_log_path=None,
):
    # Cap active drones at NUM_DRONES (4); default to NUM_DRONES.
    nd = int(
        min(max(1, num_drones if num_drones is not None else NUM_DRONES), NUM_DRONES)
    )
    dt = 1.0 / ctrl_hz

    # Quads spawn at cruise altitude — stable physics from the start.
    # Battery is NOT drained during FW phases (update skipped below).
    init_xyzs = QUAD_HOVER[:nd].copy()

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=nd,
        initial_xyzs=init_xyzs,
        initial_rpys=np.zeros((nd, 3)),
        physics=physics,
        pyb_freq=sim_hz,
        ctrl_freq=ctrl_hz,
        gui=gui,
        user_debug_gui=False,
    )
    cid = env.CLIENT

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=cid)

    # Hide the default PyBullet ground plane — the ODM heightfield replaces it
    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cid)

    # Drone colours — must recolor ALL links, not just base (linkIndex=-1).
    # The CF2X URDF has separate links for arms, motors and propellers which
    # default to light-grey and will flood the FW nadir camera with false
    # white-pixel detections when the quads hover below the FW.
    for i in range(nd):
        num_joints = p.getNumJoints(env.DRONE_IDS[i], physicsClientId=cid)
        for link_idx in range(-1, num_joints):
            try:
                p.changeVisualShape(
                    env.DRONE_IDS[i],
                    link_idx,
                    rgbaColor=QUAD_COLORS[i],
                    physicsClientId=cid,
                )
            except Exception:
                pass

    # Terrain + obstacles + waste
    # ODM map: 128×128 heightfield + 512×512 ortho-texture.
    # resolve asset locations relative to the repository root, not the
    # examples directory, since the latter does not contain the "assets"
    # folder.  This mirrors the fix applied to mission_camera_heatmap.py.
    hm_path = os.path.normpath(
        os.path.join(_HERE, "..", "assets", "odm_heightmap_128.png")
    )
    tex_path = os.path.normpath(
        os.path.join(_HERE, "..", "assets", "odm_texture_512.png")
    )
    if not os.path.isfile(hm_path):
        raise FileNotFoundError(
            f"Heightmap file not found: {hm_path}\n"
            "Please run from the repository root or adjust the path."
        )
    if not os.path.isfile(tex_path):
        print(f"[Warning] Texture file not found: {tex_path} -- terrain will be grey")
    print("[World] Loading ODM terrain...")
    terrain_id = build_odm_terrain(
        heightmap_path=hm_path,
        texture_path=tex_path,
        arena_m=34.0,
        max_height_m=0.25,  # subtle dune relief — keeps terrain below cube bases
        cid=cid,
    )

    aabb_min, aabb_max = p.getAABB(terrain_id, physicsClientId=cid)
    tex_rgb = load_texture_rgb(tex_path)
    waste_items = spawn_synthetic_waste_urdfs(
        terrain_id, tex_rgb, aabb_min, aabb_max, cid
    )

    global WASTE_POSITIONS, NUM_TARGETS
    WASTE_POSITIONS = [
        (item["world_x"], item["world_y"], item["world_z"]) for item in waste_items
    ]
    NUM_TARGETS = len(WASTE_POSITIONS)
    gt_xy = [(x, y) for x, y, _ in WASTE_POSITIONS]

    try:
        with open(GT_COORDS_CSV, "w", encoding="utf-8") as f:
            f.write("idx,name,x,y,z,burial_mode,obj_scale\n")
            for i, item in enumerate(waste_items):
                f.write(
                    f"{i},{item['name']},{item['world_x']:.4f},{item['world_y']:.4f},"
                    f"{item['world_z']:.4f},{item['burial_mode']},{item['obj_scale']:.4f}\n"
                )
        print(f"[Waste] Ground-truth coordinates saved: {GT_COORDS_CSV}")
    except Exception as e:
        print(f"[Waste] Could not save GT coordinates CSV ({e})")

    if waste_items:
        print("[Waste] Ground-truth coordinates:")
        for i, item in enumerate(waste_items):
            print(
                f"  #{i:02d} {item['name']:13s} "
                f"({item['world_x']:.2f}, {item['world_y']:.2f}, {item['world_z']:.2f})"
            )

    excl = (
        list(WASTE_POSITIONS)
        + [(HOME_XY[0], HOME_XY[1], 0.0)]
        + [(x, y, 0.0) for x, y in QUAD_START_XY]
    )
    spawn_random_obstacles(0, 0.3, 34.0, excl, 3.0, cid)
    print(
        f"[World] {NUM_TARGETS} synthetic waste URDFs | {nd} quads | strategy={strategy}"
    )

    # 2D synthetic overlay — items painted directly onto FW camera frames so
    # Roboflow always has something to detect regardless of URDF render limits.
    synth = SyntheticOverlay(
        arena_m=34.0,
        fw_cam_w=FW_CAM_W,
        fw_cam_h=FW_CAM_H,
        fw_cam_fov=FW_CAM_FOV,
        n_items=40,
        seed=42,
    )

    # Merge overlay item positions into ground truth so mAP is computed against
    # every item the FW camera could possibly see (URDF + overlay).
    overlay_gt = [(it["world_x"], it["world_y"]) for it in synth.items]
    existing_gt = set(gt_xy)
    for ox, oy in overlay_gt:
        if not any(math.hypot(ox - gx, oy - gy) < 2.0 for gx, gy in existing_gt):
            gt_xy.append((ox, oy))
            existing_gt.add((ox, oy))
    print(f"[GT] {len(gt_xy)} ground-truth points (URDF + overlay)")

    # Fixed-wing
    fw_ids = spawn_fw(cid)
    fw_pos = HOME_FW.copy().astype(float)
    fw_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    place_fw(fw_ids, fw_pos, fw_dir, cid)

    # Heatmap + detection counters
    heatmap = WasteHeatmap()
    fw_cam_ctr = 0
    total_frames = 0
    total_px = 0
    fw_pred_records = []  # (x, y, score)
    quad_pred_records = []  # (x, y, score)
    fw_eval = None
    quad_eval = None

    allowed_classes = rf_classes.split(",") if rf_classes else None

    # Mission variables
    mission_state = FW_SCANNING
    scan_idx = 0
    fw_land_step = 0
    fw_land_max = int(FW_LAND_SECS * ctrl_hz)
    quad_land_step = 0
    quad_land_max = int(LAND_SECS * ctrl_hz)

    # Per-quad inspection state  (populated after FW lands)
    assignments = [[] for _ in range(nd)]  # (wx,wy) lists
    skipped = []  # energy-infeasible sites
    # Per-quad waypoints as numpy arrays [N,3]
    quad_wps = [np.empty((0, 3), dtype=float) for _ in range(nd)]
    quad_is_site = [[] for _ in range(nd)]  # bool per WP — True = inspection site
    ret_wps_plan = [np.empty((0, 3), dtype=float) for _ in range(nd)]
    ret_wp_idx = [0] * nd  # current index in ret_wps_plan during SQUAD_BACK
    quad_wp_idx = [0] * nd  # current WP index per quad
    quad_wp_age = [0] * nd  # time spent trying current waypoint
    quad_phase = ["transit"] * nd  # 'transit'|'ascend'|'dwell'|'descend'
    quad_dwell = [0] * nd  # dwell counter per quad
    quad_site_best = [None] * nd  # best quad detection for current site
    quad_done = [False] * nd  # finished all assigned WPs?
    quad_photo_taken = [False] * nd  # photo fired for current site?
    dwell_max = int(QUAD_DWELL_SECS * ctrl_hz)
    photos_taken = 0

    # Software leaders (3-D position targets fed to PID)
    leaders = QUAD_HOVER[:nd].copy().astype(float)

    # Per-quad departure stagger: quad i waits i*TAKEOFF_STAGGER_SECS
    # before flying toward its first waypoint.  Until then it holds hover.
    stagger_steps = [int(i * TAKEOFF_STAGGER_SECS * ctrl_hz) for i in range(nd)]
    squad_go_step = 0  # counts steps inside SQUAD_GO
    squad_back_step = 0  # counts steps inside SQUAD_BACK

    # Battery system (Zeng et al. 2018)
    battery = BatterySystem(nd)

    # ── Fixed-wing battery (aerodynamic + payload model) ─────────────────
    fw_battery = FixedWingBattery()
    fw_soc = 1.0  # state-of-charge in [0, 1]
    fw_camera_active = False  # set True each step the nadir cam runs

    # ── Secure comms layer (ChaCha20-Poly1305 benchmark) ─────────────────
    comms = DroneCommsLayer(num_drones=nd)
    prev_crypto_j = [0.0] * nd  # tracks cumulative crypto energy for delta drain

    # ── Mission communication log ─────────────────────────────────────────
    comm_log = CommLog(comm_log_path) if comm_log_path else None

    # Apply pre-depletion if --battery was passed
    if init_battery:
        for drone_idx, frac in init_battery.items():
            if 0 <= drone_idx < nd:
                battery.energy_j[drone_idx] = BATTERY_CAPACITY_J * float(frac)
                print(
                    f"  [Battery] Quad-{drone_idx} ({QUAD_NAMES[drone_idx]}) "
                    f"pre-depleted to {frac*100:.0f}%"
                )
    init_bat_pct = [battery.energy_j[i] / BATTERY_CAPACITY_J * 100.0 for i in range(nd)]
    prev_vel = np.zeros((nd, 3), dtype=float)
    bat_levels = np.ones(nd, dtype=float)  # start at 100%

    # Telemetry for the metrics report
    bat_history = []  # [(time_s, [pct_per_drone])]
    bat_sample_ctr = 0  # sample every ~1 s of sim time
    bat_sample_every = max(1, int(ctrl_hz))
    quad_flight_start_step = [None] * nd
    quad_flight_end_step = [None] * nd
    _mission_wall_t0 = time.time()

    # PID + logging
    ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(nd)]
    action = np.zeros((nd, 4))
    logger = Logger(logging_freq_hz=ctrl_hz, num_drones=nd, output_folder=out_dir)
    target_pos = QUAD_HOVER[:nd].copy().astype(float)

    t0 = time.time()
    step = 0

    _det_base = (
        "Roboflow plastic-waste detection"
        if _ROBOFLOW_MODEL is not None
        else "HSV cyan-cube detection"
    )
    mode_str = _det_base
    if synth is not None:
        mode_str += " + synthetic FW overlay (2D)"
    if synth_3d:
        mode_str += " + synthetic URDF on terrain (3D)"
    banner(
        FW_SCANNING,
        f"{mode_str} | {len(FW_SCAN_WPS)} lines | "
        f"{nd} quads hovering at base — battery paused until FW returns",
    )
    print(
        f"[Fleet] Yellow FW=scout | "
        + " | ".join(f"Quad-{i}={QUAD_NAMES[i]}" for i in range(nd))
    )
    print("[Controls] Arrows:rotate | Z/X:zoom | F:free | R:reset | Ctrl+C:abort\n")

    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=cid)
            obs = [env._getDroneStateVector(i) for i in range(nd)]

            # ── FW_SCANNING ──────────────────────────────────────────
            if mission_state == FW_SCANNING:

                fw_pos, fw_dir, reached = step_fw(
                    fw_ids, fw_pos, fw_dir, FW_SCAN_WPS[scan_idx], dt, cid
                )

                fw_cam_ctr += 1
                fw_camera_active = False  # reset each step; set True below if cam fires
                if fw_cam_ctr >= FW_CAM_SAMPLE:
                    fw_cam_ctr = 0
                    fw_camera_active = True  # camera ISP running this step
                    rgb = fw_camera_rgb(fw_pos, fw_dir, cid)
                    if synth is not None:
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        synth.augment(bgr, fw_pos, fw_dir, FW_CAM_OFFSET)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if _ROBOFLOW_MODEL is not None:
                        pts, n_px, _classes_seen, rf_scores = detect_roboflow_pts(
                            rgb,
                            fw_pos,
                            fw_dir,
                            confidence=rf_conf,
                            overlap=rf_overlap,
                            allowed_classes=allowed_classes,
                        )
                        det_label = "plastic"
                        min_detection = 1
                        for (wx, wy), sc in zip(
                            pts, rf_scores if rf_scores else [0.0] * len(pts)
                        ):
                            fw_pred_records.append((float(wx), float(wy), float(sc)))
                    else:
                        pts, n_px = detect_cyan_pts(rgb, fw_pos, fw_dir)
                        det_label = "cyan_px"
                        min_detection = MIN_CYAN_PX

                    # Save frame so you can inspect what the FW actually sees
                    if _HAS_CV2 and FW_CAM_FRAMES_DIR:
                        os.makedirs(FW_CAM_FRAMES_DIR, exist_ok=True)
                        frame_path = os.path.join(
                            FW_CAM_FRAMES_DIR, f"fw_{total_frames:05d}.png"
                        )
                        cv2.imwrite(frame_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                    if n_px >= min_detection:
                        total_frames += 1
                        total_px += n_px
                        heatmap.accumulate(pts)
                        if total_frames % 10 == 1:
                            print(
                                f"  [FW-cam] ({fw_pos[0]:5.1f},{fw_pos[1]:5.1f}) "
                                f"{det_label}={n_px:4d}  "
                                f"heat_max={heatmap.grid.max():.1f}"
                            )
                    else:
                        # Always show a low-rate heartbeat so silence is obvious
                        if step % (FW_CAM_SAMPLE * 75) == 0:
                            print(
                                f"  [FW-cam] ({fw_pos[0]:5.1f},{fw_pos[1]:5.1f}) "
                                f"{det_label}={n_px} (below threshold)"
                            )

                if reached:
                    scan_idx += 1
                    print(
                        f"  [FW] WP {scan_idx}/{len(FW_SCAN_WPS)} | "
                        f"heat_max={heatmap.grid.max():.1f}"
                    )
                    if scan_idx >= len(FW_SCAN_WPS):
                        mission_state = FW_RETURN
                        banner(FW_RETURN, "Scan complete — returning home")

                # Quads hold at base hover — battery not drained yet
                for i in range(nd):
                    target_pos[i] = QUAD_HOVER[i].copy()

            # ── FW_RETURN ────────────────────────────────────────────
            elif mission_state == FW_RETURN:

                fw_pos, fw_dir, reached = step_fw(
                    fw_ids, fw_pos, fw_dir, HOME_FW, dt, cid
                )
                if reached:
                    mission_state = FW_LAND
                    fw_land_step = 0
                    banner(FW_LAND, "Fixed-wing landing — analysing heatmap...")

                for i in range(nd):
                    target_pos[i] = QUAD_HOVER[i].copy()

            # ── FW_LAND ──────────────────────────────────────────────
            elif mission_state == FW_LAND:

                alpha = min(1.0, fw_land_step / max(1, fw_land_max))
                fw_pos[2] = FW_ALT * (1.0 - alpha) + 0.15 * alpha
                place_fw(fw_ids, fw_pos, fw_dir, cid)
                fw_land_step += 1

                for i in range(nd):
                    target_pos[i] = QUAD_HOVER[i].copy()

                if fw_land_step >= fw_land_max:
                    peaks = heatmap.find_peaks()
                    heatmap.visualize(cid)
                    heatmap.mark_peaks(peaks, cid)
                    heatmap.save_png()
                    # Evaluate FW directly from Roboflow predictions collected in scan.
                    fw_eval = evaluate_detector(fw_pred_records, gt_xy, MATCH_RADIUS)

                    assignments, skipped = assign_peaks_to_quads(
                        peaks, battery, strategy=strategy, num_drones=nd
                    )
                    heatmap.mark_assignment(assignments, cid)

                    # ── Downlink: encrypt + send site assignments to each drone ──
                    for i in range(nd):
                        cmd_payload = {
                            "drone_id": i,
                            "sites": [list(s) for s in assignments[i]],
                            "strategy": strategy,
                            "state": "SQUAD_GO",
                        }
                        comms.send_command(i, cmd_payload)
                        # Record in comm-log as an encrypted downlink
                        log_msg(
                            comm_log,
                            step,
                            dt,
                            "FW_LANDING",
                            "ground_station",
                            f"quad_{i}",
                            "command_downlink",
                            cmd_payload,
                            encrypted=True,
                        )

                    # Build conflict-aware per-drone path plans
                    out_wps, ret_wps_plan, wp_is_site_raw = make_paths(
                        assignments, num_drones=nd
                    )
                    draw_drone_paths(out_wps, ret_wps_plan, cid, num_drones=nd)

                    # Populate per-quad navigation state from path plan
                    for qi in range(nd):
                        quad_wps[qi] = out_wps[qi]
                        quad_is_site[qi] = wp_is_site_raw[qi]
                        quad_wp_idx[qi] = 0
                        quad_wp_age[qi] = 0
                        quad_phase[qi] = "transit"
                        quad_dwell[qi] = 0
                        quad_site_best[qi] = None
                        quad_photo_taken[qi] = False
                        quad_done[qi] = len(quad_wps[qi]) == 0
                        ret_wp_idx[qi] = 0

                    # Leaders at current hover positions
                    for i in range(nd):
                        leaders[i] = QUAD_HOVER[i].copy()

                    squad_go_step = 0
                    mission_state = SQUAD_GO
                    total_assigned = sum(len(a) for a in assignments)
                    banner(
                        SQUAD_GO,
                        f"{nd} quads | {total_assigned} site(s) | "
                        f"staggered departure every {TAKEOFF_STAGGER_SECS:.0f}s",
                    )

            # ── SQUAD_GO ─────────────────────────────────────────────
            elif mission_state == SQUAD_GO:

                squad_go_step += 1

                for qi in range(nd):

                    # ── stagger: hold at hover until this quad's slot opens
                    if squad_go_step < stagger_steps[qi]:
                        leaders[qi] = QUAD_HOVER[qi].copy()
                        target_pos[qi] = QUAD_HOVER[qi].copy()
                        continue

                    # ── already done: hold at transit alt ────────────────
                    # Pull z back to QUAD_CRUISE_ALT so SQUAD_BACK never has
                    # to combine descent + translation in the same command.
                    if quad_done[qi]:
                        leaders[qi][2] = QUAD_CRUISE_ALT

                    # ── finished all assigned waypoints ───────────────────
                    elif quad_wp_idx[qi] >= len(quad_wps[qi]):
                        quad_done[qi] = True
                        print(f"  [Quad-{qi}/{QUAD_NAMES[qi]}] All sites done")

                    # ── TRANSIT: cruise toward next WP ────────────────────
                    elif quad_phase[qi] == "transit":
                        quad_wp_age[qi] += 1
                        wp = quad_wps[qi][quad_wp_idx[qi]]
                        leaders[qi], here = step_leader(leaders[qi], wp, dt)
                        if here:
                            # Check whether this WP is a real inspection site
                            is_site = (
                                quad_is_site[qi][quad_wp_idx[qi]]
                                if quad_wp_idx[qi] < len(quad_is_site[qi])
                                else True
                            )
                            if is_site:
                                # Arrived XY-overhead — begin vertical ascent
                                quad_phase[qi] = "ascend"
                                quad_wp_age[qi] = 0
                                quad_dwell[qi] = 0
                                quad_site_best[qi] = None
                                quad_photo_taken[qi] = False
                                wx, wy = wp[0], wp[1]
                                print(
                                    f"  [Quad-{qi}/{QUAD_NAMES[qi]}] Overhead "
                                    f"({wx:.1f},{wy:.1f}) — ascending to cam alt…"
                                )
                            else:
                                # Corridor pass-through — just advance to next WP
                                cx, cy = wp[0], wp[1]
                                print(
                                    f"  [Quad-{qi}/{QUAD_NAMES[qi]}] Corridor WP "
                                    f"({cx:.1f},{cy:.1f}) — continuing to next site"
                                )
                                quad_wp_idx[qi] += 1
                                quad_wp_age[qi] = 0
                        elif quad_wp_age[qi] >= int(WP_TIMEOUT_SECS * ctrl_hz):
                            wx, wy = quad_wps[qi][quad_wp_idx[qi]][:2]
                            print(
                                f"  [Quad-{qi}/{QUAD_NAMES[qi]}] WP timeout at "
                                f"({wx:.1f},{wy:.1f}) — skipping to next site"
                            )
                            quad_wp_idx[qi] += 1
                            quad_wp_age[qi] = 0
                            quad_phase[qi] = "transit"
                            quad_dwell[qi] = 0
                            quad_site_best[qi] = None
                            quad_photo_taken[qi] = False

                    # ── ASCEND: rise vertically to QUAD_CAM_ALT ───────────
                    # XY is held fixed; only z moves.  The drone is stationary
                    # overhead while climbing so the PID has time to settle
                    # before any Roboflow inference fires.
                    elif quad_phase[qi] == "ascend":
                        dz = QUAD_CAM_ALT - leaders[qi][2]
                        leaders[qi][2] += np.clip(dz, -SQUAD_SPD * dt, SQUAD_SPD * dt)
                        if abs(dz) < 0.05:
                            leaders[qi][2] = QUAD_CAM_ALT
                            quad_phase[qi] = "dwell"

                    # ── DWELL: hover at QUAD_CAM_ALT, run RF, save photo ──
                    # RF samples fire only in the second half of the dwell
                    # window (QUAD_RF_SAMPLE_FRACS starts at 0.55) so the PID
                    # has fully settled before any inference request is sent.
                    elif quad_phase[qi] == "dwell":
                        leaders[qi][2] = QUAD_CAM_ALT  # lock altitude
                        quad_dwell[qi] += 1
                        # Read site coordinates directly from the planned WP
                        wx, wy = float(quad_wps[qi][quad_wp_idx[qi]][0]), float(
                            quad_wps[qi][quad_wp_idx[qi]][1]
                        )

                        sample_steps = {
                            max(1, min(dwell_max, int(round(dwell_max * frac))))
                            for frac in QUAD_RF_SAMPLE_FRACS
                        }
                        # Confidence threshold is lowered relative to the FW
                        # scan so close-up quad images can surface detections
                        # that would be filtered out at wide-area survey scale.
                        _quad_conf = max(5, rf_conf - 32)
                        if (
                            _ROBOFLOW_MODEL is not None
                            and quad_dwell[qi] in sample_steps
                        ):
                            rgb_s = quad_camera_rgb(obs[qi][0:3], cid)
                            if synth is not None:
                                bgr_s = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR)
                                augment_quad_frame(bgr_s, obs[qi][0:3], synth.items)
                                rgb_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2RGB)
                            q_pts, q_n, _q_cls, q_scores = detect_roboflow_quad_pts(
                                rgb_s,
                                obs[qi][0:3],
                                confidence=_quad_conf,
                                overlap=rf_overlap,
                                allowed_classes=allowed_classes,
                            )
                            if q_n > 0:
                                quad_site_best[qi] = update_best_site_detection(
                                    quad_site_best[qi], q_pts, q_scores, (wx, wy)
                                )

                        # Fire photo exactly once at the midpoint of the dwell.
                        # Re-anchor t0 after capture so sync() doesn't race.
                        if (
                            not quad_photo_taken[qi]
                            and quad_dwell[qi] >= dwell_max // 2
                        ):
                            site_n = quad_wp_idx[qi] + 1
                            rgb = quad_camera_rgb(obs[qi][0:3], cid)
                            if synth is not None:
                                bgr_q = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                augment_quad_frame(bgr_q, obs[qi][0:3], synth.items)
                                rgb = cv2.cvtColor(bgr_q, cv2.COLOR_BGR2RGB)
                            save_quad_photo(rgb, QUAD_NAMES[qi], qi * 100 + site_n)
                            photos_taken += 1
                            quad_photo_taken[qi] = True
                            if _ROBOFLOW_MODEL is not None:
                                q_pts, q_n, _q_cls, q_scores = detect_roboflow_quad_pts(
                                    rgb,
                                    obs[qi][0:3],
                                    confidence=_quad_conf,
                                    overlap=rf_overlap,
                                    allowed_classes=allowed_classes,
                                )
                                if q_n > 0:
                                    quad_site_best[qi] = update_best_site_detection(
                                        quad_site_best[qi], q_pts, q_scores, (wx, wy)
                                    )
                                    print(
                                        f"    [Quad-{qi}] RF detect -> "
                                        f"{q_n} object(s)"
                                    )
                                else:
                                    print(f"    [Quad-{qi}] RF detect -> none")
                            t0 = time.time() - step * dt  # re-anchor sync clock
                            print(
                                f"  [Quad-{qi}/{QUAD_NAMES[qi]}] Photo @ "
                                f"({wx:.1f},{wy:.1f}) site {site_n}"
                            )

                        # Dwell complete — commit the best detection and begin
                        # descending back to transit altitude before moving on.
                        if quad_dwell[qi] >= dwell_max:
                            if quad_site_best[qi] is not None:
                                best = quad_site_best[qi]
                                quad_pred_records.append(
                                    (best["x"], best["y"], best["score"])
                                )
                            quad_phase[qi] = "descend"

                    # ── DESCEND: drop back to transit alt, then advance WP ─
                    # XY locked during descent — no lateral movement until the
                    # drone is back at cruise layer and stable.
                    elif quad_phase[qi] == "descend":
                        dz = QUAD_CRUISE_ALT - leaders[qi][2]
                        leaders[qi][2] += np.clip(dz, -SQUAD_SPD * dt, SQUAD_SPD * dt)
                        if abs(dz) < 0.05:
                            leaders[qi][2] = QUAD_CRUISE_ALT
                            quad_wp_idx[qi] += 1
                            quad_wp_age[qi] = 0
                            quad_dwell[qi] = 0
                            quad_site_best[qi] = None
                            quad_photo_taken[qi] = False
                            quad_phase[qi] = "transit"

                # Push updated leader positions to PID targets
                for i in range(nd):
                    target_pos[i] = leaders[i].copy()

                # Transition when ALL quads are done
                if all(quad_done):
                    quad_eval = evaluate_detector(
                        quad_pred_records, gt_xy, QUAD_MATCH_RADIUS
                    )
                    mission_state = SQUAD_BACK
                    banner(SQUAD_BACK, "All quads finished — returning home")
                    squad_back_step = 0
                    # Leaders already have correct XY (last site or QUAD_HOVER for
                    # no-site quads) and z was pulled to QUAD_CRUISE_ALT by the
                    # done-branch above.  Just clamp z to be safe — do NOT
                    # re-seed from obs, which adds PID-lag jitter.
                    for i in range(nd):
                        leaders[i][2] = QUAD_CRUISE_ALT
                        ret_wp_idx[i] = 0  # start from first return WP
                elif squad_go_step >= int(SQUAD_GO_TIMEOUT_SECS * ctrl_hz):
                    print(
                        f"  [Safety] SQUAD_GO timeout ({SQUAD_GO_TIMEOUT_SECS:.0f}s) — "
                        "forcing return-to-base phase"
                    )
                    quad_eval = evaluate_detector(
                        quad_pred_records, gt_xy, QUAD_MATCH_RADIUS
                    )
                    mission_state = SQUAD_BACK
                    squad_back_step = 0
                    for i in range(nd):
                        ret_wp_idx[i] = 0

            # ── SQUAD_BACK ───────────────────────────────────────────
            elif mission_state == SQUAD_BACK:

                squad_back_step += 1
                all_home = True

                for i in range(nd):
                    # Stagger return departures — hold until this drone's slot opens.
                    if squad_back_step < stagger_steps[i]:
                        target_pos[i] = leaders[i].copy()
                        all_home = False
                        continue

                    # Follow the planned return path (corridor WP → home),
                    # or fly directly home when no corridor was assigned.
                    ret_path = ret_wps_plan[i]  # numpy array [M, 3]
                    if len(ret_path) > 0 and ret_wp_idx[i] < len(ret_path):
                        # Current return waypoint
                        ret_target = ret_path[ret_wp_idx[i]]
                        leaders[i], ret_here = step_leader(leaders[i], ret_target, dt)
                        if ret_here:
                            ret_wp_idx[i] += 1  # advance to next return WP
                    else:
                        # No more return WPs (or empty plan) — fly direct to hover pad
                        leaders[i], _ = step_leader(leaders[i], QUAD_HOVER[i], dt)

                    # Gate on actual drone position to prevent premature QUAD_LAND
                    dx = obs[i][0] - QUAD_HOVER[i][0]
                    dy = obs[i][1] - QUAD_HOVER[i][1]
                    if math.hypot(dx, dy) > HOME_REACHED_DIST:
                        all_home = False
                    target_pos[i] = leaders[i].copy()

                if all_home:
                    mission_state = QUAD_LAND
                    quad_land_step = 0
                    banner(QUAD_LAND, "All quads descending")
                elif squad_back_step >= int(SQUAD_BACK_TIMEOUT_SECS * ctrl_hz):
                    print(
                        f"  [Safety] SQUAD_BACK timeout ({SQUAD_BACK_TIMEOUT_SECS:.0f}s) — "
                        "proceeding to landing to complete mission"
                    )
                    mission_state = QUAD_LAND
                    quad_land_step = 0
                    banner(
                        QUAD_LAND, "Return timeout — descending at current positions"
                    )

            # ── QUAD_LAND ────────────────────────────────────────────
            elif mission_state == QUAD_LAND:

                alpha = min(1.0, quad_land_step / max(1, quad_land_max))
                z_now = QUAD_CRUISE_ALT * (1.0 - alpha) + 0.15 * alpha
                for i in range(nd):
                    target_pos[i] = np.array(
                        [QUAD_HOVER[i][0], QUAD_HOVER[i][1], z_now]
                    )
                quad_land_step += 1

                if quad_land_step >= quad_land_max:
                    peaks = heatmap.find_peaks()
                    print_report(
                        peaks,
                        heatmap,
                        total_frames,
                        total_px,
                        photos_taken,
                        assignments,
                        fw_eval=fw_eval,
                        quad_eval=quad_eval,
                        battery=battery,
                        skipped=skipped,
                        gt_xy=gt_xy,
                    )
                    mission_state = DONE
                    break

            elif mission_state == DONE:
                break

            # ── Always refresh FW visual ──────────────────────────────
            place_fw(fw_ids, fw_pos, fw_dir, cid)

            # ── Fixed-wing battery drain ──────────────────────────────
            drain_fw = (
                mission_state in (FW_SCANNING, FW_RETURN, FW_LAND)
            ) or DRAIN_FW_OUTSIDE_FW_PHASES
            if drain_fw:
                fw_soc, _fw_p = fw_battery.update(
                    fw_pos=fw_pos,
                    fw_speed_mps=FW_SPEED,
                    dt=dt,
                    camera_active=fw_camera_active,
                )
                # Log FW telemetry uplink (unencrypted; FW uses a separate
                # radio link outside the quad ChaCha20 channel)
                log_msg(
                    comm_log,
                    step,
                    dt,
                    STATE_NAMES.get(mission_state, "UNKNOWN"),
                    "fw",
                    "ground_station",
                    "fw_telemetry_uplink",
                    {
                        "pos": [round(float(v), 3) for v in fw_pos],
                        "speed_mps": FW_SPEED,
                        "soc": round(fw_soc, 4),
                        "power_w": round(_fw_p, 2),
                        "camera_on": fw_camera_active,
                    },
                    encrypted=False,
                )

            # ── Quad-to-quad collision avoidance ─────────────────────
            for i in range(nd):
                push = np.zeros(2)
                for j in range(nd):
                    if j == i:
                        continue
                    dx = obs[i][0] - obs[j][0]
                    dy = obs[i][1] - obs[j][1]
                    d = math.hypot(dx, dy)
                    if 1e-6 < d < MIN_SEP:
                        push += AVOID_GAIN * (MIN_SEP - d) / d * np.array([dx, dy])
                n_ = np.linalg.norm(push)
                if n_ > MAX_PUSH:
                    push *= MAX_PUSH / n_
                target_pos[i, 0:2] = np.clip(
                    target_pos[i, 0:2] + push, -WORLD_LIMIT, WORLD_LIMIT
                )
                target_pos[i, 2] = np.clip(target_pos[i, 2], Z_MIN, Z_MAX)

            # ── Battery: drain every step (hovering at base costs energy too) ─
            # The energy planner reads battery.energy_j after FW lands, so it
            # automatically works with whatever is left after base hovering.
            bat_levels, _power = battery.update(obs, prev_vel, dt)
            battery.update_hud(obs, bat_levels, cid)

            # ── Uplink: encrypt + send telemetry for each airborne drone ─────
            if mission_state in (SQUAD_GO, SQUAD_BACK):
                for i in range(nd):
                    if obs[i][2] > 0.5:  # only flying drones
                        _pos = tuple(obs[i][0:3])
                        _vel = tuple(obs[i][10:13])
                        _bat = float(bat_levels[i])
                        comms.send_telemetry(i, pos=_pos, vel=_vel, battery_pct=_bat)
                        delta_j = comms.energy_consumed_j[i] - prev_crypto_j[i]
                        battery.energy_j[i] = max(0.0, battery.energy_j[i] - delta_j)
                        prev_crypto_j[i] = comms.energy_consumed_j[i]
                        # Record encrypted uplink in comm-log (sampled every
                        # ~1 s of sim time to avoid a multi-GB log file)
                        if bat_sample_ctr == 0:
                            log_msg(
                                comm_log,
                                step,
                                dt,
                                STATE_NAMES.get(mission_state, "UNKNOWN"),
                                f"quad_{i}",
                                "ground_station",
                                "telemetry_uplink",
                                {
                                    "pos": [round(float(v), 3) for v in _pos],
                                    "vel": [round(float(v), 3) for v in _vel],
                                    "battery_pct": round(_bat, 4),
                                    "crypto_energy_mj": round(
                                        comms.energy_consumed_j[i] * 1000, 6
                                    ),
                                },
                                encrypted=True,
                            )

            # Sample battery telemetry ~once per simulated second for graphs
            bat_sample_ctr += 1
            if bat_sample_ctr >= bat_sample_every:
                bat_sample_ctr = 0
                bat_history.append(
                    (step * dt, [bat_levels[i] * 100.0 for i in range(nd)])
                )

            # Track per-drone SQUAD_GO flight time
            if mission_state == SQUAD_GO:
                for i in range(nd):
                    if not quad_done[i] and squad_go_step > stagger_steps[i]:
                        if quad_flight_start_step[i] is None:
                            quad_flight_start_step[i] = step
                        quad_flight_end_step[i] = step

            # RTB checks only during active inspection
            if mission_state == SQUAD_GO:
                for i in range(nd):
                    battery.check_rtb(i, bat_levels[i], quad_done, leaders, obs)

            # ── PID ───────────────────────────────────────────────────
            for i, ctrl in enumerate(ctrls):
                raw = ctrl.computeControlFromState(
                    control_timestep=dt,
                    state=obs[i],
                    target_pos=target_pos[i],
                    target_rpy=np.zeros(3),
                    target_vel=np.zeros(3),
                    target_rpy_rates=np.zeros(3),
                )
                action[i, :] = raw[0]
                logger.log(drone=i, timestamp=step * dt, state=obs[i])

            env.step(action)

            # ── Viewport: FW during scan, Quad-0 from takeoff onwards ──
            cam_tgt = (
                fw_pos.tolist()
                if mission_state in (FW_SCANNING, FW_RETURN, FW_LAND)
                else list(obs[0][0:3])
            )
            update_viewport(cam_tgt, keys)
            env.render()
            if gui:
                sync(step, t0, env.CTRL_TIMESTEP)
            step += 1

    except KeyboardInterrupt:
        print("\n[Abort] Ctrl+C — partial report:")
        peaks = heatmap.find_peaks()
        heatmap.save_png()
        print_report(
            peaks,
            heatmap,
            total_frames,
            total_px,
            photos_taken,
            assignments,
            fw_eval=fw_eval,
            quad_eval=quad_eval,
            battery=battery,
            skipped=skipped,
            gt_xy=gt_xy,
        )

    finally:
        comms.print_report()
        saved_comm_path = None
        if comm_log is not None:
            saved_comm_path = comm_log.path
            comm_log.close()
        battery.cleanup_hud(cid)
        env.close()
        try:
            logger.save()
            logger.save_as_csv("mixed_fleet_mission")
        except Exception as _log_err:
            print(f"[Mission] logger save failed (non-fatal): {_log_err}")
        if plot:
            try:
                logger.plot()
            except Exception as _log_err:
                print(f"[Mission] logger.plot() failed (non-fatal): {_log_err}")
        print("[Mission] Logs saved.")
        if saved_comm_path:
            print(f"[Mission] Comm log: {saved_comm_path}")

        # ── Generate metrics report ───────────────────────────────────
        try:
            peaks_for_report = heatmap.find_peaks()
            _tp, _fp, _fn = 0, 0, 0
            _prec, _rec = 0.0, 0.0
            report_gt_xy = gt_xy if gt_xy else [(x, y) for x, y, *_ in WASTE_POSITIONS]
            n_gt_r = len(report_gt_xy)
            if peaks_for_report:
                _matched_gt = set()
                for _px, _py in peaks_for_report:
                    _dists = [math.hypot(_px - gx, _py - gy) for gx, gy in report_gt_xy]
                    _nd_idx = int(np.argmin(_dists))
                    if min(_dists) <= MATCH_RADIUS and _nd_idx not in _matched_gt:
                        _matched_gt.add(_nd_idx)
                        _tp += 1
                    else:
                        _fp += 1
                _fn = n_gt_r - len(_matched_gt)
                _prec = _tp / max(len(peaks_for_report), 1) * 100
                _rec = _tp / max(n_gt_r, 1) * 100

            final_bat_pct = [
                battery.energy_j[i] / BATTERY_CAPACITY_J * 100.0 for i in range(nd)
            ]
            _sim_elapsed_s = step * dt
            _wall_elapsed_s = time.time() - _mission_wall_t0

            _flight_times = []
            for _fi in range(nd):
                _fs = quad_flight_start_step[_fi]
                _fe = quad_flight_end_step[_fi]
                if _fs is not None and _fe is not None and _fe >= _fs:
                    _flight_times.append((_fe - _fs) * dt)
                else:
                    _flight_times.append(0.0)

            _det_backend_str = (
                "Roboflow plastic-waste" if _ROBOFLOW_MODEL is not None else "HSV cyan"
            )

            generate_metrics_plots(
                num_drones=nd,
                strategy=strategy,
                bat_history=bat_history,
                init_bat_pct=init_bat_pct,
                final_bat_pct=final_bat_pct,
                assignments=assignments,
                true_pos=_tp,
                false_pos=_fp,
                missed=_fn,
                precision=_prec,
                recall=_rec,
                photos_taken=photos_taken,
                sim_time_s=_sim_elapsed_s,
                wall_time_s=_wall_elapsed_s,
                flight_times_s=_flight_times,
                waste_positions=list(WASTE_POSITIONS),
                peaks_for_plot=peaks_for_report,
                fw_eval=fw_eval,
                quad_eval=quad_eval,
                gt_xy=report_gt_xy,
                detection_backend=_det_backend_str,
            )
        except Exception as _me:
            print(f"[Metrics] Plot generation failed: {_me}")
            import traceback

            traceback.print_exc()

        if gui:
            input("Press Enter to exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Battery pre-depletion examples:
  python v27.py --battery 0:50 2:30   # blue at 50%%, red at 30%%
  python v27.py --battery 1:20        # green at 20%% (likely triggers RTB early)

Strategy & fleet examples:
  python v27.py --num_drones 2 --strategy hungarian
  python v27.py --num_drones 3 --strategy genetic
  python v27.py --num_drones 4 --strategy worst_fit --battery 0:50 1:30
""",
    )
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    parser.add_argument(
        "--rf_conf",
        default=72,
        type=int,
        metavar="",
        help="Roboflow confidence threshold in percent (0-100).",
    )
    parser.add_argument(
        "--rf_overlap",
        default=30,
        type=int,
        metavar="",
        help="Roboflow overlap/NMS setting in percent (0-100).",
    )
    parser.add_argument(
        "--rf_classes",
        default="",
        type=str,
        metavar="",
        help="Comma-separated allowed class names for Roboflow filtering.",
    )
    parser.add_argument(
        "--num_drones",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        metavar="N",
        help="Number of quadcopters (1-4, default 4).",
    )
    parser.add_argument(
        "--strategy",
        default="worst_fit",
        choices=["worst_fit", "best_fit", "normalised_spare", "hungarian", "genetic"],
        metavar="NAME",
        help=(
            "Site-assignment strategy. Choices: "
            "worst_fit (default, load-balanced), "
            "best_fit (max spare Joules), "
            "normalised_spare (max spare %%), "
            "hungarian (optimal min-cost per round), "
            "genetic (GA, optimises coverage + balance). "
            "Example: --strategy hungarian"
        ),
    )
    parser.add_argument(
        "--battery",
        nargs="*",
        metavar="DRONE:PCT",
        default=None,
        help=(
            "Set starting battery level for specific drones. "
            "Format: DRONE_INDEX:PERCENT (0-100), space-separated. "
            "Unspecified drones start at 100%%. "
            "Examples: --battery 0:50 2:30   --battery 1:20"
        ),
    )
    parser.add_argument(
        "--comm-log",
        nargs="?",
        const=DEFAULT_COMM_LOG,
        default=None,
        metavar="PATH",
        help=(
            "Write a JSON mission communication log to PATH. "
            "Encrypted quad uplinks/downlinks are flagged with encrypted=true. "
            "If the flag is given without a path, writes to mission_comm_log.json "
            "in the script directory. "
            "Example: --comm-log   or   --comm-log my_log.json"
        ),
    )
    args = parser.parse_args()

    # Parse --battery "0:50 2:30" -> {0: 0.50, 2: 0.30}
    init_battery = None
    if args.battery:
        init_battery = {}
        for token in args.battery:
            try:
                idx_str, pct_str = token.split(":")
                idx = int(idx_str)
                pct = float(pct_str)
                if not (0.0 <= pct <= 100.0):
                    raise ValueError(f"percentage {pct} out of range 0-100")
                init_battery[idx] = pct / 100.0
            except ValueError as exc:
                parser.error(
                    f"Invalid --battery token '{token}': {exc}. "
                    "Expected format DRONE_INDEX:PERCENT, e.g. 0:75"
                )

    run(
        gui=args.gui,
        plot=args.plot,
        rf_conf=args.rf_conf,
        rf_overlap=args.rf_overlap,
        rf_classes=args.rf_classes,
        num_drones=args.num_drones,
        strategy=args.strategy,
        init_battery=init_battery,
        comm_log_path=args.comm_log,
    )
