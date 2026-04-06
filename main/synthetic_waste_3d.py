"""
Spawn synthetic waste as fixed URDF bodies on the ODM heightfield (visible in PyBullet GUI).
Mirrors synthetic_waste_capture.py spawn logic, without texture-crop capture.
"""

from __future__ import annotations

import math
import os
import random
import pybullet as p

# Same URDFs as synthetic_waste_capture.py (files live next to v20.py)
WASTE_URDFS = [
    {
        "name": "bottle",
        "file": "bottle.urdf",
        "half_height": 0.085,
        "footprint": 0.04,
        "burial_bias": ["surface", "half_buried", "mostly_buried", "side_exposed"],
        "scale_range": (0.85, 1.15),
    },
    {
        "name": "cardboard_bag",
        "file": "cardboard_bag.urdf",
        "half_height": 0.10,
        "footprint": 0.22,
        "burial_bias": ["surface", "surface", "half_buried", "top_only"],
        "scale_range": (1.35, 1.75),
    },
    {
        "name": "cardboard_box",
        "file": "cardboard_box.urdf",
        "half_height": 0.09,
        "footprint": 0.24,
        "burial_bias": ["surface", "surface", "half_buried", "corner_peek", "top_only"],
        "scale_range": (1.35, 1.75),
    },
    {
        "name": "garbage_bag",
        "file": "garbage_bag.urdf",
        "half_height": 0.18,
        "footprint": 0.24,
        "burial_bias": ["surface", "surface", "half_buried"],
        "scale_range": (1.00, 1.30),
    },
]

BOTTLE_POSES = ["on_side", "upright", "neck_up", "base_up"]
BOTTLE_WEIGHTS = [0.40, 0.20, 0.20, 0.20]

MAX_SPAWN_TRIES = 50
MIN_WASTE_GAP = 3.5


def _terrain_hit_z(x, y, terrain_id, cid):
    hit = p.rayTest([x, y, 200.0], [x, y, -200.0], physicsClientId=cid)
    if not hit:
        return None
    h = hit[0]
    if h[0] == terrain_id:
        return float(h[3][2])
    return None


def _far_from_existing(x, y, existing_xy, min_gap):
    for ex, ey in existing_xy:
        if (x - ex) ** 2 + (y - ey) ** 2 < min_gap**2:
            return False
    return True


def _placement_pose(kind, mode, ground_z):
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
    return z, p.getQuaternionFromEuler([roll, pitch, yaw]), yaw


def _tint_all_links(body_id, rgba, cid):
    p.changeVisualShape(body_id, -1, rgbaColor=rgba, physicsClientId=cid)
    for j in range(p.getNumJoints(body_id, physicsClientId=cid)):
        p.changeVisualShape(body_id, j, rgbaColor=rgba, physicsClientId=cid)


def spawn_synthetic_waste_urdf(
    cid,
    terrain_id,
    urdf_dir,
    *,
    arena_half_m=17.0,
    n_waste=40,
    seed=42,
    exclusion_xy=None,
    margin_m=2.5,
):
    """
    Place n_waste URDF props on the heightfield. exclusion_xy: list of (x,y) to keep
    clear (e.g. GT cube sites, base); margin_m is min distance from those points.
    Returns list of PyBullet body unique ids.
    """
    exclusion_xy = exclusion_xy or []
    rng = random.Random(seed)
    existing_xy = list(exclusion_xy)
    body_ids = []

    xmin, xmax = -arena_half_m + margin_m, arena_half_m - margin_m
    ymin, ymax = -arena_half_m + margin_m, arena_half_m - margin_m

    cluster_seed = None
    placed = 0

    for _ in range(n_waste):
        spawned = False
        for _try in range(MAX_SPAWN_TRIES):
            kind = rng.choice(WASTE_URDFS)
            if cluster_seed is not None and rng.random() < 0.30:
                x = cluster_seed[0] + rng.uniform(-6.0, 6.0)
                y = cluster_seed[1] + rng.uniform(-6.0, 6.0)
            else:
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)

            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue

            min_gap = max(MIN_WASTE_GAP, kind["footprint"] * 1.6)
            if not _far_from_existing(x, y, existing_xy, min_gap):
                continue

            ground_z = _terrain_hit_z(x, y, terrain_id, cid)
            if ground_z is None:
                continue

            mode = rng.choice(kind["burial_bias"])
            scale = rng.uniform(kind["scale_range"][0], kind["scale_range"][1])

            bottle_pose = "on_side"
            if kind["name"] == "bottle":
                bottle_pose = rng.choices(BOTTLE_POSES, weights=BOTTLE_WEIGHTS, k=1)[0]
                if bottle_pose == "upright":
                    mode = rng.choice(["surface", "surface", "half_buried"])
                elif bottle_pose in ("neck_up", "base_up"):
                    mode = rng.choice(["half_buried", "mostly_buried"])

            z, quat, _yaw = _placement_pose(kind, mode, ground_z)

            urdf_path = os.path.join(urdf_dir, kind["file"])
            if not os.path.isfile(urdf_path):
                print(f"[SyntheticWaste3D] Missing URDF: {urdf_path}")
                continue

            try:
                bid = p.loadURDF(
                    urdf_path,
                    basePosition=[x, y, z],
                    baseOrientation=quat,
                    useFixedBase=True,
                    globalScaling=scale,
                    physicsClientId=cid,
                )
                _tint_all_links(bid, [1, 1, 1, 1.0], cid)
                body_ids.append(bid)
                existing_xy.append((x, y))
                if cluster_seed is None or rng.random() < 0.18:
                    cluster_seed = (x, y)
                placed += 1
                spawned = True
                break
            except Exception as e:
                print(f"[SyntheticWaste3D] loadURDF failed {kind['file']}: {e}")
                continue

        if not spawned:
            pass

    print(f"[SyntheticWaste3D] Placed {placed} URDF waste props on terrain (requested {n_waste}).")
    return body_ids
