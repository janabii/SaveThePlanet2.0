# synthetic_overlay.py
# ─────────────────────────────────────────────────────────────────────
# Plug-in module for the Save The Planet mission script.
# Augments Fixed-Wing camera frames with synthetic waste overlays
# BEFORE they are sent to Roboflow or HSV detection.
#
# IMPORTANT: This does NOT spawn 3D waste in PyBullet. Overlays appear only
# in the FW nadir camera bitmap (and in fw_frames/*.png if enabled).
#
# USAGE in your main mission script:
#
#   from synthetic_overlay import SyntheticOverlay
#
#   # Once, after env is created (after build_odm_terrain):
#   synth = SyntheticOverlay(
#       arena_m   = 34.0,        # must match your terrain size
#       fw_cam_w  = FW_CAM_W,
#       fw_cam_h  = FW_CAM_H,
#       fw_cam_fov= FW_CAM_FOV,
#   )
#
#   # Inside the FW_SCANNING detection block (fw_camera_rgb returns RGB):
#   #   rgb = fw_camera_rgb(fw_pos, fw_dir, cid)
#   #   bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#   #   synth.augment(bgr, fw_pos, fw_dir, FW_CAM_OFFSET)
#   #   rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#
# augment() expects a BGR uint8 canvas (OpenCV drawing convention).
# ─────────────────────────────────────────────────────────────────────

import math
import random
import numpy as np
import cv2


# ── Display sizes (pixels at FW_CAM_W=320, FW_CAM_H=240) ─────────────────────
DISPLAY_PX = {
    "bottle": {"half_len": 7, "half_r": 3},
    "garbage_bag": {"half_r": 6},
    "cardboard_box": {"half_w": 8, "half_h": 6},
    "cardboard_bag": {"half_w": 7, "half_h": 7},
}

BURIAL_PARAMS = {
    "surface": (1.00, 0.45, 1.00),
    "half_buried": (0.75, 0.30, 0.85),
    "mostly_buried": (0.50, 0.20, 0.65),
    "top_only": (0.55, 0.22, 0.65),
    "corner_peek": (0.45, 0.18, 0.55),
    "side_exposed": (0.80, 0.35, 0.90),
}

WASTE_URDFS = [
    {"name": "bottle", "burial_bias": ["surface", "half_buried", "mostly_buried", "side_exposed"], "scale_range": (0.85, 1.15)},
    {"name": "cardboard_bag", "burial_bias": ["surface", "surface", "half_buried", "top_only"], "scale_range": (1.35, 1.75)},
    {"name": "cardboard_box", "burial_bias": ["surface", "surface", "half_buried", "corner_peek"], "scale_range": (1.35, 1.75)},
    {"name": "garbage_bag", "burial_bias": ["surface", "surface", "half_buried"], "scale_range": (1.00, 1.30)},
]

BOTTLE_POSES = ["on_side", "upright", "neck_up", "base_up"]
BOTTLE_WEIGHTS = [0.40, 0.20, 0.20, 0.20]


def _rng(seed):
    return np.random.default_rng(int(seed) % (2**31))


def _gauss_mask(H, W, cx, cy, rx, ry, angle_deg=0.0):
    ys, xs = np.mgrid[0:H, 0:W]
    dx = (xs - cx).astype(np.float32)
    dy = (ys - cy).astype(np.float32)
    if angle_deg:
        th = math.radians(angle_deg)
        c, s = math.cos(th), math.sin(th)
        dx, dy = c * dx + s * dy, -s * dx + c * dy
    d2 = (dx / max(rx, 1)) ** 2 + (dy / max(ry, 1)) ** 2
    return np.exp(-0.5 * d2).astype(np.float32)


def _hard_ellipse(H, W, cx, cy, rx, ry, angle_deg=0.0):
    img = np.zeros((H, W), np.uint8)
    cv2.ellipse(
        img,
        (int(cx), int(cy)),
        (max(1, int(rx)), max(1, int(ry))),
        angle_deg,
        0,
        360,
        255,
        -1,
    )
    return img.astype(np.float32) / 255.0


def _hard_rect(H, W, cx, cy, w, h, angle_deg=0.0):
    img = np.zeros((H, W), np.uint8)
    pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))).astype(np.int32)
    cv2.fillConvexPoly(img, pts, 255)
    return img.astype(np.float32) / 255.0


def _noise(H, W, amp, seed):
    return _rng(seed).standard_normal((H, W)).astype(np.float32) * amp


def _blend(canvas, bgr, alpha_map):
    a = np.clip(alpha_map, 0, 1)[:, :, np.newaxis]
    c = np.array(bgr, dtype=np.float32).reshape(1, 1, 3)
    canvas[:] = np.clip(canvas.astype(np.float32) * (1 - a) + c * a, 0, 255).astype(np.uint8)


def _blend_img(canvas, img, alpha_map):
    a = np.clip(alpha_map, 0, 1)[:, :, np.newaxis]
    canvas[:] = np.clip(
        canvas.astype(np.float32) * (1 - a) + img.astype(np.float32) * a, 0, 255
    ).astype(np.uint8)


def _bbox(cx, cy, hw, hh, W, H):
    return (max(0, int(cx - hw)), max(0, int(cy - hh)), min(W - 1, int(cx + hw)), min(H - 1, int(cy + hh)))


def _draw_bottle(canvas, cx, cy, half_len, half_r, angle_deg, body_a, shadow_a, seed, pose="on_side"):
    H, W = canvas.shape[:2]
    r = _rng(seed)
    variants = [(255, 255, 255), (60, 200, 60), (200, 60, 60), (220, 200, 50), (60, 220, 220)]
    bgr = variants[int(r.integers(len(variants)))]
    bgr = tuple(int(np.clip(c * float(r.uniform(0.85, 1.0)), 0, 255)) for c in bgr)
    caps = [(30, 30, 210), (210, 40, 40), (30, 180, 30), (20, 200, 230), (180, 20, 180)]
    cap_bgr = caps[int(r.integers(len(caps)))]
    sdx = max(1, int(half_r * 0.5))
    sdy = max(1, int(half_r * 0.25))

    if pose == "upright":
        s = _gauss_mask(H, W, cx + sdx, cy + sdy, half_r * 1.15, half_r * 1.15) * shadow_a
        _blend(canvas, (20, 20, 20), s)
        body = _hard_ellipse(H, W, cx, cy, half_r, half_r) * body_a
        _blend(canvas, bgr, body)
        hi = (
            _gauss_mask(
                H,
                W,
                cx - int(half_r * 0.30),
                cy - int(half_r * 0.30),
                max(1, half_r * 0.35),
                max(1, half_r * 0.35),
            )
            * 0.85
            * body_a
        )
        _blend(canvas, (255, 255, 255), hi)
        cap = _hard_ellipse(H, W, cx, cy, max(2, int(half_r * 0.55)), max(2, int(half_r * 0.55))) * body_a
        _blend(canvas, cap_bgr, cap)
        pad = int(half_r * 1.2) + sdx
        return _bbox(cx, cy, pad, pad, W, H)

    if pose == "neck_up":
        stub_len = max(4, int(half_len * 0.38))
        stub_r = max(2, int(half_r * 0.48))
        s = _gauss_mask(H, W, cx + sdx, cy + sdy, stub_r * 1.1, stub_len * 0.7, angle_deg) * shadow_a * 0.6
        _blend(canvas, (20, 20, 20), s)
        body = _hard_ellipse(H, W, cx, cy, stub_r, stub_len, angle_deg) * body_a
        _blend(canvas, bgr, body)
        cap_ox = int(math.cos(math.radians(angle_deg + 90)) * stub_len * 0.72)
        cap_oy = int(math.sin(math.radians(angle_deg + 90)) * stub_len * 0.72)
        cap = _hard_ellipse(H, W, cx + cap_ox, cy + cap_oy, max(2, int(stub_r * 0.80)), max(2, int(stub_r * 0.80))) * body_a
        _blend(canvas, cap_bgr, cap)
        return _bbox(cx, cy, stub_r + sdx + 2, stub_len + sdy + 2, W, H)

    if pose == "base_up":
        base_r = int(half_r * 1.05)
        s = _gauss_mask(H, W, cx + sdx, cy + sdy, base_r * 1.2, base_r * 1.2) * shadow_a
        _blend(canvas, (20, 20, 20), s)
        body = _hard_ellipse(H, W, cx, cy, base_r, base_r) * body_a
        _blend(canvas, bgr, body)
        hi = (
            _gauss_mask(H, W, cx - int(base_r * 0.28), cy - int(base_r * 0.28), max(1, base_r * 0.28), max(1, base_r * 0.28))
            * 0.75
            * body_a
        )
        _blend(canvas, (255, 255, 255), hi)
        pad = int(base_r * 1.2) + sdx
        return _bbox(cx, cy, pad, pad, W, H)

    # on_side
    body_w = half_len
    body_h = int(half_r * 1.6)
    s = _gauss_mask(H, W, cx + sdx, cy + sdy, body_w * 0.95, body_h * 1.1, angle_deg) * shadow_a
    _blend(canvas, (20, 20, 20), s)
    body = _hard_ellipse(H, W, cx, cy, body_w, body_h, angle_deg) * body_a
    _blend(canvas, bgr, body)
    hi_ox = int(math.cos(math.radians(angle_deg + 90)) * body_h * 0.32)
    hi_oy = int(math.sin(math.radians(angle_deg + 90)) * body_h * 0.32)
    hi = _gauss_mask(H, W, cx + hi_ox, cy + hi_oy, body_w * 0.80, max(1, body_h * 0.28), angle_deg) * 0.80 * body_a
    _blend(canvas, (255, 255, 255), hi)
    cap_ox = int(math.cos(math.radians(angle_deg)) * body_w * 0.88)
    cap_oy = int(math.sin(math.radians(angle_deg)) * body_w * 0.88)
    cap_r = max(2, int(body_h * 0.72))
    cap = _hard_ellipse(H, W, cx + cap_ox, cy + cap_oy, cap_r, cap_r) * body_a
    _blend(canvas, cap_bgr, cap)
    return _bbox(cx, cy, body_w + sdx + 2, body_h + sdy + 2, W, H)


def _draw_garbage_bag(canvas, cx, cy, half_r, angle_deg, body_a, shadow_a, seed):
    H, W = canvas.shape[:2]
    r = _rng(seed)
    colours = [(12, 12, 12), (18, 26, 18), (20, 20, 32), (28, 24, 24), (38, 38, 38)]
    bag_bgr = np.array(colours[int(r.integers(len(colours)))], np.float32)
    rx = half_r * float(r.uniform(0.88, 1.08))
    ry = half_r * float(r.uniform(0.70, 0.86))
    soff = max(1, int(half_r * 0.25))
    s = _gauss_mask(H, W, cx + soff, cy + soff, min(rx * 1.12, rx + 4), min(ry * 1.12, ry + 4)) * shadow_a
    _blend(canvas, (12, 12, 12), s)
    body = _hard_ellipse(H, W, cx, cy, int(rx), int(ry), angle_deg) * body_a
    _blend(canvas, bag_bgr.tolist(), body)
    dome = _gauss_mask(H, W, cx, cy, rx * 0.72, ry * 0.72, angle_deg) * body_a * 0.5
    _blend(canvas, np.clip(bag_bgr + 10, 0, 255).tolist(), dome)
    hi = _gauss_mask(H, W, cx - int(rx * 0.25), cy - int(ry * 0.22), max(1, rx * 0.16), max(1, ry * 0.16)) * 0.40 * body_a
    _blend(canvas, (68, 68, 68), hi)
    return _bbox(cx, cy, rx + soff + 1, ry + soff + 1, W, H)


def _draw_cardboard_box(canvas, cx, cy, half_w, half_h, angle_deg, body_a, shadow_a, seed):
    H, W = canvas.shape[:2]
    r = _rng(seed)
    palette = [(75, 115, 165), (65, 100, 145), (55, 85, 120), (90, 130, 178), (60, 95, 135)]
    base = np.array(palette[int(r.integers(len(palette)))], np.float32)
    base = np.clip(base * float(r.uniform(0.85, 1.10)), 0, 255)
    w, h = half_w * 2, half_h * 2
    soff = max(1, int(min(w, h) * 0.18))
    s = _hard_rect(H, W, cx + soff, cy + soff, w * 1.12, h * 1.12, angle_deg) * shadow_a
    _blend(canvas, (18, 18, 18), s)
    face = _hard_rect(H, W, cx, cy, w, h, angle_deg)
    fi = np.zeros((H, W, 3), np.float32)
    n = _noise(H, W, 10, seed + 2)
    for c in range(3):
        fi[:, :, c] = base[c] + n
    fi = np.clip(fi, 0, 255)
    _blend_img(canvas, fi.astype(np.uint8), face * body_a)
    crease = _gauss_mask(H, W, cx, cy, max(1, half_w * 0.06), half_h * 0.88, angle_deg) * 0.45 * body_a
    _blend(canvas, np.clip(base * 0.50, 0, 255).tolist(), crease)
    return _bbox(cx, cy, half_w + soff + 1, half_h + soff + 1, W, H)


def _draw_cardboard_bag(canvas, cx, cy, half_w, half_h, angle_deg, body_a, shadow_a, seed):
    H, W = canvas.shape[:2]
    r = _rng(seed)
    palette = [(80, 118, 162), (68, 102, 142), (58, 88, 122), (95, 135, 180), (72, 110, 155)]
    base = np.array(palette[int(r.integers(len(palette)))], np.float32)
    base = np.clip(base * float(r.uniform(0.85, 1.10)), 0, 255)
    w, h = half_w * 2, half_h * 2
    soff = max(1, int(min(w, h) * 0.16))
    s = _hard_rect(H, W, cx + soff, cy + soff, w * 1.08, h * 1.08, angle_deg) * shadow_a
    _blend(canvas, (18, 18, 18), s)
    face = _hard_rect(H, W, cx, cy, w, h, angle_deg)
    fi = np.zeros((H, W, 3), np.float32)
    n = _noise(H, W, 9, seed + 5)
    for c in range(3):
        fi[:, :, c] = base[c] + n
    fi = np.clip(fi, 0, 255)
    _blend_img(canvas, fi.astype(np.uint8), face * body_a)
    hcx = cx + int(math.cos(math.radians(angle_deg + 90)) * half_h * 0.72)
    hcy = cy + int(math.sin(math.radians(angle_deg + 90)) * half_h * 0.72)
    handle = _gauss_mask(H, W, hcx, hcy, max(1, half_w * 0.48), max(1, half_h * 0.09), angle_deg) * 0.75 * body_a
    _blend(canvas, np.clip(base * 0.42, 0, 255).tolist(), handle)
    return _bbox(cx, cy, half_w + soff + 1, half_h + soff + 1, W, H)


class SyntheticOverlay:
    """
    Scatters synthetic waste in world XY and draws items visible in each FW frame.
    augment() expects BGR uint8 (convert from RGB before calling).
    """

    def __init__(
        self,
        arena_m=34.0,
        fw_cam_w=320,
        fw_cam_h=240,
        fw_cam_fov=90.0,
        n_items=40,
        seed=42,
    ):
        self.arena_m = arena_m
        self.fw_cam_w = fw_cam_w
        self.fw_cam_h = fw_cam_h
        self.fw_cam_fov = fw_cam_fov
        self.items = []

        rng = random.Random(seed)
        half = arena_m / 2.0
        existing = []

        for _ in range(n_items):
            placed = False
            for _try in range(60):
                x = rng.uniform(-half + 1.0, half - 1.0)
                y = rng.uniform(-half + 1.0, half - 1.0)
                if any(math.hypot(x - ex, y - ey) < 3.5 for ex, ey in existing):
                    continue
                kind = rng.choice(WASTE_URDFS)
                mode = rng.choice(kind["burial_bias"])
                scale = rng.uniform(*kind["scale_range"])
                yaw = rng.uniform(-math.pi, math.pi)
                pose = "on_side"
                if kind["name"] == "bottle":
                    pose = random.choices(BOTTLE_POSES, weights=BOTTLE_WEIGHTS)[0]
                    if pose == "upright":
                        mode = rng.choice(["surface", "surface", "half_buried"])
                    elif pose in ("neck_up", "base_up"):
                        mode = rng.choice(["half_buried", "mostly_buried"])

                self.items.append(
                    {
                        "name": kind["name"],
                        "world_x": x,
                        "world_y": y,
                        "yaw": yaw,
                        "burial_mode": mode,
                        "obj_scale": scale,
                        "bottle_pose": pose,
                    }
                )
                existing.append((x, y))
                placed = True
                break
            if not placed:
                pass

        print(f"[SyntheticOverlay] {len(self.items)} items placed across {arena_m}m arena")

    def _footprint(self, fw_pos, fw_dir, fw_cam_offset=-0.065):
        cam_h = fw_pos[2] + fw_cam_offset
        half_fov_rad = math.radians(self.fw_cam_fov) / 2.0
        half_fwd = cam_h * math.tan(half_fov_rad)
        half_rgt = half_fwd * (self.fw_cam_w / self.fw_cam_h)
        margin = max(half_fwd, half_rgt) * 1.1
        return (fw_pos[0] - margin, fw_pos[0] + margin, fw_pos[1] - margin, fw_pos[1] + margin)

    def _world_to_pixel(self, wx, wy, fw_pos, fw_dir, fw_cam_offset=-0.065):
        cam_h = fw_pos[2] + fw_cam_offset
        half_fov_rad = math.radians(self.fw_cam_fov) / 2.0
        half_fwd = cam_h * math.tan(half_fov_rad)
        half_rgt = half_fwd * (self.fw_cam_w / self.fw_cam_h)

        yaw = math.atan2(fw_dir[1], fw_dir[0])
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        rgt = np.array([math.sin(yaw), -math.cos(yaw)])

        dx = wx - fw_pos[0]
        dy = wy - fw_pos[1]
        offset = np.array([dx, dy])

        u = np.dot(offset, fwd) / half_fwd
        v = np.dot(offset, rgt) / half_rgt

        px = int((v / 2.0 + 0.5) * self.fw_cam_w)
        py = int((-u / 2.0 + 0.5) * self.fw_cam_h)

        if 0 <= px < self.fw_cam_w and 0 <= py < self.fw_cam_h:
            return px, py
        return None

    def augment(self, rgb_bgr, fw_pos, fw_dir, fw_cam_offset=-0.065):
        """
        Render synthetic waste on an FW frame. Canvas must be BGR uint8 (in-place).
        """
        H, W = rgb_bgr.shape[:2]
        canvas = rgb_bgr

        xmin, xmax, ymin, ymax = self._footprint(fw_pos, fw_dir, fw_cam_offset)

        for item in self.items:
            wx = item["world_x"]
            wy = item["world_y"]

            if not (xmin <= wx <= xmax and ymin <= wy <= ymax):
                continue

            pix = self._world_to_pixel(wx, wy, fw_pos, fw_dir, fw_cam_offset)
            if pix is None:
                continue
            cx, cy = pix

            name = item["name"]
            angle_deg = -math.degrees(item["yaw"])
            mode = item["burial_mode"]
            oscale = float(item["obj_scale"])
            body_a, shadow_a, sfrac = BURIAL_PARAMS.get(mode, (1.0, 0.45, 1.0))
            seed = abs(hash((round(wx, 2), round(wy, 2), name))) % (2**31)

            S = sfrac * oscale

            if name == "bottle":
                dp = DISPLAY_PX["bottle"]
                pose = item.get("bottle_pose", "on_side")
                _draw_bottle(
                    canvas,
                    cx,
                    cy,
                    dp["half_len"] * S,
                    dp["half_r"] * S,
                    angle_deg,
                    body_a,
                    shadow_a,
                    seed,
                    pose=pose,
                )

            elif name == "garbage_bag":
                dp = DISPLAY_PX["garbage_bag"]
                _draw_garbage_bag(canvas, cx, cy, dp["half_r"] * S, angle_deg, body_a, shadow_a, seed)

            elif name == "cardboard_box":
                dp = DISPLAY_PX["cardboard_box"]
                _draw_cardboard_box(canvas, cx, cy, dp["half_w"] * S, dp["half_h"] * S, angle_deg, body_a, shadow_a, seed)

            elif name == "cardboard_bag":
                dp = DISPLAY_PX["cardboard_bag"]
                _draw_cardboard_bag(canvas, cx, cy, dp["half_w"] * S, dp["half_h"] * S, angle_deg, body_a, shadow_a, seed)

        return canvas
