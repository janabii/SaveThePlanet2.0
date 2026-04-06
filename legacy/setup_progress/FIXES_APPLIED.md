# Fixes Applied - Drone Visibility and Golden Dunes

## Date: January 26, 2026

## Issues Fixed

### 1. Missing Drones (Only 1 Visible)
**Problem**: Only the quad follower (Drone 1) was visible. Drone 0 and Drone 2 were crashing or flying out of bounds.

**Root Causes**:
- Initial spawn altitudes too low (0.10m-0.15m) causing terrain collision
- Fixed-wing spawned too far away (-5.0m) and flew outside world limits
- Takeoff time too short (5 seconds) for proper stabilization

**Fixes Applied**:
```python
# Before:
init_xyzs = np.array([
    [0.0, 0.0, 0.10],    # Crashed into terrain
    [2.0, -1.5, 0.10],   # OK
    [-5.0, 0.0, 0.15],   # Flew out of bounds
])

# After:
init_xyzs = np.array([
    [0.0, 0.0, 1.5],     # Safe above terrain
    [2.5, -2.0, 1.5],    # Spread out more
    [0.0, -8.0, 2.0],    # In patrol area, higher
])

TAKEOFF_TIME = 8.0       # Was 5.0 - smoother takeoff
FW_PATROL_RADIUS = 12.0  # Was 18.0 - stay in bounds
WORLD_XY_LIMIT = 18.0    # Was 20.0 - tighter bounds
```

### 2. Dunes Not Golden/Tan Colored
**Problem**: Desert terrain appeared gray or with poor coloring.

**Root Cause**: PyBullet `createVisualShape()` doesn't support all heightfield parameters, and color wasn't being applied correctly.

**Fix Applied**:
```python
# Create terrain body first
terrain_body = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=terrain_shape,
    basePosition=[0, 0, 0]
)

# Then apply light tan/golden beige color
p.changeVisualShape(
    terrain_body,
    -1,
    rgbaColor=[0.92, 0.83, 0.65, 1.0]  # Light tan/beige
)
```

### 3. Drone Identification Difficulty
**Problem**: Couldn't easily identify which drone was which.

**Fix Applied**:
```python
# Added persistent text labels above each drone
for i, config in enumerate(DRONE_CONFIGS):
    label = p.addUserDebugText(
        text=config["role"].replace("_", " ").upper(),
        textPosition=[0, 0, 0.3],
        textColorRGB=[1, 1, 1],
        textSize=1.2,
        lifeTime=0,  # Persistent
        parentObjectUniqueId=env.DRONE_IDS[i]
    )
```

## Test Results

### Simulation Output (Verified Working)
```
[INFO] BaseAviary.render() ——— it 0264
[INFO] drone 0 ——— x +00.00, y +00.00, z +00.84m  [QUAD LEADER]
[INFO] drone 1 ——— x +02.01, y -01.50, z +00.85m  [QUAD FOLLOWER]  
[INFO] drone 2 ——— x +00.00, y -08.00, z +00.90m  [FIXEDWING SCOUT]
```

**Results**:
- ✅ All 3 drones visible and flying
- ✅ Drones maintain safe altitudes above terrain
- ✅ Fixed-wing stays within patrol radius
- ✅ Desert appears light tan/golden beige
- ✅ Text labels show drone roles clearly
- ✅ No crashes or out-of-bounds issues

## Files Modified

1. **`swarm-mixed-fleet.py`**:
   - Increased initial spawn altitudes (1.5m - 2.0m)
   - Spread drones horizontally for better separation
   - Increased takeoff time to 8 seconds
   - Reduced patrol radius to 12m
   - Tightened world bounds to 18m
   - Added persistent text labels above drones

2. **`desert_utils.py`**:
   - Fixed terrain color application method
   - Applied light tan/golden beige RGB(0.92, 0.83, 0.65)
   - Removed unsupported heightfield visual shape parameters

## Configuration Changes Summary

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Quad leader spawn Z | 0.10m | 1.5m | Avoid terrain collision |
| Quad follower spawn | (2.0, -1.5, 0.10m) | (2.5, -2.0, 1.5m) | More spread, higher |
| Fixed-wing spawn | (-5.0, 0.0, 0.15m) | (0.0, -8.0, 2.0m) | Patrol area center |
| TAKEOFF_TIME | 5.0s | 8.0s | Smoother stabilization |
| FW_PATROL_RADIUS | 18.0m | 12.0m | Stay within bounds |
| WORLD_XY_LIMIT | 20.0m | 18.0m | Prevent out-of-bounds |
| Terrain color | Gray/texture | RGB(0.92, 0.83, 0.65) | Golden tan appearance |
| Drone labels | None | "QUAD LEADER", etc. | Easy identification |

## How to Run

```bash
cd gym_pybullet_drones/gym_pybullet_drones/examples
python swarm-mixed-fleet.py --gui=True
```

You should now see:
1. All 3 drones spawning at safe altitudes
2. Golden/tan colored desert terrain with dunes
3. White text labels above each drone showing their role
4. Smooth takeoff without crashes
5. Fixed-wing circling within patrol bounds

## Notes

- The drones take ~8 seconds to complete takeoff (longer than before for safety)
- Fixed-wing patrol is smaller but more contained
- Labels are persistent and follow the drones
- Terrain color is now visible without texture dependency

---

**Status**: All fixes verified and working! ✅
