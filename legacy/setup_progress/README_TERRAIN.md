# Terrain System - Quick Reference

## What's Been Implemented

### 1. Camera Detection (Disabled for Performance)
- ✅ Camera detection temporarily disabled in `swarm-mixed-fleet.py`
- ✅ Plane size adjusted to match terrain (50 units instead of 500)
- ✅ Simulation now runs smoothly without lag

### 2. OpenDroneMap Integration Tools
Three new tools for working with realistic terrain:

#### A. Example Terrain Generator (`generate_example_terrain.py`)
Generates 5 realistic terrain types without needing ODM data:

```bash
python generate_example_terrain.py
```

**Output:**
- `assets/terrain_desert_dunes.png` - Smooth desert dunes
- `assets/terrain_mountains.png` - Rocky peaks
- `assets/terrain_canyon.png` - Deep canyons
- `assets/terrain_hills.png` - Rolling hills
- `assets/terrain_valleys.png` - Plateaus

#### B. ODM Converter (`odm_to_heightmap.py`)
Converts real OpenDroneMap DEM files to heightmaps:

```bash
# Install dependencies first
pip install rasterio pillow numpy scipy

# Convert ODM output
python odm_to_heightmap.py --input path/to/dsm.tif --output assets/my_terrain.png --size 512
```

#### C. Documentation (`ODM_INTEGRATION.md`)
Complete guide covering:
- How to use example terrains
- Processing aerial images with OpenDroneMap
- Converting ODM output to heightmaps
- Adjusting terrain scales
- Troubleshooting

---

## Quick Start

### Option 1: Use Example Terrains (No ODM needed)

1. **Generate terrains:**
   ```bash
   cd gym_pybullet_drones/examples
   python generate_example_terrain.py
   ```

2. **Edit swarm-mixed-fleet.py** (line ~516):
   ```python
   # Change from:
   terrain_id = create_desert_terrain(
       heightmap_path="assets/desert_heightmap.png",
       texture_path="assets/desert_sand.png",
       terrain_scale=(0.15, 0.15, 3.0)
   )
   
   # To use different terrain:
   terrain_id = create_desert_terrain(
       heightmap_path="assets/terrain_mountains.png",  # Try different terrains!
       texture_path="assets/desert_sand.png",
       terrain_scale=(0.15, 0.15, 5.0)  # Adjust height
   )
   ```

3. **Run simulation:**
   ```bash
   python swarm-mixed-fleet.py --gui True
   ```

### Option 2: Use Real ODM Data

1. **Process aerial images with OpenDroneMap** (see `ODM_INTEGRATION.md`)

2. **Convert DEM to heightmap:**
   ```bash
   pip install rasterio
   python odm_to_heightmap.py --input odm_output/dsm.tif --output assets/real_terrain.png --size 512
   ```

3. **Use in simulation** (same as Option 1, step 2)

---

## Terrain Recommendations

| Terrain Type | File | Scale | Best For |
|--------------|------|-------|----------|
| Desert Dunes | `terrain_desert_dunes.png` | `(0.15, 0.15, 3.0)` | Smooth navigation testing |
| Mountains | `terrain_mountains.png` | `(0.15, 0.15, 5.0)` | Obstacle avoidance |
| Canyon | `terrain_canyon.png` | `(0.15, 0.15, 4.0)` | Valley navigation |
| Hills | `terrain_hills.png` | `(0.15, 0.15, 2.0)` | Gentle terrain testing |
| Valleys | `terrain_valleys.png` | `(0.15, 0.15, 3.5)` | Multi-level navigation |

---

## Understanding Terrain Scale

```python
terrain_scale = (x_scale, y_scale, z_scale)
```

- **x_scale, y_scale**: Horizontal size (meters per pixel)
  - Smaller = smaller terrain area
  - Larger = bigger terrain area
  
- **z_scale**: Vertical height multiplier
  - Smaller = flatter terrain
  - Larger = more dramatic elevation changes

**Example:**
```python
# 256x256 pixel heightmap with scale (0.15, 0.15, 3.0)
Physical size: 38.4m x 38.4m with max height of ~3.0m
```

---

## Files Created

### New Scripts
- `generate_example_terrain.py` - Generate test terrains
- `odm_to_heightmap.py` - Convert ODM DEM files
- `test_terrains.py` - Helper for testing terrains
- `ODM_INTEGRATION.md` - Complete integration guide
- `README_TERRAIN.md` - This file

### Generated Assets
- `assets/terrain_desert_dunes.png`
- `assets/terrain_mountains.png`
- `assets/terrain_canyon.png`
- `assets/terrain_hills.png`
- `assets/terrain_valleys.png`

### Modified Files
- `swarm-mixed-fleet.py`:
  - Camera detection disabled (lines ~656-660)
  - Plane size reduced to 50.0 (line ~490)

---

## FAQ

**Q: Can I use my own aerial drone photos?**
A: Yes! Process them with OpenDroneMap, then use `odm_to_heightmap.py` to convert the output.

**Q: The terrain looks stretched/wrong size**
A: Adjust the `plane_size` variable to match your terrain:
```python
terrain_pixels = 256
x_scale = 0.15
plane_size = terrain_pixels * x_scale * 1.2  # Slightly larger than terrain
```

**Q: Terrain is too flat/steep**
A: Adjust the `z_scale` parameter:
- Too flat: Increase z_scale (e.g., 3.0 → 5.0)
- Too steep: Decrease z_scale (e.g., 3.0 → 1.5)

**Q: Do I need OpenDroneMap?**
A: No! The example terrain generator creates realistic terrains without ODM. ODM is only needed if you want to use real aerial photos.

**Q: Can I re-enable camera detection?**
A: Yes, uncomment lines 656-660 in `swarm-mixed-fleet.py`. But be aware it will slow down the simulation.

---

## Next Steps

1. **Try different terrains** - Each has unique characteristics for testing
2. **Experiment with scales** - Find the right size for your use case
3. **Generate your own terrains** - Modify `generate_example_terrain.py` parameters
4. **Use real data** - Follow `ODM_INTEGRATION.md` to process aerial photos

---

## Support

For detailed documentation, see:
- **ODM Integration:** `ODM_INTEGRATION.md`
- **OpenDroneMap:** https://opendronemap.org/odm/
- **PyBullet:** https://pybullet.org/

---

*Created: 2026-01-28*
