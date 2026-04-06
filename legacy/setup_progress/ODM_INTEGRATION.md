# OpenDroneMap Integration Guide

This guide explains how to use OpenDroneMap (ODM) terrain data with the PyBullet drone simulator.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Using Example Terrains](#using-example-terrains)
3. [Processing Real Aerial Images with ODM](#processing-real-aerial-images-with-odm)
4. [Converting ODM Output to Heightmaps](#converting-odm-output-to-heightmaps)
5. [Using Heightmaps in Simulation](#using-heightmaps-in-simulation)
6. [Adjusting Terrain Scale](#adjusting-terrain-scale)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option 1: Use Pre-Generated Example Terrains

Generate example terrains without needing ODM:

```bash
cd gym_pybullet_drones/examples
python generate_example_terrain.py
```

This creates 5 example heightmaps in the `assets/` folder:
- `terrain_desert_dunes.png` - Smooth desert dunes
- `terrain_mountains.png` - Rocky mountain peaks
- `terrain_canyon.png` - Canyon with valleys
- `terrain_hills.png` - Gentle rolling hills
- `terrain_valleys.png` - Plateaus and valleys

### Option 2: Convert ODM Output

If you have ODM DEM files:

```bash
python odm_to_heightmap.py --input path/to/dsm.tif --output assets/my_terrain.png --size 512
```

---

## Using Example Terrains

### 1. Generate Example Terrains

```bash
python generate_example_terrain.py
```

### 2. Modify Your Simulation Script

Edit `swarm-mixed-fleet.py` (or your simulation file):

```python
# Replace the existing create_desert_terrain call with:
terrain_id = create_desert_terrain(
    heightmap_path="assets/terrain_desert_dunes.png",  # Choose your terrain
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.15, 0.15, 3.0)  # Adjust scale as needed
)
```

### 3. Run Simulation

```bash
python swarm-mixed-fleet.py --gui True
```

### Terrain Types and Recommended Scales

| Terrain Type | Heightmap File | Recommended Scale | Description |
|--------------|----------------|-------------------|-------------|
| Desert Dunes | `terrain_desert_dunes.png` | `(0.15, 0.15, 3.0)` | Smooth rolling sand dunes |
| Mountains | `terrain_mountains.png` | `(0.15, 0.15, 5.0)` | Rocky peaks with ridges |
| Canyon | `terrain_canyon.png` | `(0.15, 0.15, 4.0)` | Deep valleys and plateaus |
| Hills | `terrain_hills.png` | `(0.15, 0.15, 2.0)` | Gentle rolling terrain |
| Valleys | `terrain_valleys.png` | `(0.15, 0.15, 3.5)` | Stepped plateaus |

---

## Processing Real Aerial Images with ODM

### Prerequisites

- Aerial images from drone (at least 15-20 images with 70%+ overlap)
- OpenDroneMap installed (Docker recommended)

### Method 1: Using Docker (Recommended)

```bash
# Create project directory
mkdir ~/odm_projects
cd ~/odm_projects

# Create image directory
mkdir my_flight/images
# Copy your aerial images to my_flight/images/

# Run ODM processing
docker run -ti --rm \
  -v $(pwd)/my_flight:/datasets/my_flight \
  opendronemap/odm \
  --project-path /datasets \
  my_flight

# Output will be in: my_flight/odm_dem/dsm.tif
```

### Method 2: Using WebODM (GUI)

1. Install WebODM: https://www.opendronemap.org/webodm/
2. Start WebODM: `./webodm.sh start`
3. Open browser: http://localhost:8000
4. Create new project and upload images
5. Start processing
6. Download DSM from outputs

### Method 3: Using PyODM (Python API)

```python
from pyodm import Node

# Connect to ODM node
node = Node('localhost', 3000)

# Create task
task = node.create_task([
    'image1.jpg',
    'image2.jpg',
    # ... more images
])

# Wait for processing
task.wait_for_completion()

# Download DSM
task.download_asset("dsm.tif", "output/dsm.tif")
```

### ODM Output Files

ODM generates several outputs in `odm_dem/`:
- **dsm.tif** - Digital Surface Model (includes buildings, trees, etc.)
- **dtm.tif** - Digital Terrain Model (ground only)

For drone simulation, **use DSM** to include all obstacles.

---

## Converting ODM Output to Heightmaps

### Basic Conversion

```bash
python odm_to_heightmap.py \
  --input path/to/dsm.tif \
  --output assets/my_terrain.png \
  --size 256
```

### Parameters

- `--input` or `-i`: Path to ODM DEM file (dsm.tif or dtm.tif)
- `--output` or `-o`: Output PNG heightmap path
- `--size` or `-s`: Output resolution (pixels, square). Default: 256
  - 256x256: Fast, lower detail
  - 512x512: Good balance
  - 1024x1024: High detail, slower performance
- `--smooth`: Gaussian smoothing (optional)
  - 0.0: No smoothing (default)
  - 1.0-2.0: Light smoothing
  - 3.0-5.0: Heavy smoothing

### Example with Smoothing

```bash
python odm_to_heightmap.py \
  --input odm_dem/dsm.tif \
  --output assets/smooth_terrain.png \
  --size 512 \
  --smooth 2.0
```

### Installing Dependencies

```bash
pip install rasterio pillow numpy scipy
```

---

## Using Heightmaps in Simulation

### 1. Load Terrain in Simulation

Edit your simulation script:

```python
from desert_utils import create_desert_terrain

# Load custom terrain
terrain_id = create_desert_terrain(
    heightmap_path="assets/my_terrain.png",
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.15, 0.15, 3.0)
)
```

### 2. Understanding Terrain Scale

The `terrain_scale` parameter controls terrain dimensions:

```python
terrain_scale = (x_scale, y_scale, z_scale)
```

- **x_scale, y_scale**: Horizontal scaling (meters per pixel)
- **z_scale**: Vertical scaling (height multiplier)

**Example calculations:**
```
Heightmap: 256x256 pixels
x_scale = 0.15
y_scale = 0.15
z_scale = 3.0

Physical terrain size:
  Width = 256 * 0.15 = 38.4 meters
  Depth = 256 * 0.15 = 38.4 meters
  Max Height = 255 * 3.0 / 255 = 3.0 meters (if brightest pixel is 255)
```

### 3. Match Plane Size to Terrain

In `swarm-mixed-fleet.py`, adjust plane size:

```python
# Calculate terrain size
heightmap_pixels = 256  # Your heightmap resolution
x_scale = 0.15
terrain_size = heightmap_pixels * x_scale  # e.g., 38.4 meters

# Set plane slightly larger
plane_size = terrain_size * 1.2  # e.g., 46 meters
```

---

## Adjusting Terrain Scale

### Scale Guidelines

| Use Case | Recommended Scale | Notes |
|----------|-------------------|-------|
| Small indoor space | `(0.05, 0.05, 1.0)` | ~13m area, low height |
| Outdoor testing | `(0.15, 0.15, 3.0)` | ~38m area, moderate height |
| Large desert area | `(0.5, 0.5, 5.0)` | ~128m area, high variation |
| City/urban | `(1.0, 1.0, 10.0)` | ~256m area, tall buildings |

### Finding the Right Scale

1. **Start with default**: `(0.15, 0.15, 3.0)`
2. **Run simulation** and observe:
   - Is terrain too small? Increase x,y scale
   - Is terrain too flat? Increase z scale
   - Is terrain too steep? Decrease z scale
3. **Adjust iteratively**

### Common Scale Adjustments

**Terrain appears too small:**
```python
# Before
terrain_scale=(0.15, 0.15, 3.0)  # 38m x 38m

# After
terrain_scale=(0.5, 0.5, 3.0)    # 128m x 128m
```

**Terrain too flat:**
```python
# Before
terrain_scale=(0.15, 0.15, 1.0)  # Low peaks

# After
terrain_scale=(0.15, 0.15, 5.0)  # Tall peaks
```

**Terrain too jagged/steep:**
```python
# Option 1: Reduce z scale
terrain_scale=(0.15, 0.15, 1.5)

# Option 2: Smooth heightmap during conversion
python odm_to_heightmap.py --input dsm.tif --output terrain.png --smooth 3.0
```

---

## Troubleshooting

### "Module 'rasterio' not found"

Install missing dependency:
```bash
pip install rasterio
```

### "Terrain appears stretched or distorted"

**Cause**: Plane size doesn't match terrain size

**Solution**: Adjust plane size in simulation:
```python
# Calculate actual terrain size
terrain_size = 256 * 0.15  # heightmap_pixels * x_scale
plane_size = terrain_size * 1.2  # Slightly larger
```

### "Terrain is too flat / no variation"

**Causes**:
1. ODM DEM has low elevation range
2. z_scale is too small

**Solutions**:
```python
# Increase z_scale
terrain_scale=(0.15, 0.15, 5.0)  # Was 3.0, now 5.0
```

### "Terrain is too steep / drones can't navigate"

**Solutions**:
1. Reduce z_scale:
   ```python
   terrain_scale=(0.15, 0.15, 1.5)
   ```

2. Apply smoothing during conversion:
   ```bash
   python odm_to_heightmap.py --input dsm.tif --output terrain.png --smooth 3.0
   ```

### "Simulation is slow / laggy"

**Causes**:
1. Heightmap too large
2. Camera detection enabled

**Solutions**:
1. Reduce heightmap resolution:
   ```bash
   python odm_to_heightmap.py --input dsm.tif --output terrain.png --size 256
   ```

2. Disable camera detection (already done in latest version)

### "Terrain has holes or artifacts"

**Cause**: ODM DEM has nodata values

**Solution**: Use `--smooth` parameter:
```bash
python odm_to_heightmap.py --input dsm.tif --output terrain.png --smooth 1.5
```

### "Coordinate mismatch / terrain in wrong location"

OpenDroneMap uses UTM coordinates, PyBullet uses local coordinates. The heightmap is automatically centered at origin (0, 0, 0).

If you need precise georeferencing:
1. ODM provides offset information in `odm_georeferencing/`
2. Calculate drone GPS positions relative to terrain origin
3. Apply coordinate transforms in your simulation code

---

## Advanced: Full Mesh Integration

For users who need complete 3D models with textures (not covered in this quick guide):

1. ODM generates `odm_textured_model.obj` with full 3D mesh
2. Create URDF file referencing the OBJ mesh
3. Load with `p.loadURDF("terrain.urdf")`
4. **Warning**: Complex meshes have performance cost

This approach is only recommended for visualization, not real-time physics simulation.

---

## Resources

- **OpenDroneMap**: https://opendronemap.org/odm/
- **WebODM**: https://www.opendronemap.org/webodm/
- **PyODM**: https://pyodm.readthedocs.io/
- **PyBullet**: https://pybullet.org/

---

## Quick Reference

### Generate Example Terrains
```bash
python generate_example_terrain.py
```

### Convert ODM DEM
```bash
python odm_to_heightmap.py -i dsm.tif -o terrain.png --size 512 --smooth 2.0
```

### Use in Simulation
```python
terrain_id = create_desert_terrain(
    heightmap_path="assets/terrain_desert_dunes.png",
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.15, 0.15, 3.0)
)
```

---

*Last updated: 2026-01*
