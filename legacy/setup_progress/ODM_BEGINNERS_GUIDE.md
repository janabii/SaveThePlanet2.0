# OpenDroneMap for Absolute Beginners

## What is OpenDroneMap?

OpenDroneMap (ODM) is free software that takes overlapping photos from a drone and creates:
- 3D models
- Elevation maps (terrain height data)
- Orthophotos (flat, accurate maps)

**Think of it like:** Taking many photos of a mountain from different angles, and ODM stitches them together to create a 3D model of that mountain.

---

## What You Need

### 1. Aerial Photos (Choose ONE option)

**Option A: Use Your Own Drone Photos** ✈️
- At least 15-20 photos of the same area
- Photos must overlap by 70% or more
- Taken from different positions/angles
- Common formats: JPG, JPEG, PNG

**Option B: Download Sample Dataset** 📥 (Recommended for learning)
- Free sample photos available online
- Pre-made datasets for testing
- No drone needed!

**Option C: Use Our Example Terrains** 🎮 (Easiest!)
- We already generated 5 terrains for you
- Skip ODM entirely and use those
- See `QUICKSTART.md`

### 2. OpenDroneMap Software

We'll use **WebODM** (easiest option with a web interface).

---

## Complete Setup Guide

### Method 1: WebODM (Recommended for Beginners)

WebODM has a friendly web interface - like using a website instead of typing commands.

#### Step 1: Install Docker

Docker is required to run WebODM (it's like a container for the software).

**Windows:**
1. Download Docker Desktop: https://www.docker.com/products/docker-desktop
2. Install it (follow the installer)
3. Restart your computer
4. Open Docker Desktop and wait for it to start

**Verify Docker works:**
```bash
docker --version
```
You should see something like: `Docker version 24.0.x`

#### Step 2: Install WebODM

Open PowerShell or Command Prompt:

```bash
# Download WebODM
git clone https://github.com/OpenDroneMap/WebODM --config core.autocrlf=input --depth 1
cd WebODM

# Start WebODM (first time takes 10-15 minutes to download)
./webodm.bat start
```

**Note:** First run downloads ~6GB of data. Be patient!

#### Step 3: Open WebODM

1. Wait for the message: "WebODM is now running at http://localhost:8000"
2. Open your web browser
3. Go to: **http://localhost:8000**
4. Create an account (username and password - stored locally only)

#### Step 4: Process Your First Project

**Using Sample Data:**

1. Download sample images:
   - Go to: https://github.com/OpenDroneMap/ODM/tree/master/tests/test_data
   - Download images from the `images` folder (about 10-15 photos)
   - Or use this direct link: https://github.com/OpenDroneMap/odm_data/releases

2. In WebODM web interface:
   - Click "**+ Add Project**"
   - Name it: "My First Terrain"
   - Click "**Select Images**"
   - Choose your downloaded photos (select multiple)
   - Click "**Upload**" (wait for upload to complete)

3. Process the images:
   - Click "**Start Processing**"
   - Select options (keep defaults for now)
   - Click "**Start**"
   - Processing takes 10-60 minutes depending on image count

4. Monitor progress:
   - You'll see a progress bar
   - Console shows what's happening
   - Don't close the browser or turn off computer

5. When complete:
   - Status changes to "**Completed**"
   - Click on the project to view results

#### Step 5: Download the Terrain Data

Once processing is complete:

1. In WebODM, click your project
2. Click "**Download Assets**" button
3. Select "**All Assets**" (or choose specific files):
   - `dsm.tif` - Digital Surface Model (what we need!)
   - `dtm.tif` - Digital Terrain Model (ground only)
   - `orthophoto.tif` - Flat photo map
   - `textured_model.obj` - 3D model with textures

4. Download location: Usually in your `Downloads` folder
5. Save to a known location, e.g., `C:\Users\YourName\odm_output\`

---

### Method 2: Docker Command Line (Advanced)

If you prefer command line instead of WebODM:

#### Quick Start

```bash
# Create folder for your project
mkdir C:\odm_projects\my_terrain
cd C:\odm_projects\my_terrain

# Create images folder
mkdir images

# Copy your drone photos into the images folder
# (drag and drop photos into: C:\odm_projects\my_terrain\images\)

# Run ODM processing
docker run -ti --rm -v C:\odm_projects\my_terrain:/datasets/my_terrain opendronemap/odm --project-path /datasets my_terrain

# Wait for processing (can take 30+ minutes)

# Output will be in: C:\odm_projects\my_terrain\odm_dem\
# Look for: dsm.tif and dtm.tif
```

---

## Convert ODM Output to Simulation Terrain

### Step 1: Install Python Dependencies

```bash
pip install rasterio pillow numpy scipy
```

### Step 2: Convert DEM to Heightmap

```bash
# Navigate to your examples folder
cd C:\Users\z7aa\gym-pybullet-drones\gym_pybullet_drones\examples

# Convert the DEM file
python odm_to_heightmap.py --input C:\odm_projects\my_terrain\odm_dem\dsm.tif --output assets\my_custom_terrain.png --size 512 --smooth 2.0
```

**Parameters explained:**
- `--input`: Path to the DSM file from ODM
- `--output`: Where to save the PNG heightmap
- `--size`: Resolution (256, 512, or 1024) - higher = more detail but slower
- `--smooth`: Smoothing level (0-5) - higher = smoother terrain

### Step 3: Use in Simulation

Edit `swarm-mixed-fleet.py` line ~516:

```python
terrain_id = create_desert_terrain(
    heightmap_path="assets/my_custom_terrain.png",
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.15, 0.15, 3.0)  # Adjust as needed
)
```

---

## Understanding Key Files

### Input Files (What ODM Needs)
- **Drone photos** (JPG/JPEG): Your aerial images
  - Must have GPS data (EXIF metadata)
  - Should overlap by 70%+
  - 15+ photos recommended

### Output Files (What ODM Creates)

| File | Description | Use For |
|------|-------------|---------|
| `dsm.tif` | Digital Surface Model | Full terrain with objects (trees, buildings) |
| `dtm.tif` | Digital Terrain Model | Ground only, no objects |
| `orthophoto.tif` | Flat photo map | Textures, reference |
| `textured_model.obj` | 3D mesh | Advanced visualization |
| `point_cloud.ply` | Point cloud | 3D analysis |

**For drone simulation, use:** `dsm.tif` (includes all obstacles)

---

## Sample Datasets (No Drone Needed!)

### Free Sample Data Sources

1. **OpenDroneMap GitHub**
   - URL: https://github.com/OpenDroneMap/odm_data/releases
   - Download: `brighton_beach.zip` or other datasets
   - Extract and use the images folder

2. **Drone Deploy Sample**
   - URL: https://www.dronedeploy.com/sample-data.html
   - Free sample datasets
   - Various terrain types

3. **SenseFly Sample Data**
   - URL: https://www.sensefly.com/education/datasets/
   - High-quality sample flights
   - Real-world scenarios

### How to Use Sample Data

1. Download and extract the ZIP file
2. Look for the `images` folder with drone photos
3. Use those photos with WebODM or Docker method above

---

## Complete Example Workflow

Let's do a full example from start to finish:

### Scenario: Create Terrain from Sample Data

**Step 1: Get Sample Images**
```bash
# Download sample data
# Go to: https://github.com/OpenDroneMap/odm_data/releases
# Download: brighton_beach.zip
# Extract to: C:\odm_sample\brighton_beach\
```

**Step 2: Process with WebODM**
```
1. Open http://localhost:8000 in browser
2. Login to WebODM
3. Click "+ Add Project"
4. Name: "Brighton Beach Terrain"
5. Click "Select Images"
6. Navigate to: C:\odm_sample\brighton_beach\images\
7. Select all images (Ctrl+A)
8. Click "Upload" (wait for upload)
9. Click "Start Processing"
10. Keep default options
11. Click "Start"
12. Wait 15-30 minutes
13. Check progress periodically
```

**Step 3: Download Results**
```
1. Click on your completed project
2. Click "Download Assets"
3. Download "All Assets" or just "dsm.tif"
4. Save to: C:\odm_output\brighton_beach_dsm.tif
```

**Step 4: Convert to Heightmap**
```bash
cd C:\Users\z7aa\gym-pybullet-drones\gym_pybullet_drones\examples

python odm_to_heightmap.py --input C:\odm_output\brighton_beach_dsm.tif --output assets\brighton_beach_terrain.png --size 512 --smooth 2.0
```

**Step 5: Use in Simulation**
```python
# Edit swarm-mixed-fleet.py line ~516
terrain_id = create_desert_terrain(
    heightmap_path="assets/brighton_beach_terrain.png",
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.2, 0.2, 4.0)  # Adjust based on terrain size
)
```

**Step 6: Run Simulation**
```bash
python swarm-mixed-fleet.py --gui True
```

---

## Troubleshooting

### WebODM Issues

**Problem: WebODM won't start**
```bash
# Check Docker is running
docker ps

# Restart Docker Desktop
# Then try again:
cd WebODM
./webodm.bat restart
```

**Problem: Can't access http://localhost:8000**
- Wait 2-3 minutes after starting
- Check if Docker Desktop is running
- Try restarting Docker

**Problem: Processing fails**
- Check you have at least 15 photos
- Verify photos have GPS data (EXIF)
- Try with sample data first to test setup

### Conversion Issues

**Problem: "rasterio not found"**
```bash
pip install rasterio
```

**Problem: "Input file not found"**
- Check file path is correct
- Use full path: `C:\Users\...\dsm.tif`
- Check file extension is `.tif` not `.tiff`

**Problem: Terrain looks wrong**
- Try different smoothing: `--smooth 3.0`
- Adjust size: `--size 256` (faster) or `--size 1024` (more detail)

---

## Quick Reference Card

### Start WebODM
```bash
cd WebODM
./webodm.bat start
```

### Stop WebODM
```bash
cd WebODM
./webodm.bat stop
```

### Convert DEM
```bash
python odm_to_heightmap.py -i input.tif -o output.png --size 512 --smooth 2.0
```

### Check Docker Status
```bash
docker ps
```

---

## Tips for Best Results

### Taking Drone Photos

1. **Overlap:** 70-80% forward overlap, 60% side overlap
2. **Height:** Consistent altitude (e.g., 50 meters)
3. **Speed:** Slow and steady
4. **Pattern:** Grid pattern covering entire area
5. **Weather:** Overcast is better (no harsh shadows)
6. **Images:** Need at least 15-20 photos

### Processing Settings

**Fast Processing (Testing):**
- Size: 256x256
- No smoothing
- Lower quality settings in WebODM

**High Quality (Final):**
- Size: 512x512 or 1024x1024
- Smoothing: 2.0-3.0
- High quality settings in WebODM
- More processing time (1-2 hours)

---

## Recommended Learning Path

### Beginner (Start Here!)
1. ✅ Use pre-generated example terrains (skip ODM)
2. ✅ Try `generate_example_terrain.py`
3. Run simulation with different terrains

### Intermediate (Learn ODM)
1. Install WebODM
2. Download sample dataset
3. Process in WebODM
4. Convert DEM to heightmap
5. Use in simulation

### Advanced (Real Drone Data)
1. Collect your own aerial photos
2. Process with ODM
3. Fine-tune conversion parameters
4. Optimize terrain scale

---

## Cost and Requirements

### Free and Open Source
- ✅ OpenDroneMap: Free
- ✅ WebODM: Free
- ✅ Docker: Free
- ✅ Our tools: Free

### Computer Requirements
- **Minimum:** 8GB RAM, 4 CPU cores
- **Recommended:** 16GB RAM, 8 CPU cores
- **Storage:** 10GB free space (for Docker + projects)
- **OS:** Windows 10/11, Linux, or Mac

### Time Investment
- **Setup:** 30 minutes (first time)
- **Processing:** 15-60 minutes per project
- **Learning:** 1-2 hours to understand basics

---

## Next Steps

### Option 1: Skip ODM (Easiest)
Just use the pre-generated terrains we created:
- See: `QUICKSTART.md`
- Edit one line in the simulation
- Done!

### Option 2: Try ODM (Learning)
Follow this guide to:
1. Install WebODM
2. Download sample data
3. Process your first project
4. Convert and use in simulation

### Option 3: Real Data (Advanced)
After learning with samples:
1. Collect your own drone photos
2. Process with ODM
3. Create custom terrains for your research

---

## Support Resources

### Official Documentation
- WebODM: https://www.opendronemap.org/webodm/
- ODM Docs: https://docs.opendronemap.org/
- Community: https://community.opendronemap.org/

### Video Tutorials
- YouTube: Search "OpenDroneMap tutorial"
- WebODM basics: Search "WebODM getting started"

### Our Documentation
- Quick Start: `QUICKSTART.md`
- Integration Guide: `ODM_INTEGRATION.md`
- Technical Details: `IMPLEMENTATION_SUMMARY.md`

---

## Summary

**Absolute Simplest Path:**
1. Skip ODM entirely
2. Use our 5 pre-generated terrains
3. See `QUICKSTART.md`

**Learning ODM Path:**
1. Install Docker + WebODM (30 min)
2. Download sample images (5 min)
3. Process in WebODM (30 min)
4. Convert with our tool (1 min)
5. Use in simulation (1 min)

**Both paths work great!** Start with pre-generated terrains, learn ODM when you need custom terrains.

---

*Need help? Check `ODM_INTEGRATION.md` for more details or ask questions!*
