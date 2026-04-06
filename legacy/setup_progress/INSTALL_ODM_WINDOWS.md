# Complete OpenDroneMap Installation - Windows

## 🎯 Your Situation

✅ **You have:** Git (already installed)
❌ **You need:** Docker Desktop (must install)
⏳ **Then install:** WebODM (after Docker)

---

## 📦 Step 1: Install Docker Desktop (15 minutes)

### 1.1: Download Docker

**Option A: Direct Download (Fastest)**
1. Open this link in your browser:
   ```
   https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
   ```
2. Download will start automatically (~500MB)

**Option B: From Website**
1. Go to: https://www.docker.com/products/docker-desktop
2. Click "Download for Windows"

### 1.2: Install Docker

1. **Run** `Docker Desktop Installer.exe`
2. **Configuration screen:**
   - ✅ Use WSL 2 instead of Hyper-V (recommended)
   - ✅ Add shortcut to desktop
3. Click **"Ok"** to start installation
4. Wait for installation (3-5 minutes)
5. Click **"Close and restart"** when prompted

### 1.3: After Restart

1. **Docker Desktop will auto-start** (if not, find it in Start Menu)
2. **Accept** the Service Agreement
3. **Wait** for Docker to start (2-3 minutes - you'll see whale icon in system tray)
4. When ready, whale icon is stable (not animated)

### 1.4: Verify Docker Works

Open **PowerShell** and run:
```powershell
docker --version
```

Should show: `Docker version 24.x.x` or similar

✅ **Docker is ready!** Move to Step 2.

---

## 🌐 Step 2: Install WebODM (20 minutes)

### 2.1: Download WebODM

Open **PowerShell** and run:

```powershell
cd C:\Users\z7aa\
git clone https://github.com/OpenDroneMap/WebODM --depth 1
```

This downloads WebODM (takes 1-2 minutes).

### 2.2: Start WebODM (IMPORTANT: First time is slow!)

```powershell
cd WebODM
.\webodm.bat start
```

**What happens:**
- First run downloads **~6GB** of Docker images
- Takes **10-20 minutes** depending on internet speed
- You'll see lots of "Pulling from..." messages
- **Be patient!** This is one-time setup

**When successful, you'll see:**
```
WebODM is now running at http://localhost:8000
```

### 2.3: Open WebODM

1. Open your browser (Chrome, Firefox, Edge)
2. Go to: **http://localhost:8000**
3. You should see WebODM login page!

### 2.4: Create Account

1. Click **"Sign Up"**
2. Create account:
   - Username: (your choice, e.g., "admin")
   - Email: (can be fake, e.g., "test@test.com")
   - Password: (your choice)
3. Click **"Sign Up"**
4. You're in! 🎉

✅ **WebODM is ready!** Move to Step 3.

---

## 📸 Step 3: Download Sample Data (5 minutes)

### 3.1: Run Download Script

Open **PowerShell** and run:

```powershell
cd C:\Users\z7aa\gym-pybullet-drones\gym_pybullet_drones\examples
.\download_sample_data.ps1
```

This downloads sample drone images for testing.

**If script doesn't work, manual download:**
1. Go to: https://github.com/OpenDroneMap/odm_data/releases
2. Download: `brighton_beach.zip`
3. Extract to: `C:\odm_samples\brighton_beach\`

### 3.2: Verify Sample Data

Check that folder exists:
```
C:\odm_samples\brighton_beach\images\
```

Should contain ~20 JPG images.

✅ **Sample data ready!** Move to Step 4.

---

## 🚁 Step 4: Process Your First Terrain (30 minutes)

### 4.1: Make Sure WebODM is Running

If you closed it, start again:
```powershell
cd C:\Users\z7aa\WebODM
.\webodm.bat start
```

Wait for: `WebODM is now running at http://localhost:8000`

### 4.2: Create Project in WebODM

1. Open: **http://localhost:8000** in browser
2. Login with your account
3. Click **"+ Add Project"** (top right)
4. Enter project name: **"My First Terrain"**
5. Click **"Create Project"**

### 4.3: Upload Images

1. In your project, click **"Select Images"**
2. Navigate to: `C:\odm_samples\brighton_beach\images\`
3. Select **ALL images** (Ctrl + A)
4. Click **"Open"**
5. Wait for upload progress (1-2 minutes)
6. All images should appear in the list

### 4.4: Start Processing

1. Click **"Start Processing"** button
2. **Options dialog appears** - for now, keep all defaults
3. Scroll down and click **"Start"** button at bottom
4. Processing begins!

**What to expect:**
- Status shows: "Running"
- Progress bar appears
- Console shows technical details
- Takes **20-40 minutes** for sample data
- Computer may get warm (normal)
- **Don't close browser or turn off computer!**

### 4.5: Monitor Progress

You can:
- ✅ Minimize browser window
- ✅ Do other light work
- ✅ Check back periodically
- ❌ Don't close browser
- ❌ Don't shut down computer
- ❌ Don't stop Docker

**When complete:**
- Status changes to: **"Completed"**
- Green checkmark appears
- You'll see buttons to view results

✅ **Processing complete!** Move to Step 5.

---

## 💾 Step 5: Download Terrain Data (2 minutes)

### 5.1: Download DSM File

1. Click on your completed project name
2. Click **"Download Assets"** button (on right side)
3. In the download menu, find: **"dsm.tif"**
4. Click to download (file is ~10-50MB)
5. Wait for download to complete
6. Find file in your Downloads folder

### 5.2: Move DSM to Known Location

Move the DSM file:
```
From: C:\Users\z7aa\Downloads\dsm.tif
To:   C:\odm_output\brighton_beach_dsm.tif
```

Or create the folder first:
```powershell
mkdir C:\odm_output
move C:\Users\z7aa\Downloads\dsm.tif C:\odm_output\brighton_beach_dsm.tif
```

✅ **DSM downloaded!** Move to Step 6.

---

## 🔄 Step 6: Convert DSM to Heightmap (1 minute)

### 6.1: Install Python Dependencies (first time only)

```powershell
pip install rasterio pillow numpy scipy
```

Wait for installation (2-3 minutes).

### 6.2: Convert DSM to PNG

```powershell
cd C:\Users\z7aa\gym-pybullet-drones\gym_pybullet_drones\examples

python odm_to_heightmap.py --input C:\odm_output\brighton_beach_dsm.tif --output assets\brighton_beach_terrain.png --size 512 --smooth 2.0
```

**Output should show:**
```
[DEM] Loading: C:\odm_output\brighton_beach_dsm.tif
[DEM] Loaded heightmap: ...
[SUCCESS] Saved heightmap: assets\brighton_beach_terrain.png
```

✅ **Heightmap created!** Move to Step 7.

---

## 🎮 Step 7: Use in Simulation (1 minute)

### 7.1: Edit Simulation File

Open: `swarm-mixed-fleet.py`

Find line ~516 and change to:

```python
terrain_id = create_desert_terrain(
    heightmap_path="assets/brighton_beach_terrain.png",  # Your custom terrain!
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.2, 0.2, 3.0)  # Adjust as needed
)
```

### 7.2: Run Simulation

```powershell
python swarm-mixed-fleet.py --gui True
```

**You should see your custom terrain with drones flying over it!** 🎉

---

## 🎊 Success! What You've Done:

1. ✅ Installed Docker Desktop
2. ✅ Installed WebODM
3. ✅ Downloaded sample drone images
4. ✅ Processed images to create 3D terrain data
5. ✅ Downloaded DSM file
6. ✅ Converted DSM to heightmap PNG
7. ✅ Used custom terrain in simulation

---

## 🔄 Using ODM Again (Future Projects)

### Start WebODM
```powershell
cd C:\Users\z7aa\WebODM
.\webodm.bat start
```

### Stop WebODM
```powershell
cd C:\Users\z7aa\WebODM
.\webodm.bat stop
```

### Process New Images
1. Start WebODM
2. Open http://localhost:8000
3. Create new project
4. Upload your images
5. Process
6. Download DSM
7. Convert and use!

---

## 📊 Quick Reference Card

```powershell
# Check Docker
docker --version

# Start WebODM
cd C:\Users\z7aa\WebODM
.\webodm.bat start

# Stop WebODM
.\webodm.bat stop

# Convert DSM to heightmap
cd C:\Users\z7aa\gym-pybullet-drones\gym_pybullet_drones\examples
python odm_to_heightmap.py --input path\to\dsm.tif --output assets\terrain.png --size 512
```

---

## ❓ Troubleshooting

### Docker won't start
- Restart computer
- Make sure virtualization is enabled in BIOS
- Try running as administrator

### WebODM won't start
- Make sure Docker Desktop is running
- Wait 2-3 minutes after starting Docker
- Check if port 8000 is already in use

### Processing failed
- Make sure you have at least 15 photos
- Photos must have GPS data (EXIF)
- Try with sample data first to test

### Can't convert DSM
```powershell
pip install rasterio pillow numpy scipy
```

### Terrain looks wrong
- Adjust terrain_scale values
- Try different smoothing: `--smooth 3.0`
- See README_TERRAIN.md for scale guide

---

## 🆘 Need Help?

1. Check: `ODM_BEGINNERS_GUIDE.md`
2. Check: `ODM_VISUAL_GUIDE.md`
3. ODM Community: https://community.opendronemap.org/

---

*Installation time: ~1 hour total (mostly waiting for downloads)*
*Future processing: 20-40 minutes per project*
