# OpenDroneMap Setup - Complete Guide for Your System

## Step 1: Install Docker Desktop (Required)

Docker is needed to run OpenDroneMap. Follow these steps:

### Download and Install

1. **Open your browser** and go to:
   ```
   https://www.docker.com/products/docker-desktop
   ```

2. **Click "Download for Windows"**

3. **Run the installer** (Docker Desktop Installer.exe)
   - Accept the license agreement
   - Keep default settings
   - Click "Install"

4. **Restart your computer** (required for Docker to work)

5. **Start Docker Desktop**
   - Find "Docker Desktop" in Start Menu
   - Open it and wait for it to start (may take 2-3 minutes first time)
   - You'll see a whale icon in your system tray when ready

6. **Verify Docker works:**
   ```powershell
   docker --version
   ```
   Should show: `Docker version 24.x.x` or similar

---

## Step 2: Install WebODM (Once Docker is Ready)

After Docker is installed and running, run this command:

```powershell
cd C:\Users\z7aa\
git clone https://github.com/OpenDroneMap/WebODM --depth 1
cd WebODM
.\webodm.bat start
```

**What happens:**
- Downloads WebODM (takes 2-3 minutes)
- First start downloads ~6GB of data (takes 10-20 minutes)
- Be patient! This is one-time setup

**When ready, you'll see:**
```
WebODM is now running at http://localhost:8000
```

---

## Step 3: Download Sample Data (For Testing)

I'll provide a script to download sample drone images for you to test with.

---

## Quick Start After Setup

1. **Start Docker Desktop** (if not running)

2. **Start WebODM:**
   ```powershell
   cd C:\Users\z7aa\WebODM
   .\webodm.bat start
   ```

3. **Open browser:** http://localhost:8000

4. **Create account** (local only, not shared online)

5. **Upload sample images** and process them

6. **Download DSM file**

7. **Convert to heightmap:**
   ```powershell
   cd C:\Users\z7aa\gym-pybullet-drones\gym_pybullet_drones\examples
   python odm_to_heightmap.py --input path\to\dsm.tif --output assets\my_terrain.png --size 512
   ```

---

## Current Status

✅ Git installed
❌ Docker not installed - **Install this first!**
⏳ WebODM - Install after Docker

---

## Next Steps for You

1. **NOW:** Install Docker Desktop (link above, restart after)
2. **AFTER RESTART:** Come back and I'll help you install WebODM
3. **THEN:** We'll download sample data and process it

---

*Save this file for reference!*
