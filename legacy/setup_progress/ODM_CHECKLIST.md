# OpenDroneMap Setup Checklist

## ✅ Current Status

- [x] Git installed (version 2.51.2)
- [ ] Docker Desktop installed
- [ ] Computer restarted after Docker install
- [ ] Docker Desktop running
- [ ] WebODM downloaded
- [ ] WebODM started successfully
- [ ] WebODM account created
- [ ] Sample data downloaded
- [ ] First project processed
- [ ] DSM file downloaded
- [ ] Python dependencies installed (rasterio)
- [ ] DSM converted to heightmap
- [ ] Custom terrain used in simulation

---

## 📝 Your Next Steps (In Order)

### RIGHT NOW:

**1. Install Docker Desktop**
   - Go to: https://www.docker.com/products/docker-desktop
   - Download and run installer
   - Restart computer when prompted
   - **Time: 15 minutes**

### AFTER RESTART:

**2. Verify Docker**
   ```powershell
   docker --version
   ```
   Should show Docker version

**3. Install WebODM**
   ```powershell
   cd C:\Users\z7aa\
   git clone https://github.com/OpenDroneMap/WebODM --depth 1
   cd WebODM
   .\webodm.bat start
   ```
   **Wait 10-20 minutes for first-time download**

**4. Download Sample Data**
   ```powershell
   cd C:\Users\z7aa\gym-pybullet-drones\gym_pybullet_drones\examples
   .\download_sample_data.ps1
   ```

**5. Process Images in WebODM**
   - Open: http://localhost:8000
   - Create account
   - Upload sample images
   - Process (wait 20-40 minutes)

**6. Convert and Use**
   - Download DSM from WebODM
   - Convert to PNG
   - Use in simulation

---

## 📚 Detailed Instructions

See: **INSTALL_ODM_WINDOWS.md** for complete step-by-step guide

---

## ⏱️ Time Estimate

- Docker install + restart: **15 minutes**
- WebODM first-time setup: **20 minutes**
- Sample data download: **5 minutes**
- Image processing: **30 minutes**
- Convert and test: **5 minutes**

**Total: ~1.5 hours** (mostly waiting)

---

## 🆘 If Something Goes Wrong

1. Check: `INSTALL_ODM_WINDOWS.md` - Troubleshooting section
2. Check: `ODM_BEGINNERS_GUIDE.md` - Detailed explanations
3. Ask me for help!

---

*Start with: Install Docker Desktop*
*Then come back and tell me when it's done!*
