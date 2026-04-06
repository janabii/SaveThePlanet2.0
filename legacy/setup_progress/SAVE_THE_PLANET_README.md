# Save the Planet - Mixed Fleet Waste Detection System

## Overview

This project implements a multi-drone waste detection system using gym-pybullet-drones for the "Save the Planet" mission. The system combines **2 quadcopters** for detailed inspection with **1 fixed-wing UAV** for wide-area reconnaissance in a desert environment.

## Features Implemented

### 1. Mixed Drone Fleet
- **2 Quadcopters (CF2X)**: Detailed waste inspection with formation flying
- **1 Fixed-wing UAV**: Wide-area patrol with circular flight pattern
- Custom `FIXEDWING` drone model with aerodynamic simulation

### 2. Fixed-Wing Implementation
- **New Files Created:**
  - `fixedwing.urdf` - Physical model with 500g mass, modified inertia
  - `FixedWingControl.py` - Autopilot with throttle/elevator/aileron control
  - `BaseAviary._fixedWingPhysics()` - Lift, drag, and stall prevention

- **Flight Characteristics:**
  - Minimum speed: 8 m/s (stall prevention)
  - Cruise speed: 12 m/s
  - Turn radius: ~10m at cruise
  - Altitude control via pitch/lift
  - Bank-to-turn navigation

### 3. Desert Environment
- **Terrain System:**
  - Procedural heightmap generation (256x256 pixels)
  - Perlin noise-based dunes with realistic topology
  - Height variation: 2-4m elevation changes
  - Sand texture application

- **Obstacles:**
  - 12 rock formations (cubes, spheres, cylinders)
  - 6 vegetation elements (dead bushes)
  - Random placement avoiding waste sites
  - Realistic desert colors (browns, tans)

### 4. Waste Detection
- **8 waste sites** distributed across 35m x 35m area
- **Orange color detection** using OpenCV HSV filtering
- **Automatic screenshot capture** when orange detected
- **Green bounding boxes** around detected waste
- Camera frames saved to `camera_frames_mixed/`

### 5. Coordination Logic
- **Quadcopters:**
  - Leader follows waypoints above waste sites
  - Follower maintains formation (2m side, 1.5m back)
  - Altitude: 2.5m for detailed inspection
  
- **Fixed-wing:**
  - Circular patrol pattern around center
  - Patrol radius: 18m
  - Altitude: 4.5m (higher for safety)
  - Angular speed: 0.15 rad/s (~14 seconds per circle)

- **Collision Avoidance:**
  - Inter-drone separation: 1.5m minimum
  - Altitude separation between types
  - Repulsive forces when drones get too close

## File Structure

```
gym_pybullet_drones/
├── assets/
│   └── fixedwing.urdf              # Fixed-wing drone model (NEW)
│
├── control/
│   ├── DSLPIDControl.py            # Quadcopter control (existing)
│   └── FixedWingControl.py         # Fixed-wing autopilot (NEW)
│
├── envs/
│   ├── BaseAviary.py               # Modified: added _fixedWingPhysics()
│   └── CtrlAviary.py               # Used as base environment
│
├── utils/
│   └── enums.py                    # Modified: added FIXEDWING enum
│
└── examples/
    ├── swarm-5c.py                     # Original 5-drone demo
    ├── swarm-mixed-fleet.py            # Mixed fleet implementation (NEW)
    ├── desert_utils.py                 # Terrain/obstacle utilities (NEW)
    ├── generate_desert_heightmap.py    # Heightmap generator (NEW)
    ├── SAVE_THE_PLANET_README.md       # This file (NEW)
    └── assets/
        ├── desert_heightmap.png        # Generated terrain (NEW)
        ├── desert_sand.png             # Texture (existing)
        └── sand_texture.jpg            # Alt texture (existing)
```

## Usage

### Running the Simulation

```bash
cd gym_pybullet_drones/gym_pybullet_drones/examples
python swarm-mixed-fleet.py --gui=True
```

### Controls

- **Movement**: Fully autonomous (no keyboard needed)
- **Manual Save**: Press `C` to save all drone camera views
- **Camera Control**:
  - Arrow keys: Rotate view (yaw/pitch)
  - `Z`/`X`: Zoom in/out
  - `R`: Reset camera to default
- **Exit**: `Ctrl+C` or close window

### Regenerating Terrain

If you want to modify the desert terrain:

```bash
python generate_desert_heightmap.py
```

Edit parameters in the script:
- `size`: Resolution (default 256x256)
- `scale`: Height multiplier (default 3.0m)
- `octaves`: Detail levels (default 5)
- `seed`: Random seed for reproducibility

## Technical Details

### Fixed-Wing Physics Model

The fixed-wing implementation adds aerodynamic forces on top of the base PyBullet physics:

1. **Lift Force**:
   ```
   L = 0.5 * ρ * V² * S * CL
   CL = CL₀ + CL_α * α
   ```
   - Air density ρ = 1.225 kg/m³
   - Wing area S = 0.15 m²
   - Stall at α > 15°

2. **Drag Force**:
   ```
   D = 0.5 * ρ * V² * S * (CD₀ + CD_induced)
   ```
   - Parasitic drag CD₀ = 0.03
   - Induced drag CD = 0.05

3. **Stall Prevention**:
   - Boost thrust when forward velocity < 8 m/s
   - Automatic nose-down pitch correction

### Coordination Algorithm

**Quadcopter Formation:**
```python
target_pos[follower] = leader_pos + formation_offset
+ collision_avoidance_vector
```

**Fixed-Wing Patrol:**
```python
angle += angular_speed * dt
x = center_x + radius * cos(angle)
y = center_y + radius * sin(angle)
z = patrol_altitude
```

### Waste Detection Pipeline

1. Capture RGB image from drone camera
2. Convert to HSV color space
3. Threshold for orange (H: 5-25°, S: 150-255, V: 100-255)
4. Morphological operations (open, dilate)
5. Find contours, filter by area (>80 pixels)
6. Draw bounding box, save with timestamp

## Configuration Options

Edit these constants in `swarm-mixed-fleet.py`:

```python
# Drone Configuration
DRONE_CONFIGS = [
    {"model": DroneModel.CF2X, "role": "quad_leader"},
    {"model": DroneModel.CF2X, "role": "quad_follower"},
    {"model": DroneModel.FIXEDWING, "role": "fixedwing_scout"},
]

# Flight Parameters
TAKEOFF_Z_QUAD = 2.5        # Quadcopter altitude (m)
TAKEOFF_Z_FIXEDWING = 4.0   # Fixed-wing altitude (m)
FW_PATROL_RADIUS = 18.0     # Patrol radius (m)
FW_ANGULAR_SPEED = 0.15     # Patrol speed (rad/s)

# Environment
num_rocks = 12               # Desert rock obstacles
num_vegetation = 6           # Desert vegetation
area_size = 35.0             # Search area size (m)
```

## Known Limitations

1. **Simplified Aerodynamics**: Fixed-wing uses basic lift/drag model, not high-fidelity CFD
2. **No Wind Simulation**: Calm conditions assumed
3. **Single Environment**: All drones share CF2X base physics (fixed-wing adds custom forces)
4. **Terrain Collision**: PyBullet heightfield collision can be imprecise at boundaries
5. **Camera Stabilization**: No gimbal simulation for fixed-wing

## Future Enhancements

Potential improvements for the system:

1. **Advanced Fixed-Wing**:
   - Aileron/elevator/rudder surface models
   - Wing flex and structural dynamics
   - Propeller wash effects

2. **Environment**:
   - Wind gusts and turbulence
   - Thermal updrafts over sand
   - Dynamic time-of-day (shadows)
   - Weather effects (sandstorms)

3. **AI/ML Integration**:
   - Reinforcement learning for patrol optimization
   - Neural network waste classification
   - Autonomous task allocation

4. **Multi-Agent Coordination**:
   - Swarm intelligence algorithms
   - Leader election protocols
   - Communication delays/dropouts

## Troubleshooting

### Drones Crash on Takeoff
- Check initial positions don't overlap terrain
- Increase `TAKEOFF_TIME` for gentler ascent
- Verify heightmap is loaded correctly

### Fixed-Wing Won't Fly
- Ensure minimum speed >8 m/s
- Check patrol radius allows turning room
- Verify altitude >2m for ground clearance

### No Orange Detection
- Ensure OpenCV (cv2) is installed
- Check waste cube colors in PyBullet GUI
- Adjust HSV threshold ranges if needed

### Performance Issues
- Reduce `num_rocks` and `num_vegetation`
- Lower heightmap resolution (128x128)
- Decrease `DEFAULT_SIMULATION_FREQ_HZ`

## Credits

- **Base Framework**: gym-pybullet-drones by University of Toronto
- **Fixed-Wing Inspiration**: PyFly aerodynamic models
- **Perlin Noise**: Procedural generation algorithms
- **Project**: Save the Planet waste detection mission

## License

This extension maintains the same license as gym-pybullet-drones (check parent repository).

## Contact

For questions or improvements, please open an issue in the repository or contact the development team.

---

**Mission Status**: ✅ All systems operational. Ready for deployment!
