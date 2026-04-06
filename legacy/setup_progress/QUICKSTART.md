# Quick Start Guide - Terrain System

## ✅ What's Done

1. **Camera lag fixed** - Simulation runs smoothly now
2. **Plane size fixed** - Terrain fits properly without stretching
3. **5 realistic terrains** - Ready to use immediately
4. **ODM integration** - Tools ready for real aerial imagery

---

## 🚀 Try It Now (3 Steps)

### 1. Terrains Are Already Generated

Five terrain heightmaps are ready in `assets/`:
- ✅ `terrain_desert_dunes.png` - Smooth dunes
- ✅ `terrain_mountains.png` - Rocky peaks  
- ✅ `terrain_canyon.png` - Deep valleys
- ✅ `terrain_hills.png` - Rolling hills
- ✅ `terrain_valleys.png` - Plateaus

### 2. Edit Simulation (One Line)

Open `swarm-mixed-fleet.py`, find line ~516, and change:

```python
# FROM THIS:
terrain_id = create_desert_terrain(
    heightmap_path="assets/desert_heightmap.png",
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.15, 0.15, 3.0)
)

# TO THIS (try different terrains!):
terrain_id = create_desert_terrain(
    heightmap_path="assets/terrain_mountains.png",  # ← Changed this line
    texture_path="assets/desert_sand.png",
    terrain_scale=(0.15, 0.15, 5.0)  # ← Adjusted height for mountains
)
```

### 3. Run

```bash
python swarm-mixed-fleet.py --gui True
```

---

## 🎮 Terrain Options

| Change Line 516 to: | Result |
|---------------------|--------|
| `terrain_desert_dunes.png` + scale `(0.15, 0.15, 3.0)` | Smooth desert dunes |
| `terrain_mountains.png` + scale `(0.15, 0.15, 5.0)` | Rocky mountain peaks |
| `terrain_canyon.png` + scale `(0.15, 0.15, 4.0)` | Deep canyon valleys |
| `terrain_hills.png` + scale `(0.15, 0.15, 2.0)` | Gentle rolling hills |
| `terrain_valleys.png` + scale `(0.15, 0.15, 3.5)` | Plateaus and valleys |

---

## 🔧 Need More Terrains?

### Generate New Random Terrains

```bash
python generate_example_terrain.py
```

This regenerates all 5 terrains with new random variations.

### Use Real Aerial Photos (Advanced)

See **`ODM_INTEGRATION.md`** for complete guide on using OpenDroneMap.

---

## 📖 Full Documentation

- **Quick Reference:** `README_TERRAIN.md`
- **ODM Integration:** `ODM_INTEGRATION.md`
- **Implementation Details:** `IMPLEMENTATION_SUMMARY.md`

---

## ❓ Quick Troubleshooting

**Problem: Terrain looks flat**
- Solution: Increase the z_scale value (third number)
  ```python
  terrain_scale=(0.15, 0.15, 8.0)  # More dramatic height
  ```

**Problem: Terrain looks stretched**
- Solution: Already fixed! Plane size now matches terrain.

**Problem: Simulation is laggy**
- Solution: Already fixed! Camera detection is disabled.

**Problem: Want camera detection back**
- Solution: Uncomment lines 656-660 in `swarm-mixed-fleet.py`

---

## ✨ That's It!

You're ready to test realistic terrain with your drones. Just edit one line and run!

---

*Questions? Check `README_TERRAIN.md` or `ODM_INTEGRATION.md`*
