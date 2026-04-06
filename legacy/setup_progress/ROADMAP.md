# Save the Planet - Development Roadmap

## Current Status ✅
- ✅ 5 drones in W formation
- ✅ Autonomous waypoint following
- ✅ Basic orange color detection (HSV)
- ✅ Camera capture with bounding boxes
- ✅ Desert ground texture
- ✅ Collision avoidance

## Next Steps - Priority Order

### Phase 1: Enhanced Environment & Realism (Weeks 1-2)

#### 1.1 Diverse Waste Objects
**Goal**: Create realistic, varied waste objects for better training data

**Tasks**:
- [ ] Create multiple waste types: plastic bottles, cans, bags, cardboard boxes
- [ ] Add different sizes (small, medium, large)
- [ ] Implement random orientations and positions
- [ ] Add material properties (some should be semi-transparent, reflective)
- [ ] Create URDF files for complex shapes (not just cubes)

**Files to create**:
- `assets/waste_objects/` folder with URDF files
- `waste_generator.py` - procedural waste spawning

#### 1.2 Improved Terrain
**Goal**: More realistic and varied environments

**Tasks**:
- [ ] Add heightmaps for varied terrain (hills, valleys)
- [ ] Multiple environment types: desert, beach, forest, urban
- [ ] Add obstacles: trees, buildings, rocks
- [ ] Dynamic lighting (time of day simulation)
- [ ] Weather effects (wind, fog density)

**Files to create**:
- `terrain_generator.py`
- `environment_presets.py`

#### 1.3 Better Camera System
**Goal**: More realistic camera with noise and calibration

**Tasks**:
- [ ] Add camera noise (Gaussian, salt-and-pepper)
- [ ] Implement depth camera option
- [ ] Add camera calibration parameters
- [ ] Support multiple camera angles (front, bottom, gimbal)
- [ ] Add motion blur simulation

### Phase 2: Advanced Detection System (Weeks 3-4)

#### 2.1 Multi-Class Detection
**Goal**: Detect multiple waste types, not just orange

**Tasks**:
- [ ] Extend detection to multiple colors/materials
- [ ] Implement class labels (plastic, metal, paper, organic)
- [ ] Add confidence scores for detections
- [ ] Create detection history/validation system

**Files to modify**:
- `swarm-5c.py` → `detection_system.py`

#### 2.2 ML-Ready Data Pipeline
**Goal**: Prepare data for training ML models

**Tasks**:
- [ ] Create annotation format (COCO, YOLO, or custom JSON)
- [ ] Auto-generate ground truth labels from simulation
- [ ] Data augmentation pipeline (rotation, brightness, contrast)
- [ ] Dataset organization (train/val/test splits)
- [ ] Metadata logging (GPS coordinates, altitude, timestamp)

**Files to create**:
- `data_collector.py`
- `annotation_generator.py`
- `dataset_utils.py`

#### 2.3 Integration with ML Models
**Goal**: Use real object detection models (YOLO, Faster R-CNN, etc.)

**Tasks**:
- [ ] Integrate YOLOv8 or similar for real-time detection
- [ ] Create inference pipeline
- [ ] Compare simulation detections vs ML model detections
- [ ] Add model evaluation metrics (precision, recall, mAP)

**Files to create**:
- `ml_detector.py`
- `model_integration.py`

### Phase 3: Proper Gymnasium Environment (Weeks 5-6)

#### 3.1 Create WasteDetectionAviary
**Goal**: Proper Gymnasium environment for RL training

**Tasks**:
- [ ] Create `WasteDetectionAviary` class inheriting from `BaseRLAviary`
- [ ] Define observation space (camera images + state)
- [ ] Define action space (waypoint navigation)
- [ ] Implement reward function:
  - Positive: detecting waste, covering area efficiently
  - Negative: collisions, energy consumption, missing waste
- [ ] Add episode termination conditions
- [ ] Implement reset() with random waste configurations

**Files to create**:
- `envs/WasteDetectionAviary.py`

#### 3.2 Reward Engineering
**Goal**: Design effective rewards for RL training

**Reward Components**:
```python
reward = (
    +10 * waste_detected  # Per new waste found
    +5 * coverage_bonus   # For exploring new areas
    -0.1 * energy_cost    # Penalize high RPM
    -50 * collision       # Heavy penalty for crashes
    -1 * time_penalty     # Encourage efficiency
)
```

### Phase 4: Advanced Features (Weeks 7-8)

#### 4.1 Multi-Agent Coordination
**Goal**: Improve swarm intelligence

**Tasks**:
- [ ] Implement communication between drones
- [ ] Add role assignment (leader, searcher, verifier)
- [ ] Create coverage path planning algorithms
- [ ] Implement distributed decision making

#### 4.2 Performance Metrics & Visualization
**Goal**: Track and visualize mission performance

**Tasks**:
- [ ] Coverage map visualization
- [ ] Detection heatmap
- [ ] Energy consumption tracking
- [ ] Mission completion statistics
- [ ] Real-time dashboard

**Files to create**:
- `visualization/dashboard.py`
- `metrics/performance_tracker.py`

#### 4.3 Simulation-to-Reality Transfer
**Goal**: Make simulation more realistic for real-world deployment

**Tasks**:
- [ ] Add sensor noise models
- [ ] Implement GPS inaccuracy
- [ ] Add wind disturbances
- [ ] Battery degradation model
- [ ] Communication delays

### Phase 5: Production Readiness (Weeks 9-10)

#### 5.1 Testing & Validation
**Tasks**:
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Validation on diverse scenarios

#### 5.2 Documentation
**Tasks**:
- [ ] API documentation
- [ ] User guide
- [ ] Example notebooks
- [ ] Video tutorials

#### 5.3 Deployment Preparation
**Tasks**:
- [ ] Configuration files for different scenarios
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Model export for edge devices

---

## Immediate Next Steps (This Week)

### Step 1: Enhanced Waste Objects
Create a waste object generator that spawns varied, realistic waste.

### Step 2: Improved Detection Pipeline
Upgrade from simple HSV to a more robust detection system with ML integration.

### Step 3: Data Collection System
Build a system to automatically collect and annotate training data.

---

## Quick Wins (Can Do Today)

1. **Add more waste types**: Create URDF files for bottles, cans, bags
2. **Improve detection**: Add morphological operations, better filtering
3. **Add metrics**: Track detection count, coverage area, mission time
4. **Better visualization**: Add on-screen HUD showing detections, battery, etc.

---

## Questions to Consider

1. **RL Training**: Do you want to train RL agents, or use classical control?
2. **Real-world deployment**: Will this run on real drones or stay in simulation?
3. **Detection accuracy**: What's the target precision/recall?
4. **Scale**: How many drones in final system? How large an area?
5. **Waste types**: Which waste types are most important to detect?

---

## Recommended Tech Stack

- **Detection**: YOLOv8 (Ultralytics) or Detectron2
- **RL**: Stable-Baselines3 or Ray RLlib
- **Data**: COCO format for annotations
- **Visualization**: Matplotlib, OpenCV, Plotly
- **Logging**: Weights & Biases or TensorBoard
