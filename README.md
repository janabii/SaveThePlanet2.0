# Save the Planet 2.0 – Multi-UAV Waste Detection System

A cyber-physical simulation framework for autonomous waste detection using a coordinated multi-drone system that combines fixed-wing scanning and quadrotor verification.

---

## Team and Supervision

**Team Members:**
- Ahmed Al Janabi  
- Abdulla Moulana  
- Abdulaziz PoorKalati  
- Mohammed Ali Hajji  
- Abdulrhem Abdullah  
- Sultan Albadr  

**Supervisors:**
- Dr. Gouissem, Ala  
- Dr. Maaradji, Abderrahmane  

---

## Project Overview

Illegal waste dumping in desert environments is difficult to monitor due to the scale and inaccessibility of the terrain. Traditional inspection methods are slow and resource intensive.

This project presents a multi-UAV autonomous detection system that combines:
- A fixed-wing drone for large-scale scanning  
- A quadcopter swarm for precise inspection and verification  

The system integrates AI-based detection, simulation, and coordination into a unified workflow to improve efficiency and accuracy in environmental monitoring.

---

## Key Contributions

- Two-tier UAV architecture combining fixed-wing and quadcopters  
- Heatmap-based localisation for candidate waste sites  
- Integration of a Roboflow object detection model  
- Synthetic data generation pipeline to reduce domain gap  
- Full PyBullet simulation environment with reconstructed terrain  
- Multi-drone task allocation strategies  
- Energy-aware mission execution  

---

## System Architecture

The system operates in two stages.

**Fixed-Wing Stage**
- Performs lawnmower scanning across the terrain  
- Detects waste using AI or HSV fallback  
- Generates a heatmap of candidate locations  

**Quadrotor Stage**
- Assigned to detected hotspots  
- Performs low-altitude hover inspection  
- Captures high-precision detections  

This design allows high coverage with accurate verification.

---

## Methodology

1. Terrain reconstruction using WebODM  
2. Synthetic waste placement using URDF objects  
3. Fixed-wing scanning with nadir camera  
4. AI-based detection using Roboflow or HSV fallback  
5. Heatmap generation and peak extraction  
6. Task allocation to quadcopters  
7. Quadrotor inspection and verification  
8. Performance evaluation using precision, recall, and mAP  

---

## Strategy-Based Execution

The system supports multiple task allocation strategies that control how quadcopters are assigned to detected waste locations.

### Available Strategies

- Worst-Fit  
- Normalised-Spare  
- Hungarian  

Each strategy affects:
- Drone paths  
- Assignment efficiency  
- Detection performance  

### Running the Simulation

#### Default run:

```bash
python v33.py
```
#### Other Strategy Runs: 
```bash
python v33.py --strategy worst_fit
python v33.py --strategy normalised_spare
python v33.py --strategy hungarian
```

## Results and Performance

**Synthetic Yolo26 Trained model (v7)**
- mAP@50: 90.9%
- Precision: 86.6%
- Recall: 91.2%
- F1 Score: 88.8%

### Fixed-Wing Stage
*Note: These metrics are averaged from a recorded 15 run*

- mAP: 21.4%
- Recall: ~89%
- Precision: ~1.5%

The fixed-wing stage provides strong coverage but produces many false positives due to high altitude and wide field of view.

---

### Quadcopter Stage
*Note: These metrics are averaged from a recorded 15 run*

- mAP: 32.7%
- Precision: ~98%
- Recall: ~33%
- Localisation accuracy: ~0.27 metres

The quadcopter stage significantly improves precision by performing close-range verification of detected sites.

---

## Visual Results

### Fixed-Wing Scanning
![fixedwing-flying](https://github.com/user-attachments/assets/b65942a1-17e5-4ec8-858c-c7d39ce70be2)

### Heatmap Detection
<img width="1478" height="726" alt="save-the-planet2-coderun" src="https://github.com/user-attachments/assets/33be9c55-48ce-44f3-95fe-23eb5c05a259" />

### Detection Output
![ending-run](https://github.com/user-attachments/assets/581af74c-608c-4479-b78e-d6b7b56ff8c8)

---

## Output Files

After running the simulation, the following outputs are generated:

- Detection metrics printed in the terminal and in the exit screen
- Captured images in:
  - `camera_frames/`
  - `quad_frames/`
  - run metrics are automatically recorded and saved into excel sheet alongside json file in the `runs/` folder
---

## Limitations

- PyBullet rendering reduces visibility of small objects  
- Domain gap between real-world data and simulation  
- Detection performance depends on camera resolution and altitude  
- Synthetic environment does not fully reflect real-world conditions  

---

## Future Work

- Real-world drone deployment and testing  
- Improved simulation rendering  
- More advanced detection models  
- Enhanced coordination strategies  
- Real-time system integration  
