"""
Utility functions for creating desert environments in PyBullet.

Includes terrain generation and obstacle placement for realistic desert scenes.
"""

import os
import numpy as np
import pybullet as p
from PIL import Image


def create_base_plane(texture_path="assets/Sand_terrain.jpg", plane_size=100.0):
    """Create a textured base plane (floor) underneath the terrain.
    
    Parameters
    ----------
    texture_path : str
        Path to the texture image file for the plane
    plane_size : float
        Size of the plane (will be plane_size x plane_size)
        
    Returns
    -------
    int
        PyBullet body ID of the created plane
    """
    # Create a large plane as the base floor
    plane_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[plane_size/2, plane_size/2, 0.01]  # Very thin box as plane
    )
    
    plane_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[plane_size/2, plane_size/2, 0.01],
        rgbaColor=[0.9, 0.85, 0.7, 1.0]  # Light sandy color as base
    )
    
    plane_body = p.createMultiBody(
        baseMass=0,  # Static
        baseCollisionShapeIndex=plane_shape,
        baseVisualShapeIndex=plane_visual,
        basePosition=[0, 0, -0.02]  # Slightly below ground level
    )
    
    # Apply texture if available
    if os.path.exists(texture_path):
        try:
            texture_id = p.loadTexture(texture_path)
            p.changeVisualShape(
                plane_body,
                -1,
                textureUniqueId=texture_id,
                rgbaColor=[1.0, 1.0, 1.0, 1.0]  # White to show texture naturally
            )
            print(f"[Base Plane] Created with texture: {texture_path}")
        except Exception as e:
            print(f"[Base Plane] Could not apply texture: {e}")
    else:
        print(f"[Base Plane] Texture not found: {texture_path}, using solid color")
    
    return plane_body


def create_desert_terrain(heightmap_path="assets/desert_heightmap.png", 
                          texture_path="assets/desert_sand.png",
                          terrain_scale=(0.15, 0.15, 1.5),
                          heightfield_texture_scaling=128,
                          base_height=0.0):
    """Create a desert terrain with dunes from a heightmap.
    
    Parameters
    ----------
    heightmap_path : str
        Path to heightmap PNG file (grayscale, 0=low, 255=high)
    texture_path : str
        Path to desert sand texture PNG file
    terrain_scale : tuple
        (x_scale, y_scale, z_scale) for terrain dimensions
    heightfield_texture_scaling : int
        Texture repeat scaling
        
    Returns
    -------
    int
        PyBullet body ID of the created terrain
        
    Notes
    -----
    The heightmap should be a square grayscale image. Higher pixel values
    represent higher elevations. Typical size is 256x256 pixels.
    """
    # Check if heightmap exists
    if not os.path.exists(heightmap_path):
        print(f"[Terrain] Heightmap not found: {heightmap_path}")
        print("[Terrain] Using flat plane instead.")
        return None
    
    # Load heightmap
    try:
        img = Image.open(heightmap_path)
        if img.mode != 'L':
            img = img.convert('L')
        heightmap_data = np.array(img)
        
        # Normalize to [0, 1] range for PyBullet
        heightmap_normalized = heightmap_data.astype(float) / 255.0
        
        # Flatten for PyBullet (row-major order)
        heightmap_flat = heightmap_normalized.flatten()
        
        num_rows, num_cols = heightmap_data.shape
        
        print(f"[Terrain] Loaded heightmap: {num_rows}x{num_cols}")
        
    except Exception as e:
        print(f"[Terrain] Error loading heightmap: {e}")
        return None
    
    # Create collision shape
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=terrain_scale,
        heightfieldTextureScaling=heightfield_texture_scaling,
        heightfieldData=heightmap_flat.tolist(),
        numHeightfieldRows=num_rows,
        numHeightfieldColumns=num_cols,
        replaceHeightfieldIndex=-1
    )
    
    # Create terrain body (collision only, then apply visual properties)
    terrain_body = p.createMultiBody(
        baseMass=0,  # Static terrain
        baseCollisionShapeIndex=terrain_shape,
        basePosition=[0, 0, base_height]
    )
    
    # Apply texture to dunes if available
    if os.path.exists(texture_path):
        try:
            texture_id = p.loadTexture(texture_path)
            p.changeVisualShape(
                terrain_body,
                -1,
                textureUniqueId=texture_id,
                rgbaColor=[1.0, 1.0, 1.0, 1.0]  # White to show texture color naturally
            )
            print(f"[Terrain] Desert dunes created with texture: {texture_path}")
        except Exception as e:
            print(f"[Terrain] Could not apply texture: {e}")
            # Fallback to solid color
            p.changeVisualShape(
                terrain_body,
                -1,
                rgbaColor=[0.92, 0.83, 0.65, 1.0]  # Light tan/beige
            )
    else:
        # No texture found, use solid color
        p.changeVisualShape(
            terrain_body,
            -1,
            rgbaColor=[0.92, 0.83, 0.65, 1.0]  # Light tan/beige
        )
        print(f"[Terrain] Texture not found: {texture_path}, using solid color")
    
    print(f"[Terrain] Desert terrain created with ID: {terrain_body}")
    return terrain_body


def spawn_desert_obstacles(num_rocks=10, num_vegetation=5, area_size=30.0, 
                           exclude_positions=None, min_distance=3.0):
    """Spawn desert obstacles (rocks and vegetation) in the scene.
    
    Parameters
    ----------
    num_rocks : int
        Number of rock obstacles to spawn
    num_vegetation : int
        Number of vegetation obstacles to spawn
    area_size : float
        Size of the area to spawn obstacles (square, ±area_size/2)
    exclude_positions : list of tuples, optional
        List of (x, y, z) positions to avoid (e.g., waste locations)
    min_distance : float
        Minimum distance from excluded positions
        
    Returns
    -------
    list
        List of PyBullet body IDs for spawned obstacles
    """
    obstacle_ids = []
    
    if exclude_positions is None:
        exclude_positions = []
    
    # Rock types (using primitive shapes)
    rock_types = [
        {"shape": "cube", "scale": (0.5, 0.8), "color": [0.55, 0.47, 0.42, 1.0]},
        {"shape": "sphere", "scale": (0.3, 0.6), "color": [0.6, 0.52, 0.45, 1.0]},
        {"shape": "cylinder", "scale": (0.4, 0.7), "color": [0.5, 0.44, 0.38, 1.0]},
    ]
    
    print(f"[Obstacles] Spawning {num_rocks} rocks and {num_vegetation} vegetation...")
    
    # Spawn rocks
    for i in range(num_rocks):
        # Find valid position
        max_attempts = 50
        for attempt in range(max_attempts):
            x = np.random.uniform(-area_size/2, area_size/2)
            y = np.random.uniform(-area_size/2, area_size/2)
            z = 0.3  # Approximate ground level
            
            # Check distance from excluded positions
            valid_pos = True
            for ex_pos in exclude_positions:
                dist = np.sqrt((x - ex_pos[0])**2 + (y - ex_pos[1])**2)
                if dist < min_distance:
                    valid_pos = False
                    break
            
            if valid_pos:
                break
        
        if not valid_pos:
            continue  # Skip if no valid position found
        
        # Choose random rock type
        rock = rock_types[np.random.randint(0, len(rock_types))]
        scale = np.random.uniform(rock["scale"][0], rock["scale"][1])
        
        # Create rock
        if rock["shape"] == "cube":
            col_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[scale*0.6, scale*0.4, scale*0.3]
            )
            vis_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[scale*0.6, scale*0.4, scale*0.3],
                rgbaColor=rock["color"]
            )
        elif rock["shape"] == "sphere":
            col_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=scale
            )
            vis_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=scale,
                rgbaColor=rock["color"]
            )
        else:  # cylinder
            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=scale*0.4,
                height=scale*0.8
            )
            vis_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=scale*0.4,
                length=scale*0.8,
                rgbaColor=rock["color"]
            )
        
        # Random orientation
        orientation = p.getQuaternionFromEuler([
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0, 2*np.pi)
        ])
        
        rock_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[x, y, z],
            baseOrientation=orientation
        )
        
        obstacle_ids.append(rock_id)
    
    # Spawn vegetation (dead bushes/trees)
    for i in range(num_vegetation):
        # Find valid position
        max_attempts = 50
        for attempt in range(max_attempts):
            x = np.random.uniform(-area_size/2, area_size/2)
            y = np.random.uniform(-area_size/2, area_size/2)
            z = 0.2
            
            # Check distance from excluded positions
            valid_pos = True
            for ex_pos in exclude_positions:
                dist = np.sqrt((x - ex_pos[0])**2 + (y - ex_pos[1])**2)
                if dist < min_distance:
                    valid_pos = False
                    break
            
            if valid_pos:
                break
        
        if not valid_pos:
            continue
        
        # Create simple vegetation (cylinder for trunk, sphere for foliage)
        scale = np.random.uniform(0.3, 0.6)
        
        # Trunk
        trunk_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=scale*0.1,
            height=scale*1.5
        )
        trunk_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=scale*0.1,
            length=scale*1.5,
            rgbaColor=[0.55, 0.47, 0.35, 1.0]  # Brown
        )
        
        veg_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=trunk_col,
            baseVisualShapeIndex=trunk_vis,
            basePosition=[x, y, z]
        )
        
        obstacle_ids.append(veg_id)
    
    print(f"[Obstacles] Spawned {len(obstacle_ids)} obstacles successfully")
    return obstacle_ids


def get_terrain_height_at_position(terrain_body_id, x, y):
    """Get the terrain height at a specific (x, y) position.
    
    Parameters
    ----------
    terrain_body_id : int
        PyBullet body ID of the terrain
    x : float
        X coordinate
    y : float
        Y coordinate
        
    Returns
    -------
    float
        Height (z coordinate) of terrain at (x, y), or 0 if not found
    """
    if terrain_body_id is None:
        return 0.0
    
    # Ray cast from above to find terrain height
    ray_from = [x, y, 10.0]
    ray_to = [x, y, -1.0]
    
    result = p.rayTest(ray_from, ray_to)
    
    if result and len(result) > 0:
        hit_fraction = result[0][2]
        if hit_fraction < 1.0:
            # Calculate hit position
            height = 10.0 + hit_fraction * (-1.0 - 10.0)
            return height
    
    return 0.0
