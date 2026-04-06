"""
Generate Example Terrain Heightmaps
====================================

Creates realistic heightmap PNG files for testing without needing real ODM data.
Generates various terrain types using procedural noise algorithms.

Usage:
    python generate_example_terrain.py

Output:
    - assets/terrain_desert_dunes.png
    - assets/terrain_mountains.png
    - assets/terrain_canyon.png
    - assets/terrain_hills.png
    - assets/terrain_valleys.png
"""

import os
import numpy as np
from PIL import Image


def generate_simple_noise_2d(shape, scale=10, seed=None):
    """Generate simple 2D noise using sine waves.
    
    Parameters
    ----------
    shape : tuple
        Output shape (height, width)
    scale : float
        Frequency scale for noise
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    ndarray
        2D array of noise values in range [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = shape
    
    # Create coordinate grids
    x = np.linspace(0, scale, width)
    y = np.linspace(0, scale, height)
    X, Y = np.meshgrid(x, y)
    
    # Generate multiple layers of sine waves with random phases
    noise = np.zeros(shape)
    num_waves = 4
    
    for i in range(num_waves):
        # Random frequencies and phases
        fx = np.random.uniform(0.5, 2.0)
        fy = np.random.uniform(0.5, 2.0)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        
        # Add wave
        wave = np.sin(fx * X + phase_x) * np.sin(fy * Y + phase_y)
        noise += wave / num_waves
    
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def generate_fbm(shape, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """Generate Fractal Brownian Motion (FBM) noise.
    
    Parameters
    ----------
    shape : tuple
        Output shape (height, width)
    octaves : int
        Number of noise layers to combine
    persistence : float
        Amplitude multiplier per octave
    lacunarity : float
        Frequency multiplier per octave
    seed : int, optional
        Random seed
    
    Returns
    -------
    ndarray
        2D array of FBM values
    """
    noise = np.zeros(shape)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for i in range(octaves):
        scale = 10.0 * frequency
        octave_seed = seed + i * 100 if seed is not None else None
        noise += amplitude * generate_simple_noise_2d(shape, scale=scale, seed=octave_seed)
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    noise /= max_value
    return noise


def generate_desert_dunes(size=256, seed=42):
    """Generate desert dune terrain."""
    print(f"[Terrain] Generating desert dunes ({size}x{size})...")
    
    # Base rolling hills
    base = generate_fbm((size, size), octaves=4, persistence=0.6, lacunarity=2.0, seed=seed)
    
    # Add larger dune patterns
    dunes = generate_fbm((size, size), octaves=3, persistence=0.7, lacunarity=1.8, seed=seed+1)
    
    # Combine with emphasis on smooth undulations
    terrain = 0.4 * base + 0.6 * dunes
    
    # Apply gentle smoothing
    terrain = np.power(terrain, 0.8)
    
    return terrain


def generate_mountains(size=256, seed=42):
    """Generate mountainous terrain."""
    print(f"[Terrain] Generating mountains ({size}x{size})...")
    
    # High frequency rocky details
    base = generate_fbm((size, size), octaves=8, persistence=0.5, lacunarity=2.5, seed=seed)
    
    # Create ridges by taking absolute value
    ridges = 1 - np.abs(2 * base - 1)
    
    # Sharpen peaks
    terrain = np.power(ridges, 1.5)
    
    return terrain


def generate_canyon(size=256, seed=42):
    """Generate canyon terrain with valleys."""
    print(f"[Terrain] Generating canyon ({size}x{size})...")
    
    # Base elevation
    base = generate_fbm((size, size), octaves=5, persistence=0.6, lacunarity=2.0, seed=seed)
    
    # Create valleys by inverting and emphasizing low areas
    valleys = np.power(base, 2.0)
    
    # Add some variation to valley floors
    details = generate_fbm((size, size), octaves=6, persistence=0.4, lacunarity=2.5, seed=seed+1)
    
    terrain = 0.7 * valleys + 0.3 * details
    
    return terrain


def generate_hills(size=256, seed=42):
    """Generate gentle rolling hills."""
    print(f"[Terrain] Generating rolling hills ({size}x{size})...")
    
    # Smooth, gentle variations
    terrain = generate_fbm((size, size), octaves=4, persistence=0.65, lacunarity=1.8, seed=seed)
    
    # Make it very smooth
    terrain = np.power(terrain, 0.7)
    
    return terrain


def generate_valleys(size=256, seed=42):
    """Generate terrain with valleys and plateaus."""
    print(f"[Terrain] Generating valleys ({size}x{size})...")
    
    # Base terrain
    base = generate_fbm((size, size), octaves=5, persistence=0.5, lacunarity=2.0, seed=seed)
    
    # Create terracing effect for plateaus
    levels = 5
    terrain = np.floor(base * levels) / levels
    
    # Add subtle noise to plateau surfaces
    details = generate_fbm((size, size), octaves=6, persistence=0.3, lacunarity=2.5, seed=seed+1)
    terrain = 0.8 * terrain + 0.2 * details
    
    return terrain


def save_heightmap(terrain, filename, output_dir="assets"):
    """Save terrain as grayscale PNG heightmap.
    
    Parameters
    ----------
    terrain : ndarray
        2D array of terrain heights in range [0, 1]
    filename : str
        Output filename (e.g., 'terrain_dunes.png')
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize to 0-255
    heightmap = (terrain * 255).astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(heightmap, mode='L')
    output_path = os.path.join(output_dir, filename)
    img.save(output_path)
    
    print(f"[Terrain] Saved heightmap: {output_path}")
    print(f"          Elevation range: {terrain.min():.3f} to {terrain.max():.3f}")
    print(f"          Mean elevation: {terrain.mean():.3f}")


def main():
    """Generate all example terrains."""
    print("\n" + "="*60)
    print("  Terrain Heightmap Generator")
    print("="*60 + "\n")
    
    size = 256  # Output resolution
    seed = 42   # For reproducibility
    
    # Generate various terrain types
    terrains = [
        ("terrain_desert_dunes.png", generate_desert_dunes(size, seed)),
        ("terrain_mountains.png", generate_mountains(size, seed)),
        ("terrain_canyon.png", generate_canyon(size, seed)),
        ("terrain_hills.png", generate_hills(size, seed)),
        ("terrain_valleys.png", generate_valleys(size, seed)),
    ]
    
    # Save all terrains
    for filename, terrain in terrains:
        save_heightmap(terrain, filename)
        print()
    
    print("="*60)
    print("  Generation Complete!")
    print("="*60)
    print("\nUsage in simulation:")
    print("  terrain_id = create_desert_terrain(")
    print("      heightmap_path='assets/terrain_desert_dunes.png',")
    print("      texture_path='assets/desert_sand.png',")
    print("      terrain_scale=(0.15, 0.15, 3.0)")
    print("  )")
    print()


if __name__ == "__main__":
    main()
