"""
Generate a desert heightmap with dunes for PyBullet terrain.

This script creates a grayscale heightmap PNG where pixel brightness
represents elevation. The terrain features smooth dunes with realistic
desert topography.
"""

import numpy as np
from PIL import Image
import os

def generate_perlin_noise_2d(shape, res, seed=None):
    """Generate 2D Perlin-like noise for terrain generation.
    
    Parameters
    ----------
    shape : tuple
        Output shape (height, width)
    res : tuple
        Resolution/frequency of noise (periods in y, periods in x)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    ndarray
        2D array of noise values in [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_desert_heightmap(size=256, scale=2.0, octaves=4, seed=42):
    """Generate a desert terrain heightmap with multiple octaves of noise.
    
    Parameters
    ----------
    size : int
        Size of the heightmap (will be size x size pixels)
    scale : float
        Overall height scale (meters)
    octaves : int
        Number of noise octaves (detail levels)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    ndarray
        Heightmap as 2D array with values in [0, 255]
    """
    heightmap = np.zeros((size, size))
    
    # Generate multiple octaves of noise for natural-looking dunes
    for octave in range(octaves):
        frequency = 2 ** octave
        amplitude = 1.0 / (2 ** octave)
        
        # Generate noise layer
        noise_layer = generate_perlin_noise_2d(
            shape=(size, size),
            res=(4 * frequency, 4 * frequency),
            seed=seed + octave
        )
        
        # Add to heightmap with amplitude
        heightmap += noise_layer * amplitude
    
    # Normalize to [0, 1]
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    
    # Apply power curve for more pronounced dunes
    heightmap = np.power(heightmap, 1.5)
    
    # Add gentle overall slope (optional - for more natural appearance)
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    gentle_slope = 0.1 * (1 - (x**2 + y**2) / 2)  # Slight bowl shape
    heightmap = heightmap + gentle_slope
    
    # Re-normalize
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    
    # Scale to desired height range and convert to uint8
    # Height values: 0 = low, 255 = high
    heightmap_scaled = (heightmap * scale * 255 / 4.0).astype(np.uint8)
    
    # Add minimum elevation offset (to avoid ground plane clipping)
    heightmap_scaled = np.clip(heightmap_scaled + 20, 0, 255)
    
    return heightmap_scaled


if __name__ == "__main__":
    # Generate heightmap
    print("Generating desert heightmap...")
    heightmap = generate_desert_heightmap(size=256, scale=3.0, octaves=5, seed=42)
    
    # Save as PNG in assets folder
    output_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "desert_heightmap.png")
    
    # Create PIL Image and save
    img = Image.fromarray(heightmap, mode='L')  # 'L' mode for grayscale
    img.save(output_path)
    
    print(f"Desert heightmap saved to: {output_path}")
    print(f"Heightmap size: {heightmap.shape}")
    print(f"Height range: {heightmap.min()} to {heightmap.max()}")
    print("\nTo use in PyBullet:")
    print("1. Load as collision shape with p.createCollisionShape(p.GEOM_HEIGHTFIELD, ...)")
    print("2. Apply desert_sand.png texture for visual appearance")
