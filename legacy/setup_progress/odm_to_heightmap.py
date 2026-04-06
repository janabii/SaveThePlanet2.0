"""
OpenDroneMap DEM to Heightmap Converter
========================================

Converts OpenDroneMap DEM files (GeoTIFF format) to grayscale PNG heightmaps
compatible with PyBullet's heightfield terrain system.

Usage:
    python odm_to_heightmap.py --input path/to/dsm.tif --output assets/terrain.png

Requirements:
    pip install rasterio pillow numpy

Example:
    # Convert ODM Digital Surface Model to 512x512 heightmap
    python odm_to_heightmap.py --input odm_output/odm_dem/dsm.tif --output assets/my_terrain.png --size 512

    # With smoothing
    python odm_to_heightmap.py --input dsm.tif --output terrain.png --size 256 --smooth 2.0
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import rasterio
        return True
    except ImportError:
        print("[ERROR] Missing dependency: rasterio")
        print("        Install with: pip install rasterio")
        return False


def load_dem(input_path):
    """Load DEM file and extract elevation data.
    
    Parameters
    ----------
    input_path : str
        Path to GeoTIFF DEM file
    
    Returns
    -------
    tuple
        (elevation_array, metadata_dict)
    """
    try:
        import rasterio
    except ImportError:
        print("[ERROR] rasterio not installed. Install with: pip install rasterio")
        sys.exit(1)
    
    print(f"[DEM] Loading: {input_path}")
    
    try:
        with rasterio.open(input_path) as src:
            # Read elevation data
            elevation = src.read(1)  # Read first band
            
            # Get metadata
            metadata = {
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds,
                'crs': src.crs,
                'transform': src.transform,
                'nodata': src.nodata
            }
            
            print(f"[DEM] Dimensions: {metadata['width']} x {metadata['height']}")
            print(f"[DEM] CRS: {metadata['crs']}")
            print(f"[DEM] Bounds: {metadata['bounds']}")
            
            return elevation, metadata
            
    except Exception as e:
        print(f"[ERROR] Failed to load DEM: {e}")
        sys.exit(1)


def process_elevation_data(elevation, metadata, smooth_sigma=0.0):
    """Process elevation data for heightmap conversion.
    
    Parameters
    ----------
    elevation : ndarray
        Raw elevation data
    metadata : dict
        DEM metadata
    smooth_sigma : float
        Gaussian smoothing sigma (0 = no smoothing)
    
    Returns
    -------
    ndarray
        Processed elevation data
    """
    # Handle nodata values
    nodata = metadata.get('nodata')
    if nodata is not None:
        mask = elevation == nodata
        if mask.any():
            print(f"[DEM] Found {mask.sum()} nodata values")
            # Replace nodata with mean of valid values
            valid_mean = elevation[~mask].mean()
            elevation = elevation.copy()
            elevation[mask] = valid_mean
    
    # Remove invalid values (NaN, Inf)
    if np.isnan(elevation).any() or np.isinf(elevation).any():
        print("[DEM] Cleaning invalid values...")
        elevation = np.nan_to_num(elevation, nan=np.nanmean(elevation), 
                                   posinf=np.nanmax(elevation[np.isfinite(elevation)]),
                                   neginf=np.nanmin(elevation[np.isfinite(elevation)]))
    
    # Apply smoothing if requested
    if smooth_sigma > 0:
        print(f"[DEM] Applying Gaussian smoothing (sigma={smooth_sigma})...")
        try:
            from scipy.ndimage import gaussian_filter
            elevation = gaussian_filter(elevation, sigma=smooth_sigma)
        except ImportError:
            print("[WARNING] scipy not installed, skipping smoothing")
            print("          Install with: pip install scipy")
    
    return elevation


def resize_elevation(elevation, target_size):
    """Resize elevation data to target dimensions.
    
    Parameters
    ----------
    elevation : ndarray
        Elevation data
    target_size : int or tuple
        Target size (width, height) or single int for square
    
    Returns
    -------
    ndarray
        Resized elevation data
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    current_shape = elevation.shape
    if current_shape[0] == target_size[1] and current_shape[1] == target_size[0]:
        print(f"[DEM] Size already matches target: {target_size}")
        return elevation
    
    print(f"[DEM] Resizing from {current_shape} to {target_size}...")
    
    # Use PIL for high-quality resampling
    img = Image.fromarray(elevation)
    img_resized = img.resize(target_size, resample=Image.LANCZOS)
    elevation_resized = np.array(img_resized)
    
    return elevation_resized


def normalize_to_heightmap(elevation):
    """Normalize elevation data to 0-255 range for heightmap.
    
    Parameters
    ----------
    elevation : ndarray
        Elevation data in meters
    
    Returns
    -------
    tuple
        (heightmap_uint8, elevation_info_dict)
    """
    min_elev = elevation.min()
    max_elev = elevation.max()
    elev_range = max_elev - min_elev
    
    print(f"[DEM] Elevation range: {min_elev:.2f} to {max_elev:.2f} m ({elev_range:.2f} m total)")
    
    # Normalize to [0, 1]
    if elev_range > 0:
        normalized = (elevation - min_elev) / elev_range
    else:
        print("[WARNING] Flat terrain detected (no elevation variation)")
        normalized = np.zeros_like(elevation)
    
    # Convert to uint8 [0, 255]
    heightmap = (normalized * 255).astype(np.uint8)
    
    elevation_info = {
        'min': min_elev,
        'max': max_elev,
        'range': elev_range,
        'mean': elevation.mean(),
        'std': elevation.std()
    }
    
    return heightmap, elevation_info


def save_heightmap(heightmap, output_path, elevation_info):
    """Save heightmap as PNG with metadata.
    
    Parameters
    ----------
    heightmap : ndarray
        uint8 heightmap data
    output_path : str
        Output PNG file path
    elevation_info : dict
        Elevation statistics
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as grayscale PNG
    img = Image.fromarray(heightmap, mode='L')
    img.save(output_path)
    
    print(f"\n[SUCCESS] Saved heightmap: {output_path}")
    print(f"          Size: {heightmap.shape[1]} x {heightmap.shape[0]}")
    print(f"          Elevation: {elevation_info['min']:.2f} to {elevation_info['max']:.2f} m")
    print(f"          Mean: {elevation_info['mean']:.2f} m, Std: {elevation_info['std']:.2f} m")
    
    # Save metadata as text file
    metadata_path = output_path.replace('.png', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Heightmap Metadata\n")
        f.write("=" * 50 + "\n")
        f.write(f"Source: {output_path}\n")
        f.write(f"Size: {heightmap.shape[1]} x {heightmap.shape[0]}\n")
        f.write(f"\nElevation Statistics:\n")
        f.write(f"  Minimum: {elevation_info['min']:.2f} m\n")
        f.write(f"  Maximum: {elevation_info['max']:.2f} m\n")
        f.write(f"  Range:   {elevation_info['range']:.2f} m\n")
        f.write(f"  Mean:    {elevation_info['mean']:.2f} m\n")
        f.write(f"  Std Dev: {elevation_info['std']:.2f} m\n")
        f.write(f"\nUsage in PyBullet:\n")
        f.write(f"  terrain_id = create_desert_terrain(\n")
        f.write(f"      heightmap_path='{output_path}',\n")
        f.write(f"      texture_path='assets/desert_sand.png',\n")
        f.write(f"      terrain_scale=(0.15, 0.15, 3.0)  # Adjust as needed\n")
        f.write(f"  )\n")
    
    print(f"          Metadata: {metadata_path}")


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description='Convert OpenDroneMap DEM to PyBullet heightmap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python odm_to_heightmap.py --input dsm.tif --output terrain.png
  
  # Custom size with smoothing
  python odm_to_heightmap.py --input dsm.tif --output terrain.png --size 512 --smooth 2.0
  
  # Using DTM instead of DSM
  python odm_to_heightmap.py --input dtm.tif --output terrain_ground.png
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input DEM file (GeoTIFF format: dsm.tif or dtm.tif)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output heightmap PNG file')
    parser.add_argument('--size', '-s', type=int, default=256,
                        help='Output size in pixels (square) (default: 256)')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='Gaussian smoothing sigma (0=no smoothing) (default: 0.0)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  OpenDroneMap DEM to Heightmap Converter")
    print("="*60 + "\n")
    
    # Load DEM
    elevation, metadata = load_dem(args.input)
    
    # Process elevation data
    elevation = process_elevation_data(elevation, metadata, args.smooth)
    
    # Resize if needed
    elevation = resize_elevation(elevation, args.size)
    
    # Normalize to heightmap
    heightmap, elevation_info = normalize_to_heightmap(elevation)
    
    # Save heightmap
    save_heightmap(heightmap, args.output, elevation_info)
    
    print("\n" + "="*60)
    print("  Conversion Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
