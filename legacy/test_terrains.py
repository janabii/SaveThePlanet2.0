"""
Test Different Terrain Heightmaps
==================================

Quick script to test and visualize different terrain heightmaps in the simulation.

Usage:
    python test_terrains.py --terrain dunes
    python test_terrains.py --terrain mountains
    python test_terrains.py --terrain canyon
    python test_terrains.py --terrain hills
    python test_terrains.py --terrain valleys
"""

import argparse
from swarm_mixed_fleet import run

# Terrain configurations
TERRAINS = {
    'dunes': {
        'heightmap': 'assets/terrain_desert_dunes.png',
        'texture': 'assets/desert_sand.png',
        'scale': (0.15, 0.15, 3.0),
        'description': 'Smooth desert sand dunes'
    },
    'mountains': {
        'heightmap': 'assets/terrain_mountains.png',
        'texture': 'assets/desert_sand.png',
        'scale': (0.15, 0.15, 5.0),
        'description': 'Rocky mountain peaks with ridges'
    },
    'canyon': {
        'heightmap': 'assets/terrain_canyon.png',
        'texture': 'assets/desert_sand.png',
        'scale': (0.15, 0.15, 4.0),
        'description': 'Canyon with deep valleys'
    },
    'hills': {
        'heightmap': 'assets/terrain_hills.png',
        'texture': 'assets/desert_sand.png',
        'scale': (0.15, 0.15, 2.0),
        'description': 'Gentle rolling hills'
    },
    'valleys': {
        'heightmap': 'assets/terrain_valleys.png',
        'texture': 'assets/desert_sand.png',
        'scale': (0.15, 0.15, 3.5),
        'description': 'Plateaus and valleys with terracing'
    }
}


def main():
    parser = argparse.ArgumentParser(description='Test different terrain types')
    parser.add_argument('--terrain', '-t', 
                        choices=list(TERRAINS.keys()),
                        default='dunes',
                        help='Terrain type to load')
    parser.add_argument('--gui', type=bool, default=True,
                        help='Show GUI')
    
    args = parser.parse_args()
    
    # Get terrain config
    terrain_config = TERRAINS[args.terrain]
    
    print("\n" + "="*60)
    print(f"  Testing Terrain: {args.terrain.upper()}")
    print("="*60)
    print(f"  Description: {terrain_config['description']}")
    print(f"  Heightmap: {terrain_config['heightmap']}")
    print(f"  Scale: {terrain_config['scale']}")
    print("="*60 + "\n")
    
    # Note: This would require modifying swarm-mixed-fleet.py to accept
    # terrain parameters, or you can manually edit the file
    print("[INFO] To use this terrain in swarm-mixed-fleet.py:")
    print(f"       Change create_desert_terrain() parameters to:")
    print(f"         heightmap_path='{terrain_config['heightmap']}'")
    print(f"         terrain_scale={terrain_config['scale']}")
    print()


if __name__ == "__main__":
    main()
