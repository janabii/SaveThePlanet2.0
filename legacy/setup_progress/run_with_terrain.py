"""
Run Simulation with Different Terrains
=======================================

Wrapper script to easily run the simulation with different terrain types.

Usage:
    python run_with_terrain.py dunes
    python run_with_terrain.py mountains
    python run_with_terrain.py canyon --no-gui
"""

import sys
import os
import argparse

# Terrain configurations
TERRAINS = {
    'dunes': {
        'heightmap': 'assets/terrain_desert_dunes.png',
        'scale': (0.15, 0.15, 3.0),
        'description': 'Smooth desert sand dunes'
    },
    'mountains': {
        'heightmap': 'assets/terrain_mountains.png',
        'scale': (0.15, 0.15, 5.0),
        'description': 'Rocky mountain peaks'
    },
    'canyon': {
        'heightmap': 'assets/terrain_canyon.png',
        'scale': (0.15, 0.15, 4.0),
        'description': 'Deep canyon valleys'
    },
    'hills': {
        'heightmap': 'assets/terrain_hills.png',
        'scale': (0.15, 0.15, 2.0),
        'description': 'Gentle rolling hills'
    },
    'valleys': {
        'heightmap': 'assets/terrain_valleys.png',
        'scale': (0.15, 0.15, 3.5),
        'description': 'Plateaus and valleys'
    },
    'original': {
        'heightmap': 'assets/desert_heightmap.png',
        'scale': (0.15, 0.15, 3.0),
        'description': 'Original desert terrain'
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='Run drone simulation with different terrain types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Terrains:
  dunes      - Smooth desert dunes (default)
  mountains  - Rocky peaks with ridges
  canyon     - Deep canyons and valleys
  hills      - Gentle rolling terrain
  valleys    - Plateaus with terracing
  original   - Original desert heightmap

Examples:
  python run_with_terrain.py dunes
  python run_with_terrain.py mountains --no-gui
        """
    )
    
    parser.add_argument('terrain', 
                        choices=list(TERRAINS.keys()),
                        nargs='?',
                        default='dunes',
                        help='Terrain type to use')
    parser.add_argument('--no-gui', 
                        action='store_true',
                        help='Run without GUI')
    
    args = parser.parse_args()
    
    # Get terrain config
    config = TERRAINS[args.terrain]
    
    print("\n" + "="*70)
    print(f"  Running Simulation with: {args.terrain.upper()} TERRAIN")
    print("="*70)
    print(f"  Description: {config['description']}")
    print(f"  Heightmap:   {config['heightmap']}")
    print(f"  Scale:       {config['scale']}")
    print("="*70 + "\n")
    
    # Check if heightmap exists
    if not os.path.exists(config['heightmap']):
        print(f"[ERROR] Heightmap not found: {config['heightmap']}")
        print("\nRun this first to generate terrains:")
        print("  python generate_example_terrain.py\n")
        sys.exit(1)
    
    # Set environment variables to pass terrain config to simulation
    os.environ['TERRAIN_HEIGHTMAP'] = config['heightmap']
    os.environ['TERRAIN_SCALE_X'] = str(config['scale'][0])
    os.environ['TERRAIN_SCALE_Y'] = str(config['scale'][1])
    os.environ['TERRAIN_SCALE_Z'] = str(config['scale'][2])
    
    print("[INFO] To use this terrain configuration:")
    print("       Edit swarm-mixed-fleet.py, line ~516:")
    print(f"\n       terrain_id = create_desert_terrain(")
    print(f"           heightmap_path='{config['heightmap']}',")
    print(f"           texture_path='assets/desert_sand.png',")
    print(f"           terrain_scale={config['scale']}")
    print(f"       )\n")
    
    # Note: To fully automate this, swarm-mixed-fleet.py would need to be modified
    # to read these environment variables or accept command-line arguments
    print("[NOTE] Currently you need to manually edit swarm-mixed-fleet.py")
    print("       A future update could add command-line terrain selection.\n")


if __name__ == "__main__":
    main()
