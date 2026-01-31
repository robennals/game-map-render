# Game Map Tile Renderer

Generate beautiful, photorealistic tile images for game maps using local Stable Diffusion. Given a 16x12 ASCII grid of terrain types and a scene description, this tool generates seamlessly blending tiles that together form a coherent scene.

## Features

- **Context-aware tile generation**: Each tile is generated with awareness of its neighbors for seamless blending
- **Reference image guidance**: A low-resolution overview establishes global style and lighting
- **Photorealistic output**: Optimized prompts for realistic textures and natural lighting
- **Multiple terrain types**: Grass, water, trees, sand, path, wall, stone, lava, snow, and more
- **Flexible output**: Individual tile PNGs and assembled composite images

## Requirements

- Python 3.10+
- GPU with 8GB+ VRAM (12GB+ recommended for SDXL)
- ~10GB disk space for models

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd game-map-render

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
game-map-render examples/forest.json -o ./output
```

### Options

```
Usage: game-map-render [OPTIONS] INPUT_FILE

Options:
  -o, --output-dir TEXT      Output directory for generated images
  -t, --tile-size INTEGER    Size of each tile in pixels (default: 64)
  -s, --seed INTEGER         Random seed for reproducibility
  --steps INTEGER            Number of inference steps (default: 30)
  -g, --guidance-scale FLOAT Guidance scale for generation (default: 7.5)
  --no-individual-tiles      Skip saving individual tile images
  --debug-grid               Also save a debug image with grid overlay
  -n, --name TEXT            Base name for output files
  --device TEXT              Device: cuda, mps, or cpu (default: auto)
  --fp32                     Use full precision instead of fp16
  --help                     Show this message and exit.
```

### Input Format

Create a JSON file with the following structure:

```json
{
  "description": "A mystical forest glade at twilight with soft purple lighting",
  "tiles": [
    "TTTTTTTTTTTTTTTT",
    "TGGGGGGGGGGGGGGT",
    "TGGGGWWWWGGGGPGT",
    "TGGGWWWWWWGGGPGT",
    "TGGGGWWWWGGGGPGT",
    "TGGGGGGGGGGGPPGT",
    "TGGGGGGGGGGPPGGT",
    "TGGGGGGGGGPPGGGT",
    "TGGGSSSSSSPPGGGT",
    "TGGGSSSSSSPPGGGT",
    "TGGGGGGGGGGGGGGT",
    "TTTTTTTTTTTTTTTT"
  ],
  "legend": {
    "T": "trees",
    "G": "grass",
    "W": "water",
    "P": "path",
    "S": "sand"
  }
}
```

### Supported Terrain Types

| Type | Description |
|------|-------------|
| `grass` | Lush green grass |
| `water` | Clear water with ripples |
| `trees` | Dense forest canopy |
| `wall` | Stone wall texture |
| `sand` | Beach sand |
| `path` | Dirt path |
| `stone` | Rocky ground |
| `lava` | Molten lava |
| `snow` | Fresh white snow |
| `dirt` | Bare earth |
| `bridge` | Wooden planks |
| `flowers` | Wildflower meadow |

## Output

The tool generates:

1. **Composite image**: `{name}_composite.png` - Full assembled map
2. **Individual tiles**: `{name}_tiles/tile_XX_YY.png` - Each tile separately
3. **Debug grid** (optional): `{name}_debug_grid.png` - Composite with grid overlay

For a 16x12 map with 64px tiles, the composite image is 1024x768 pixels.

## How It Works

1. **Parse the ASCII map** into terrain types using the legend
2. **Generate a reference image** - low-resolution scene overview for style consistency
3. **Generate tiles in scanline order** (top-left to bottom-right):
   - Extract context from already-generated neighbors (left, top, top-left)
   - Use inpainting to blend with neighbor edges
   - Apply terrain-specific prompts with scene description
4. **Assemble tiles** into final composite image

## Examples

See the `examples/` directory for sample input files:

- `forest.json` - Mystical forest glade with pond
- `dungeon.json` - Dark stone dungeon with lava
- `beach.json` - Tropical beach scene

## Architecture

```
src/
├── main.py              # CLI entry point
├── config.py            # Configuration settings
├── types.py             # Type definitions
├── tilemap.py           # ASCII map parsing
├── generation/
│   ├── pipeline.py      # SDXL pipeline wrapper
│   ├── reference.py     # Reference image generation
│   ├── tile_gen.py      # Individual tile generation
│   └── prompts.py       # Prompt templates
└── render/
    ├── compositor.py    # Tile assembly
    └── output.py        # File output
```

## License

MIT
