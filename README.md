# Game Map Tile Renderer

Generate photorealistic game map images from ASCII tilemaps using Stable Diffusion. Define terrain layouts with simple ASCII characters and get beautiful rendered maps with proper terrain placement and natural blending.

## Features

- **Regional prompt conditioning**: Each terrain type gets its own prompt, blended in latent space
- **Tilemap visualization**: Preview your layout before generation
- **Overlay comparison**: Verify terrain placement matches your tilemap
- **Multiple terrain types**: Grass, water, trees, sand, path, wall, stone, lava, snow, and more
- **Apple Silicon support**: Optimized for MPS (M1/M2/M3/M4 Macs)

## Requirements

- Python 3.9+
- Apple Silicon Mac with 16GB+ unified memory, or NVIDIA GPU with 8GB+ VRAM
- ~7GB disk space for SDXL model

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd game-map-render

# Install dependencies
pip install torch diffusers transformers accelerate pillow numpy

# (Optional) For Flux.1 model support, see "Using Flux" section below
```

## Quick Start

```bash
# Generate a map from a tilemap
./gamemap generate examples/forest.json

# Preview tilemap layout without generating (fast)
./gamemap visualize examples/forest.json

# Compare generated image with tilemap
./gamemap overlay output/forest_generated.png examples/forest.json
```

## Commands

### generate
Generate an AI image from a tilemap:
```bash
./gamemap generate examples/forest.json [options]

Options:
  -o, --output PATH      Output image path
  --seed INT             Random seed (default: 42)
  --steps INT            Inference steps (default: 25)
  --width INT            Output width (default: 1024)
  --height INT           Output height (default: 768)
  --guidance FLOAT       Guidance scale (default: 7.5)
  --cpu                  Force CPU (slower but more stable)
```

### visualize
Create a color-coded tilemap preview (no AI, instant):
```bash
./gamemap visualize examples/forest.json -o preview.png
```

### overlay
Overlay tilemap colors on an existing image for comparison:
```bash
./gamemap overlay output/forest_generated.png examples/forest.json
```

## Using Flux.1 (Better Quality)

Flux.1-dev has better prompt following than SDXL but requires HuggingFace authentication:

### Step 1: Create a HuggingFace Account
Go to https://huggingface.co/join and create an account.

### Step 2: Accept the Flux License
1. Go to https://huggingface.co/black-forest-labs/FLUX.1-dev
2. Click "Agree and access repository"

### Step 3: Create an Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "gamemap") and select "Read" access
4. Copy the token

### Step 4: Login via CLI
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

After authentication, edit `gamemap` to use Flux instead of SDXL.

## Input Format

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

## Output Files

The tool generates:

- `{name}_generated.png` - The AI-generated map image
- `{name}_tilemap.png` - Color-coded tilemap visualization
- `{name}_overlay.png` - Overlay comparison for verification

## How It Works

1. **Parse tilemap** - Load terrain layout from JSON
2. **Create terrain masks** - Generate soft-edged masks for each terrain region
3. **Regional prompt conditioning** - Run SDXL with different prompts per region
4. **Latent blending** - Blend noise predictions based on terrain masks
5. **Decode** - Convert latents to final image

## Examples

See the `examples/` directory:

- `forest.json` - Mystical forest glade with pond
- `dungeon.json` - Dark stone dungeon with lava
- `beach.json` - Tropical beach scene

## Troubleshooting

### Out of Memory on MPS (Apple Silicon)
The script uses float32 on MPS for stability. If you still hit memory issues:
- Close other applications
- Try `--cpu` flag (slower but stable)
- Reduce `--width` and `--height`

### Black/Corrupted Images
This usually means dtype issues. The script auto-detects MPS and uses float32.

### Terrain Bleeding
If terrain types bleed into each other:
- The description should focus on lighting/mood, not terrain content
- Terrain prompts in the code explicitly exclude other terrain types

## License

MIT
