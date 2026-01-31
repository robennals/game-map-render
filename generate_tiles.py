#!/usr/bin/env python3
"""Generate tiles individually at 512x512, then scale down and composite."""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image

# Terrain-specific prompts for photorealistic output
TERRAIN_PROMPTS = {
    "grass": "lush green grass field, individual blades visible, natural variation, meadow",
    "water": "clear blue water surface, gentle ripples, reflections, pond or lake",
    "trees": "dense forest canopy from above, tree tops, varied green foliage, shadows",
    "wall": "stone wall texture, weathered gray masonry, castle or dungeon wall",
    "sand": "sandy beach or desert, fine golden sand, natural patterns",
    "path": "dirt path or trail, worn brown earth, footpath through nature",
    "stone": "rocky stone ground, gray boulders and pebbles, rough terrain",
    "lava": "molten lava flow, glowing orange magma, dark volcanic rock",
    "snow": "fresh white snow cover, pristine winter surface, subtle shadows",
    "dirt": "bare brown dirt ground, earthy soil, natural earth",
    "bridge": "wooden plank bridge from above, weathered timber boards",
    "flowers": "colorful wildflower meadow, blooming flowers in grass",
}

STYLE_SUFFIX = "aerial view, top-down photograph, photorealistic, natural lighting, seamless texture, 8k"
NEGATIVE_PROMPT = "cartoon, illustration, painting, anime, text, watermark, blurry, low quality"


def load_tilemap(input_file: str) -> tuple[str, list[list[str]], int, int]:
    """Load tilemap from JSON file."""
    with open(input_file) as f:
        data = json.load(f)

    description = data.get("description", "A game map")
    tile_rows = data.get("tiles", [])
    legend = data.get("legend", {})

    # Convert to terrain names
    terrain_grid = []
    for row in tile_rows:
        terrain_row = [legend.get(char, "grass") for char in row]
        terrain_grid.append(terrain_row)

    height = len(terrain_grid)
    width = len(terrain_grid[0]) if terrain_grid else 0

    return description, terrain_grid, width, height


def build_tile_prompt(terrain: str, scene_description: str) -> str:
    """Build prompt for a specific terrain tile."""
    terrain_desc = TERRAIN_PROMPTS.get(terrain, f"{terrain} terrain")
    return f"{scene_description}, {terrain_desc}, {STYLE_SUFFIX}"


def main():
    parser = argparse.ArgumentParser(description="Generate tilemap with individual tile generation")
    parser.add_argument("input_file", help="JSON file with tilemap")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--tile-size", type=int, default=64, help="Final tile size in composite")
    parser.add_argument("--gen-size", type=int, default=512, help="Generation size (before scaling)")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("--model", default="stabilityai/sdxl-turbo", help="Model to use")
    parser.add_argument("--blend", type=float, default=0.3, help="Blend strength with neighbors (0-1)")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load tilemap
    description, terrain_grid, width, height = load_tilemap(args.input_file)
    total_tiles = width * height
    print(f"Loaded {width}x{height} tilemap ({total_tiles} tiles)")
    print(f"Description: {description}")

    # Load model
    print(f"Loading model: {args.model}")
    txt2img = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    txt2img.to(device)

    # Also prepare img2img for blending
    img2img = AutoPipelineForImage2Image.from_pipe(txt2img)

    # Generate tiles
    print(f"\nGenerating {total_tiles} tiles at {args.gen_size}x{args.gen_size}, scaling to {args.tile_size}x{args.tile_size}")

    # Store generated tiles for neighbor context
    generated_tiles: dict[tuple[int, int], Image.Image] = {}

    start_time = time.time()

    for y in range(height):
        for x in range(width):
            tile_idx = y * width + x + 1
            terrain = terrain_grid[y][x]

            # Build prompt
            prompt = build_tile_prompt(terrain, description)

            # Seed for this tile
            generator = None
            if args.seed is not None:
                tile_seed = args.seed + y * width + x
                generator = torch.Generator(device=device).manual_seed(tile_seed)

            # Check for neighbor context (for blending)
            has_left = x > 0 and (x - 1, y) in generated_tiles
            has_top = y > 0 and (x, y - 1) in generated_tiles

            if (has_left or has_top) and args.blend > 0:
                # Create context image from neighbors
                context = Image.new("RGB", (args.gen_size, args.gen_size), (128, 128, 128))

                if has_left:
                    left_tile = generated_tiles[(x - 1, y)]
                    # Take right edge of left neighbor
                    edge_width = args.gen_size // 4
                    left_edge = left_tile.crop((args.gen_size - edge_width, 0, args.gen_size, args.gen_size))
                    left_edge = left_edge.resize((edge_width, args.gen_size), Image.Resampling.LANCZOS)
                    context.paste(left_edge, (0, 0))

                if has_top:
                    top_tile = generated_tiles[(x, y - 1)]
                    # Take bottom edge of top neighbor
                    edge_height = args.gen_size // 4
                    top_edge = top_tile.crop((0, args.gen_size - edge_height, args.gen_size, args.gen_size))
                    top_edge = top_edge.resize((args.gen_size, edge_height), Image.Resampling.LANCZOS)
                    context.paste(top_edge, (0, 0))

                # Use img2img with context
                result = img2img(
                    prompt=prompt,
                    image=context,
                    strength=1.0 - args.blend,  # Lower strength = more influence from context
                    num_inference_steps=max(args.steps, 2),
                    guidance_scale=args.guidance,
                    generator=generator,
                )
                tile_img = result.images[0]
            else:
                # Generate from scratch
                result = txt2img(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    width=args.gen_size,
                    height=args.gen_size,
                    generator=generator,
                )
                tile_img = result.images[0]

            # Store full-size for neighbor context
            generated_tiles[(x, y)] = tile_img

            # Save scaled tile
            scaled_tile = tile_img.resize(
                (args.tile_size, args.tile_size),
                Image.Resampling.LANCZOS
            )
            tile_path = tiles_dir / f"tile_{x:02d}_{y:02d}.png"
            scaled_tile.save(tile_path)

            # Progress
            elapsed = time.time() - start_time
            per_tile = elapsed / tile_idx
            remaining = per_tile * (total_tiles - tile_idx)
            print(f"\r[{tile_idx}/{total_tiles}] {terrain:8s} @ ({x},{y}) - {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining", end="", flush=True)

    print(f"\n\nGeneration complete in {time.time() - start_time:.1f}s")

    # Create composite
    print("Creating composite image...")
    composite = Image.new("RGB", (width * args.tile_size, height * args.tile_size))

    for y in range(height):
        for x in range(width):
            tile_path = tiles_dir / f"tile_{x:02d}_{y:02d}.png"
            tile = Image.open(tile_path)
            composite.paste(tile, (x * args.tile_size, y * args.tile_size))

    composite_path = output_dir / "composite.png"
    composite.save(composite_path)
    print(f"Saved composite to {composite_path}")

    # Also save a larger version
    large_composite = Image.new("RGB", (width * args.gen_size, height * args.gen_size))
    for (x, y), tile in generated_tiles.items():
        large_composite.paste(tile, (x * args.gen_size, y * args.gen_size))
    large_path = output_dir / "composite_large.png"
    large_composite.save(large_path)
    print(f"Saved large composite to {large_path}")


if __name__ == "__main__":
    main()
