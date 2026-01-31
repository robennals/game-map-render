#!/usr/bin/env python3
"""Generate tiles individually at 512x512 (no blending), then scale and composite."""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

# Terrain-specific prompts
TERRAIN_PROMPTS = {
    "grass": "lush green grass field, meadow, lawn",
    "water": "clear blue water surface, gentle ripples, pond",
    "trees": "dense forest canopy from above, tree tops, foliage",
    "wall": "stone wall texture, gray masonry",
    "sand": "sandy beach, golden sand",
    "path": "dirt path, brown earth trail",
    "stone": "rocky stone ground, gray rocks",
    "lava": "molten lava, glowing orange magma",
    "snow": "white snow cover, winter",
    "dirt": "brown dirt ground, soil",
    "bridge": "wooden planks, timber boards",
    "flowers": "colorful wildflower meadow",
}

STYLE = "aerial view, top-down, photorealistic, natural lighting, seamless texture"


def load_tilemap(input_file: str) -> tuple[str, list[list[str]], int, int]:
    """Load tilemap from JSON file."""
    with open(input_file) as f:
        data = json.load(f)

    description = data.get("description", "A game map")
    tile_rows = data.get("tiles", [])
    legend = data.get("legend", {})

    terrain_grid = []
    for row in tile_rows:
        terrain_row = [legend.get(char, "grass") for char in row]
        terrain_grid.append(terrain_row)

    height = len(terrain_grid)
    width = len(terrain_grid[0]) if terrain_grid else 0

    return description, terrain_grid, width, height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="JSON tilemap file")
    parser.add_argument("-o", "--output-dir", default="output")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--model", default="stabilityai/sdxl-turbo")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    description, terrain_grid, width, height = load_tilemap(args.input_file)
    total = width * height
    print(f"Map: {width}x{height} = {total} tiles")
    print(f"Description: {description}")

    print(f"Loading {args.model}...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device)

    tiles: dict[tuple[int, int], Image.Image] = {}
    start = time.time()

    for y in range(height):
        for x in range(width):
            idx = y * width + x + 1
            terrain = terrain_grid[y][x]
            terrain_desc = TERRAIN_PROMPTS.get(terrain, terrain)

            prompt = f"{description}, {terrain_desc}, {STYLE}"

            # Fresh generator for each tile
            tile_seed = args.seed + y * width + x
            gen = torch.Generator(device=device).manual_seed(tile_seed)

            result = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                width=args.gen_size,
                height=args.gen_size,
                generator=gen,
            )

            tiles[(x, y)] = result.images[0]

            elapsed = time.time() - start
            remaining = (elapsed / idx) * (total - idx)
            print(f"\r[{idx}/{total}] {terrain:8s} - {elapsed:.0f}s elapsed, ~{remaining:.0f}s left", end="")

    print(f"\n\nDone in {time.time() - start:.0f}s")

    # Save composite
    print("Creating composites...")

    # Small composite
    small = Image.new("RGB", (width * args.tile_size, height * args.tile_size))
    for (x, y), tile in tiles.items():
        scaled = tile.resize((args.tile_size, args.tile_size), Image.Resampling.LANCZOS)
        small.paste(scaled, (x * args.tile_size, y * args.tile_size))
    small.save(output_dir / "composite_small.png")
    print(f"Saved {output_dir}/composite_small.png ({small.size})")

    # Large composite
    large = Image.new("RGB", (width * args.gen_size, height * args.gen_size))
    for (x, y), tile in tiles.items():
        large.paste(tile, (x * args.gen_size, y * args.gen_size))
    large.save(output_dir / "composite_large.png")
    print(f"Saved {output_dir}/composite_large.png ({large.size})")


if __name__ == "__main__":
    main()
