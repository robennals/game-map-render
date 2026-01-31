#!/usr/bin/env python3
"""Test overlap+blend approach for seamless tiles."""

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

print("Loading pipeline...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to(device)

TILE_SIZE = 512
OVERLAP = 64  # Overlap region for blending

def get_prompt(terrain):
    base = "aerial photograph from 100 meters altitude, top-down view, consistent scale, photorealistic, natural daylight"
    prompts = {
        "trees": f"dense forest canopy filling entire frame, tree tops visible, {base}",
        "grass": f"grass field filling entire frame, lawn texture, no trees, {base}",
        "water": f"calm water surface filling entire frame, subtle ripples, {base}",
    }
    return prompts.get(terrain, f"{terrain} terrain filling frame, {base}")

def generate_tile(terrain, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=get_prompt(terrain),
        num_inference_steps=4,
        guidance_scale=0.0,
        width=TILE_SIZE,
        height=TILE_SIZE,
        generator=gen,
    )
    return result.images[0]

def create_blend_mask(width, height, overlap, direction):
    """Create a gradient mask for blending.

    direction: 'left', 'top', 'right', 'bottom'
    """
    mask = np.ones((height, width), dtype=np.float32)

    if direction == 'left':
        for i in range(overlap):
            mask[:, i] = i / overlap
    elif direction == 'right':
        for i in range(overlap):
            mask[:, width - 1 - i] = i / overlap
    elif direction == 'top':
        for i in range(overlap):
            mask[i, :] = i / overlap
    elif direction == 'bottom':
        for i in range(overlap):
            mask[height - 1 - i, :] = i / overlap

    return mask

def blend_tiles(tile1, tile2, direction, overlap):
    """Blend two tiles with gradient at the seam.

    direction: which edge of tile1 meets tile2
    """
    arr1 = np.array(tile1, dtype=np.float32)
    arr2 = np.array(tile2, dtype=np.float32)

    if direction == 'right':
        # tile2 is to the right of tile1
        # Blend the right edge of tile1 with left edge of tile2
        mask = create_blend_mask(overlap, TILE_SIZE, overlap, 'left')
        mask = np.stack([mask] * 3, axis=-1)

        # Extract overlap regions
        region1 = arr1[:, -overlap:]
        region2 = arr2[:, :overlap]

        # Blend
        blended = region1 * (1 - mask) + region2 * mask

        # Create result by concatenating
        result = np.concatenate([
            arr1[:, :-overlap],
            blended,
            arr2[:, overlap:]
        ], axis=1)

    elif direction == 'bottom':
        # tile2 is below tile1
        mask = create_blend_mask(TILE_SIZE, overlap, overlap, 'top')
        mask = np.stack([mask] * 3, axis=-1)

        region1 = arr1[-overlap:, :]
        region2 = arr2[:overlap, :]

        blended = region1 * (1 - mask) + region2 * mask

        result = np.concatenate([
            arr1[:-overlap, :],
            blended,
            arr2[overlap:, :]
        ], axis=0)

    return Image.fromarray(result.astype(np.uint8))

# Test: Generate 2x2 grid with blending
print("\n=== Generating 2x2 grid ===")

# Layout:
#   trees | grass
#   water | grass

tiles = {}

print("Generating trees...")
tiles[(0,0)] = generate_tile("trees", seed=200)

print("Generating grass (top-right)...")
tiles[(1,0)] = generate_tile("grass", seed=201)

print("Generating water...")
tiles[(0,1)] = generate_tile("water", seed=202)

print("Generating grass (bottom-right)...")
tiles[(1,1)] = generate_tile("grass", seed=203)

# Save individual tiles
for (x, y), tile in tiles.items():
    tile.save(f"output/blend_tile_{x}_{y}.png")

# Create blended composite
print("\nBlending tiles...")

# First blend horizontally: top row and bottom row
top_row = blend_tiles(tiles[(0,0)], tiles[(1,0)], 'right', OVERLAP)
bottom_row = blend_tiles(tiles[(0,1)], tiles[(1,1)], 'right', OVERLAP)

top_row.save("output/blend_top_row.png")
bottom_row.save("output/blend_bottom_row.png")

# Then blend vertically
final = blend_tiles(top_row, bottom_row, 'bottom', OVERLAP)
final.save("output/blend_composite.png")

# Also create non-blended version for comparison
plain = Image.new("RGB", (TILE_SIZE * 2, TILE_SIZE * 2))
plain.paste(tiles[(0,0)], (0, 0))
plain.paste(tiles[(1,0)], (TILE_SIZE, 0))
plain.paste(tiles[(0,1)], (0, TILE_SIZE))
plain.paste(tiles[(1,1)], (TILE_SIZE, TILE_SIZE))
plain.save("output/blend_plain.png")

print(f"\nSaved:")
print(f"  output/blend_plain.png (no blending)")
print(f"  output/blend_composite.png (with blending)")
print("Done!")
