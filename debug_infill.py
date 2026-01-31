#!/usr/bin/env python3
"""Test infill/outpainting for seamless tiles."""

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load pipelines
print("Loading pipelines...")
txt2img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
txt2img.to(device)

img2img = AutoPipelineForImage2Image.from_pipe(txt2img)

TILE_SIZE = 512
OVERLAP = 64  # How much of neighbor's edge to use

def get_prompt(terrain):
    # Consistent height/scale across all tiles
    base = "aerial photograph from 100 meters altitude, top-down view, consistent scale, photorealistic, twilight lighting"
    prompts = {
        "trees": f"dense forest canopy filling entire frame, tree tops visible, {base}",
        "grass": f"grass field filling entire frame, lawn texture, {base}",
        "water": f"calm water surface filling entire frame, subtle ripples, {base}",
    }
    return prompts.get(terrain, f"{terrain} terrain filling frame, {base}")

def generate_first_tile(terrain, seed):
    """Generate a tile with no neighbors."""
    gen = torch.Generator(device=device).manual_seed(seed)
    result = txt2img(
        prompt=get_prompt(terrain),
        num_inference_steps=4,
        guidance_scale=0.0,
        width=TILE_SIZE,
        height=TILE_SIZE,
        generator=gen,
    )
    return result.images[0]

def generate_with_context(terrain, context_img, seed):
    """Generate a tile using context from neighbors via img2img."""
    gen = torch.Generator(device=device).manual_seed(seed)

    # Use img2img with the context image
    # Higher strength = more change from context, lower = keep more of context
    result = img2img(
        prompt=get_prompt(terrain),
        image=context_img,
        strength=0.8,  # Keep some of the context edges
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=gen,
    )
    return result.images[0]

def create_context_from_left(left_tile, terrain):
    """Create context image with left edge from neighbor, rest is noise-ish."""
    context = Image.new("RGB", (TILE_SIZE, TILE_SIZE))

    # Copy right edge of left tile to left edge of context
    right_strip = left_tile.crop((TILE_SIZE - OVERLAP, 0, TILE_SIZE, TILE_SIZE))
    context.paste(right_strip, (0, 0))

    # Fill rest with a solid color hint based on terrain
    colors = {"trees": (50, 80, 50), "grass": (80, 150, 80), "water": (60, 100, 120)}
    fill_color = colors.get(terrain, (128, 128, 128))

    fill = Image.new("RGB", (TILE_SIZE - OVERLAP, TILE_SIZE), fill_color)
    context.paste(fill, (OVERLAP, 0))

    return context

def create_context_from_top(top_tile, terrain):
    """Create context image with top edge from neighbor."""
    context = Image.new("RGB", (TILE_SIZE, TILE_SIZE))

    # Copy bottom edge of top tile to top edge of context
    bottom_strip = top_tile.crop((0, TILE_SIZE - OVERLAP, TILE_SIZE, TILE_SIZE))
    context.paste(bottom_strip, (0, 0))

    # Fill rest
    colors = {"trees": (50, 80, 50), "grass": (80, 150, 80), "water": (60, 100, 120)}
    fill_color = colors.get(terrain, (128, 128, 128))

    fill = Image.new("RGB", (TILE_SIZE, TILE_SIZE - OVERLAP), fill_color)
    context.paste(fill, (0, OVERLAP))

    return context

def create_context_from_left_and_top(left_tile, top_tile, terrain):
    """Create context with both left and top edges from neighbors."""
    context = Image.new("RGB", (TILE_SIZE, TILE_SIZE))

    # Fill with terrain color first
    colors = {"trees": (50, 80, 50), "grass": (80, 150, 80), "water": (60, 100, 120)}
    fill_color = colors.get(terrain, (128, 128, 128))
    context.paste(fill_color, (0, 0, TILE_SIZE, TILE_SIZE))

    # Left edge from left neighbor
    right_strip = left_tile.crop((TILE_SIZE - OVERLAP, 0, TILE_SIZE, TILE_SIZE))
    context.paste(right_strip, (0, 0))

    # Top edge from top neighbor
    bottom_strip = top_tile.crop((0, TILE_SIZE - OVERLAP, TILE_SIZE, TILE_SIZE))
    context.paste(bottom_strip, (0, 0))

    return context

# Test: Generate a 2x2 grid
# Layout:
#   trees | grass
#   water | grass

print("\n=== Generating 2x2 grid with infill ===")

grid = {}

# (0,0) - trees - no neighbors
print("Generating (0,0) trees...")
grid[(0,0)] = generate_first_tile("trees", seed=100)
grid[(0,0)].save("output/infill_0_0_trees.png")

# (1,0) - grass - has left neighbor (trees)
print("Generating (1,0) grass with left context...")
context = create_context_from_left(grid[(0,0)], "grass")
context.save("output/infill_1_0_context.png")  # Save context for debugging
grid[(1,0)] = generate_with_context("grass", context, seed=101)
grid[(1,0)].save("output/infill_1_0_grass.png")

# (0,1) - water - has top neighbor (trees)
print("Generating (0,1) water with top context...")
context = create_context_from_top(grid[(0,0)], "water")
context.save("output/infill_0_1_context.png")
grid[(0,1)] = generate_with_context("water", context, seed=102)
grid[(0,1)].save("output/infill_0_1_water.png")

# (1,1) - grass - has left (water) and top (grass) neighbors
print("Generating (1,1) grass with left+top context...")
context = create_context_from_left_and_top(grid[(0,1)], grid[(1,0)], "grass")
context.save("output/infill_1_1_context.png")
grid[(1,1)] = generate_with_context("grass", context, seed=103)
grid[(1,1)].save("output/infill_1_1_grass.png")

# Create composite
print("\nCreating composite...")
composite = Image.new("RGB", (TILE_SIZE * 2, TILE_SIZE * 2))
composite.paste(grid[(0,0)], (0, 0))
composite.paste(grid[(1,0)], (TILE_SIZE, 0))
composite.paste(grid[(0,1)], (0, TILE_SIZE))
composite.paste(grid[(1,1)], (TILE_SIZE, TILE_SIZE))
composite.save("output/infill_composite.png")

print(f"Saved composite to output/infill_composite.png ({composite.size})")
print("Done!")
