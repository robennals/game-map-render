#!/usr/bin/env python3
"""Proper inpainting: fix neighbor edges, generate center."""

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
from PIL import Image
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

TILE_SIZE = 512
OVERLAP = 64  # Edge region to keep from neighbors

def get_prompt(terrain):
    base = "aerial photograph from 100 meters altitude, top-down view, photorealistic, natural daylight"
    prompts = {
        "trees": f"dense forest canopy, tree tops, {base}",
        "grass": f"grass field, lawn texture, {base}",
        "water": f"water surface, subtle ripples, {base}",
    }
    return prompts.get(terrain, f"{terrain} terrain, {base}")

# Load pipelines
print("Loading txt2img (SDXL Turbo for first tile)...")
txt2img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
txt2img.to(device)

print("Loading inpainting model (SDXL Inpainting)...")
inpaint = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
)
inpaint.to(device)

def generate_first_tile(terrain, seed):
    """Generate first tile with no neighbors using fast turbo."""
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

def generate_with_inpainting(terrain, context_img, mask, seed):
    """Generate tile using inpainting - edges are fixed by mask."""
    gen = torch.Generator(device=device).manual_seed(seed)

    result = inpaint(
        prompt=get_prompt(terrain),
        image=context_img,
        mask_image=mask,
        num_inference_steps=25,  # More steps for quality
        guidance_scale=7.5,
        width=TILE_SIZE,
        height=TILE_SIZE,
        generator=gen,
    )
    return result.images[0]

def create_context_and_mask_from_left(left_tile, fill_color=(128, 128, 128)):
    """Create context image and mask for tile with left neighbor.

    Context: left edge from neighbor, rest filled with neutral color
    Mask: black (keep) on left edge, white (generate) elsewhere
    """
    # Context image
    context = Image.new("RGB", (TILE_SIZE, TILE_SIZE), fill_color)
    right_edge = left_tile.crop((TILE_SIZE - OVERLAP, 0, TILE_SIZE, TILE_SIZE))
    context.paste(right_edge, (0, 0))

    # Mask: 0 = keep, 255 = generate
    mask = Image.new("L", (TILE_SIZE, TILE_SIZE), 255)  # All white (generate)
    # Black out the left edge (keep it)
    mask.paste(0, (0, 0, OVERLAP, TILE_SIZE))

    return context, mask

def create_context_and_mask_from_top(top_tile, fill_color=(128, 128, 128)):
    """Create context and mask for tile with top neighbor."""
    context = Image.new("RGB", (TILE_SIZE, TILE_SIZE), fill_color)
    bottom_edge = top_tile.crop((0, TILE_SIZE - OVERLAP, TILE_SIZE, TILE_SIZE))
    context.paste(bottom_edge, (0, 0))

    mask = Image.new("L", (TILE_SIZE, TILE_SIZE), 255)
    mask.paste(0, (0, 0, TILE_SIZE, OVERLAP))  # Keep top edge

    return context, mask

def create_context_and_mask_from_left_and_top(left_tile, top_tile, fill_color=(128, 128, 128)):
    """Create context and mask for tile with both left and top neighbors."""
    context = Image.new("RGB", (TILE_SIZE, TILE_SIZE), fill_color)

    # Left edge
    right_edge = left_tile.crop((TILE_SIZE - OVERLAP, 0, TILE_SIZE, TILE_SIZE))
    context.paste(right_edge, (0, 0))

    # Top edge
    bottom_edge = top_tile.crop((0, TILE_SIZE - OVERLAP, TILE_SIZE, TILE_SIZE))
    context.paste(bottom_edge, (0, 0))

    # Mask
    mask = Image.new("L", (TILE_SIZE, TILE_SIZE), 255)
    mask.paste(0, (0, 0, OVERLAP, TILE_SIZE))  # Left edge
    mask.paste(0, (0, 0, TILE_SIZE, OVERLAP))  # Top edge

    return context, mask

# Test: 2x2 grid
#   trees | grass
#   water | grass

print("\n=== Generating 2x2 grid with proper inpainting ===")
tiles = {}

# (0,0) - trees - no neighbors, use fast turbo
print("\n[0,0] trees (no neighbors, using turbo)...")
tiles[(0,0)] = generate_first_tile("trees", seed=300)
tiles[(0,0)].save("output/inpaint_0_0.png")

# (1,0) - grass - left neighbor is trees
print("\n[1,0] grass (left neighbor: trees)...")
context, mask = create_context_and_mask_from_left(tiles[(0,0)], fill_color=(100, 150, 100))
context.save("output/inpaint_1_0_context.png")
mask.save("output/inpaint_1_0_mask.png")
tiles[(1,0)] = generate_with_inpainting("grass", context, mask, seed=301)
tiles[(1,0)].save("output/inpaint_1_0.png")

# (0,1) - water - top neighbor is trees
print("\n[0,1] water (top neighbor: trees)...")
context, mask = create_context_and_mask_from_top(tiles[(0,0)], fill_color=(80, 120, 140))
context.save("output/inpaint_0_1_context.png")
mask.save("output/inpaint_0_1_mask.png")
tiles[(0,1)] = generate_with_inpainting("water", context, mask, seed=302)
tiles[(0,1)].save("output/inpaint_0_1.png")

# (1,1) - grass - left neighbor is water, top neighbor is grass
print("\n[1,1] grass (left: water, top: grass)...")
context, mask = create_context_and_mask_from_left_and_top(tiles[(0,1)], tiles[(1,0)], fill_color=(100, 150, 100))
context.save("output/inpaint_1_1_context.png")
mask.save("output/inpaint_1_1_mask.png")
tiles[(1,1)] = generate_with_inpainting("grass", context, mask, seed=303)
tiles[(1,1)].save("output/inpaint_1_1.png")

# Composite
print("\nCreating composite...")
composite = Image.new("RGB", (TILE_SIZE * 2, TILE_SIZE * 2))
composite.paste(tiles[(0,0)], (0, 0))
composite.paste(tiles[(1,0)], (TILE_SIZE, 0))
composite.paste(tiles[(0,1)], (0, TILE_SIZE))
composite.paste(tiles[(1,1)], (TILE_SIZE, TILE_SIZE))
composite.save("output/inpaint_composite.png")

print(f"\nSaved composite to output/inpaint_composite.png")
print("Done!")
