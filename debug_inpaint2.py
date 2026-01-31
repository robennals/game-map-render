#!/usr/bin/env python3
"""Inpainting v2: Use existing terrain tiles as fill context, not solid colors."""

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
from PIL import Image, ImageFilter
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

TILE_SIZE = 512
OVERLAP = 64

def get_prompt(terrain):
    base = "aerial photograph from 100 meters altitude, top-down view, photorealistic, natural daylight"
    prompts = {
        "trees": f"dense forest canopy, tree tops, {base}",
        "grass": f"grass field, lawn texture, {base}",
        "water": f"water surface, subtle ripples, {base}",
    }
    return prompts.get(terrain, f"{terrain} terrain, {base}")

print("Loading pipelines...")
txt2img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16",
)
txt2img.to(device)

inpaint = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16, variant="fp16",
)
inpaint.to(device)

def generate_tile_turbo(terrain, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    result = txt2img(
        prompt=get_prompt(terrain),
        num_inference_steps=4,
        guidance_scale=0.0,
        width=TILE_SIZE, height=TILE_SIZE,
        generator=gen,
    )
    return result.images[0]

def generate_with_inpaint(terrain, context_img, mask, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    result = inpaint(
        prompt=get_prompt(terrain),
        image=context_img,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=7.5,
        width=TILE_SIZE, height=TILE_SIZE,
        generator=gen,
    )
    return result.images[0]

# First, generate "template" tiles for each terrain type
print("\n=== Generating template tiles ===")
templates = {}
for terrain, seed in [("trees", 400), ("grass", 401), ("water", 402)]:
    print(f"  {terrain}...")
    templates[terrain] = generate_tile_turbo(terrain, seed)
    templates[terrain].save(f"output/template_{terrain}.png")

def create_context_and_mask(left_tile, top_tile, terrain):
    """Create context using template tile as base, paste neighbor edges on top."""
    # Start with a blurred version of the template for this terrain
    base = templates[terrain].copy()
    base = base.filter(ImageFilter.GaussianBlur(radius=8))

    context = base.copy()

    # Create mask - start all white (generate everything)
    mask = Image.new("L", (TILE_SIZE, TILE_SIZE), 255)

    # If we have a left neighbor, paste its right edge
    if left_tile is not None:
        right_edge = left_tile.crop((TILE_SIZE - OVERLAP, 0, TILE_SIZE, TILE_SIZE))
        context.paste(right_edge, (0, 0))
        # Mark left edge as "keep" (black)
        mask.paste(0, (0, 0, OVERLAP, TILE_SIZE))

    # If we have a top neighbor, paste its bottom edge
    if top_tile is not None:
        bottom_edge = top_tile.crop((0, TILE_SIZE - OVERLAP, TILE_SIZE, TILE_SIZE))
        context.paste(bottom_edge, (0, 0))
        # Mark top edge as "keep" (black)
        mask.paste(0, (0, 0, TILE_SIZE, OVERLAP))

    return context, mask

# Generate 2x2 grid
#   trees | grass
#   water | grass

print("\n=== Generating 2x2 grid ===")
tiles = {}

# (0,0) trees - no neighbors
print("\n[0,0] trees...")
tiles[(0,0)] = generate_tile_turbo("trees", seed=500)
tiles[(0,0)].save("output/inp2_0_0.png")

# (1,0) grass - left neighbor
print("\n[1,0] grass (left=trees)...")
ctx, mask = create_context_and_mask(tiles[(0,0)], None, "grass")
ctx.save("output/inp2_1_0_ctx.png")
mask.save("output/inp2_1_0_mask.png")
tiles[(1,0)] = generate_with_inpaint("grass", ctx, mask, seed=501)
tiles[(1,0)].save("output/inp2_1_0.png")

# (0,1) water - top neighbor
print("\n[0,1] water (top=trees)...")
ctx, mask = create_context_and_mask(None, tiles[(0,0)], "water")
ctx.save("output/inp2_0_1_ctx.png")
mask.save("output/inp2_0_1_mask.png")
tiles[(0,1)] = generate_with_inpaint("water", ctx, mask, seed=502)
tiles[(0,1)].save("output/inp2_0_1.png")

# (1,1) grass - left and top neighbors
print("\n[1,1] grass (left=water, top=grass)...")
ctx, mask = create_context_and_mask(tiles[(0,1)], tiles[(1,0)], "grass")
ctx.save("output/inp2_1_1_ctx.png")
mask.save("output/inp2_1_1_mask.png")
tiles[(1,1)] = generate_with_inpaint("grass", ctx, mask, seed=503)
tiles[(1,1)].save("output/inp2_1_1.png")

# Composite
print("\nCompositing...")
comp = Image.new("RGB", (TILE_SIZE*2, TILE_SIZE*2))
comp.paste(tiles[(0,0)], (0, 0))
comp.paste(tiles[(1,0)], (TILE_SIZE, 0))
comp.paste(tiles[(0,1)], (0, TILE_SIZE))
comp.paste(tiles[(1,1)], (TILE_SIZE, TILE_SIZE))
comp.save("output/inp2_composite.png")

print("Done! Check output/inp2_composite.png")
