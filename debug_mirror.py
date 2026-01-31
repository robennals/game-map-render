#!/usr/bin/env python3
"""Approach: Mirror/extend neighbor edges, use img2img to transform."""

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image, ImageFilter
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

TILE_SIZE = 512
EDGE_SIZE = 128  # How much of neighbor edge to use

def get_prompt(terrain):
    base = "aerial photograph, top-down view, photorealistic, natural daylight"
    prompts = {
        "trees": f"dense forest canopy, tree tops, {base}",
        "grass": f"grass lawn texture, {base}",
        "water": f"calm water surface, ripples, {base}",
    }
    return prompts.get(terrain, f"{terrain}, {base}")

print("Loading pipelines...")
txt2img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
txt2img.to(device)

img2img = AutoPipelineForImage2Image.from_pipe(txt2img)

def generate_turbo(terrain, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return txt2img(
        prompt=get_prompt(terrain),
        num_inference_steps=4, guidance_scale=0.0,
        width=TILE_SIZE, height=TILE_SIZE, generator=gen,
    ).images[0]

def img2img_transform(terrain, context, strength, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return img2img(
        prompt=get_prompt(terrain),
        image=context,
        strength=strength,
        num_inference_steps=4, guidance_scale=0.0,
        generator=gen,
    ).images[0]

def extend_from_left(left_tile):
    """Create full tile by mirroring/extending right edge of left neighbor."""
    # Take right portion of left tile
    edge = left_tile.crop((TILE_SIZE - EDGE_SIZE, 0, TILE_SIZE, TILE_SIZE))

    # Create new tile by mirroring this edge across
    result = Image.new("RGB", (TILE_SIZE, TILE_SIZE))

    # Paste original edge on left
    result.paste(edge, (0, 0))

    # Mirror and paste
    mirrored = edge.transpose(Image.FLIP_LEFT_RIGHT)
    result.paste(mirrored, (EDGE_SIZE, 0))

    # Continue pattern
    result.paste(edge, (EDGE_SIZE * 2, 0))
    result.paste(mirrored, (EDGE_SIZE * 3, 0))

    return result

def extend_from_top(top_tile):
    """Create full tile by mirroring/extending bottom edge of top neighbor."""
    edge = top_tile.crop((0, TILE_SIZE - EDGE_SIZE, TILE_SIZE, TILE_SIZE))

    result = Image.new("RGB", (TILE_SIZE, TILE_SIZE))
    result.paste(edge, (0, 0))

    mirrored = edge.transpose(Image.FLIP_TOP_BOTTOM)
    result.paste(mirrored, (0, EDGE_SIZE))
    result.paste(edge, (0, EDGE_SIZE * 2))
    result.paste(mirrored, (0, EDGE_SIZE * 3))

    return result

def extend_from_left_and_top(left_tile, top_tile):
    """Blend extensions from both neighbors."""
    left_ext = extend_from_left(left_tile)
    top_ext = extend_from_top(top_tile)

    # Blend them - use left_ext on left half, top_ext on top half, blend in middle
    result = Image.new("RGB", (TILE_SIZE, TILE_SIZE))

    left_arr = np.array(left_ext, dtype=np.float32)
    top_arr = np.array(top_ext, dtype=np.float32)

    # Create blend mask - gradient from left to right and top to bottom
    y, x = np.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
    # Weight towards left extension on the left, top extension on top
    left_weight = 1.0 - (x / TILE_SIZE)
    top_weight = 1.0 - (y / TILE_SIZE)

    # Normalize
    total = left_weight + top_weight + 0.001
    left_weight = left_weight / total
    top_weight = top_weight / total

    blended = (left_arr * left_weight[:,:,np.newaxis] +
               top_arr * top_weight[:,:,np.newaxis])

    return Image.fromarray(blended.astype(np.uint8))

# Generate 2x2 grid
print("\n=== Generating 2x2 grid ===")
tiles = {}

# (0,0) trees - no neighbors
print("\n[0,0] trees...")
tiles[(0,0)] = generate_turbo("trees", seed=600)
tiles[(0,0)].save("output/mir_0_0.png")

# (1,0) grass - left neighbor
print("\n[1,0] grass (extend from left)...")
context = extend_from_left(tiles[(0,0)])
context.save("output/mir_1_0_ctx.png")
# Higher strength to transform content while keeping edge structure
tiles[(1,0)] = img2img_transform("grass", context, strength=0.85, seed=601)
tiles[(1,0)].save("output/mir_1_0.png")

# (0,1) water - top neighbor
print("\n[0,1] water (extend from top)...")
context = extend_from_top(tiles[(0,0)])
context.save("output/mir_0_1_ctx.png")
tiles[(0,1)] = img2img_transform("water", context, strength=0.85, seed=602)
tiles[(0,1)].save("output/mir_0_1.png")

# (1,1) grass - both neighbors
print("\n[1,1] grass (extend from left+top)...")
context = extend_from_left_and_top(tiles[(0,1)], tiles[(1,0)])
context.save("output/mir_1_1_ctx.png")
tiles[(1,1)] = img2img_transform("grass", context, strength=0.85, seed=603)
tiles[(1,1)].save("output/mir_1_1.png")

# Composite
print("\nCompositing...")
comp = Image.new("RGB", (TILE_SIZE*2, TILE_SIZE*2))
comp.paste(tiles[(0,0)], (0, 0))
comp.paste(tiles[(1,0)], (TILE_SIZE, 0))
comp.paste(tiles[(0,1)], (0, TILE_SIZE))
comp.paste(tiles[(1,1)], (TILE_SIZE, TILE_SIZE))
comp.save("output/mir_composite.png")

print("Done! Check output/mir_composite.png")
