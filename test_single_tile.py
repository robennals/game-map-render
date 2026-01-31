#!/usr/bin/env python3
"""Test generating a single terrain tile at 512x512."""

import torch
from diffusers import AutoPipelineForText2Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to(device)

# Test different terrain prompts
terrains = [
    ("grass", "A mystical forest glade at twilight, lush green grass field, individual blades visible, aerial view, top-down photograph, photorealistic, 8k"),
    ("water", "A mystical forest glade at twilight, clear blue water surface, gentle ripples, reflections, aerial view, top-down photograph, photorealistic, 8k"),
    ("trees", "A mystical forest glade at twilight, dense forest canopy from above, tree tops, aerial view, top-down photograph, photorealistic, 8k"),
]

generator = torch.Generator(device=device).manual_seed(42)

for terrain, prompt in terrains:
    print(f"\nGenerating {terrain}...")
    print(f"Prompt: {prompt[:80]}...")

    result = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=512,
        height=512,
        generator=generator,
    )

    result.images[0].save(f"output/test_tile_{terrain}.png")
    print(f"Saved output/test_tile_{terrain}.png")
