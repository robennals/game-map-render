#!/usr/bin/env python3
"""Debug: Generate uniform terrain tiles."""

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

# More tile-focused prompts - uniform terrain filling the frame
tiles = [
    ("trees", "dense forest canopy filling entire frame, tree tops only, no ground visible, aerial view, top-down, photorealistic, twilight purple lighting"),
    ("grass", "grass field filling entire frame, lawn texture, no trees, aerial view, top-down, photorealistic, twilight lighting"),
    ("water", "water surface filling entire frame, pond water, ripples, no shore, aerial view, top-down, photorealistic, twilight lighting"),
    ("path", "dirt path texture filling entire frame, brown earth, no grass, aerial view, top-down, photorealistic, twilight lighting"),
]

for i, (name, prompt) in enumerate(tiles):
    print(f"\n=== {name} ===")
    print(f"Prompt: {prompt[:80]}...")

    gen = torch.Generator(device=device).manual_seed(42 + i)

    result = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=512,
        height=512,
        generator=gen,
    )

    result.images[0].save(f"output/tile_uniform_{name}.png")
    print(f"Saved output/tile_uniform_{name}.png")

print("\nDone!")
