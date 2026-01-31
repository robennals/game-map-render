#!/usr/bin/env python3
"""Simple test to verify the pipeline generates coherent images."""

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

# Auto-detect device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load SDXL Turbo
print("Loading SDXL Turbo...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to(device)

# Generate a simple test image
prompt = "A mystical forest glade at twilight, aerial view, photorealistic, top-down photograph"

print(f"Generating image with prompt: {prompt}")
print("Using 4 steps, guidance_scale=0.0 (turbo settings)")

result = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    width=512,
    height=512,
)

image = result.images[0]
image.save("output/test_simple.png")
print(f"Saved to output/test_simple.png")
print(f"Image size: {image.size}")
