#!/usr/bin/env python3
"""Step 1: Verify basic SDXL generation works on MPS.

This is a minimal test to ensure we can generate images before adding complexity.
"""

import time
import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path


def main():
    print("Step 1: Basic SDXL Generation Test")
    print("=" * 50)

    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA")
    else:
        device = "cpu"
        print(f"Warning: Using CPU (will be slow)")

    # Load pipeline
    print("\nLoading SDXL pipeline...")
    start = time.time()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.to(device)

    print(f"Pipeline loaded in {time.time() - start:.1f}s")

    # Generate a simple test image
    prompt = "aerial view of a forest clearing with a pond, photorealistic, top-down"
    negative_prompt = "cartoon, illustration, blurry, low quality"

    print(f"\nGenerating test image...")
    print(f"Prompt: {prompt}")

    start = time.time()

    generator = torch.Generator(device=device).manual_seed(42)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    gen_time = time.time() - start
    print(f"Generated in {gen_time:.1f}s ({gen_time/20:.2f}s per step)")

    # Save output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "step1_basic_sdxl.png"
    image.save(output_path)
    print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 50)
    print("Step 1 PASSED: Basic SDXL generation works!")
    print("=" * 50)


if __name__ == "__main__":
    main()
