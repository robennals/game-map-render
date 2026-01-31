#!/usr/bin/env python3
"""Generate a full scene image from a tilemap description (no tiling)."""

import argparse
import json
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path


def load_description(input_file: str) -> tuple[str, dict]:
    """Load description and tile info from JSON file."""
    with open(input_file) as f:
        data = json.load(f)

    description = data.get("description", "A game map scene")
    tiles = data.get("tiles", [])
    legend = data.get("legend", {})

    # Count terrain types
    terrain_counts = {}
    total = 0
    for row in tiles:
        for char in row:
            terrain = legend.get(char, "unknown")
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
            total += 1

    # Sort by frequency
    sorted_terrain = sorted(terrain_counts.items(), key=lambda x: x[1], reverse=True)

    return description, sorted_terrain, total


def build_prompt(description: str, terrain_info: list, total: int) -> str:
    """Build a prompt describing the full scene."""
    # Describe terrain composition
    terrain_parts = []
    for terrain, count in terrain_info[:4]:
        pct = (count / total) * 100
        if pct > 5:
            terrain_parts.append(f"{terrain} ({pct:.0f}%)")

    terrain_desc = ", ".join(terrain_parts) if terrain_parts else "mixed terrain"

    prompt = (
        f"{description}, "
        f"landscape with {terrain_desc}, "
        f"aerial view, top-down photograph, game map, "
        f"photorealistic, natural lighting, high detail, 8k uhd"
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate a full scene image")
    parser.add_argument("input_file", help="JSON file with description and tiles")
    parser.add_argument("-o", "--output", default="output/scene.png", help="Output file")
    parser.add_argument("-W", "--width", type=int, default=1024, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=768, help="Image height")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("--model", default="stabilityai/sdxl-turbo", help="Model to use")
    args = parser.parse_args()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load and build prompt
    description, terrain_info, total = load_description(args.input_file)
    prompt = build_prompt(description, terrain_info, total)

    print(f"Description: {description}")
    print(f"Terrain: {terrain_info[:4]}")
    print(f"Full prompt: {prompt}")
    print()

    # Load model
    print(f"Loading model: {args.model}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)

    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Generate
    print(f"Generating {args.width}x{args.height} image with {args.steps} steps...")
    result = pipe(
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
    )

    # Save
    image = result.images[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
