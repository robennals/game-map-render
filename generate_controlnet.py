#!/usr/bin/env python3
"""Generate game map images using ControlNet segmentation.

This script uses a ControlNet segmentation model to generate entire game level
images at once, using a color-coded segmentation map derived from the tilemap
to guide terrain placement.
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

from src.config import ControlNetConfig, DEFAULT_CONFIG
from src.generation import ControlNetGenerator, build_terrain_prompt, tilemap_to_segmentation
from src.tilemap import load_tilemap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate game map images using ControlNet segmentation"
    )
    parser.add_argument(
        "tilemap",
        type=str,
        help="Path to tilemap JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output image path (default: output/<tilemap_name>_controlnet.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--conditioning-scale",
        type=float,
        default=0.5,
        help="ControlNet conditioning scale (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Output image height (default: 768)",
    )
    parser.add_argument(
        "--save-segmentation",
        action="store_true",
        help="Save the segmentation map alongside the output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override the prompt (default: uses tilemap description)",
    )

    args = parser.parse_args()

    # Load tilemap
    tilemap_path = Path(args.tilemap)
    if not tilemap_path.exists():
        logger.error(f"Tilemap not found: {tilemap_path}")
        sys.exit(1)

    logger.info(f"Loading tilemap: {tilemap_path}")
    tilemap = load_tilemap(tilemap_path)
    logger.info(f"Tilemap loaded: {tilemap.width}x{tilemap.height} tiles")
    logger.info(f"Description: {tilemap.description}")

    # Create segmentation image
    output_size = (args.width, args.height)
    logger.info(f"Creating segmentation map: {output_size[0]}x{output_size[1]}")
    segmentation = tilemap_to_segmentation(tilemap, output_size=output_size)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(DEFAULT_CONFIG.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{tilemap_path.stem}_controlnet.png"

    # Save segmentation if requested
    if args.save_segmentation:
        seg_path = output_path.with_name(output_path.stem + "_segmentation.png")
        segmentation.save(seg_path)
        logger.info(f"Segmentation map saved: {seg_path}")

    # Build prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = build_terrain_prompt(tilemap, tilemap.description)

    # Add style suffix
    style_suffix = DEFAULT_CONFIG.prompt.style_suffix
    full_prompt = f"{prompt}, {style_suffix}"
    logger.info(f"Prompt: {full_prompt}")

    # Create generator
    config = ControlNetConfig()
    generator = ControlNetGenerator(
        controlnet_model=config.controlnet_model,
        base_model=config.base_model,
        use_fp16=config.use_fp16,
        device=config.device,
    )

    # Generate image
    logger.info("Generating image...")
    image = generator.generate(
        prompt=full_prompt,
        segmentation_image=segmentation,
        negative_prompt=DEFAULT_CONFIG.prompt.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.conditioning_scale,
        seed=args.seed,
    )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info(f"Image saved: {output_path}")

    # Clean up
    generator.unload()
    logger.info("Done!")


if __name__ == "__main__":
    main()
