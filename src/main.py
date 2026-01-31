#!/usr/bin/env python3
"""CLI entry point for the game map tile renderer."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .config import Config, GenerationConfig, ModelConfig, PromptConfig
from .generation.pipeline import SDXLPipeline
from .generation.tile_gen import TileGenerator
from .render.output import OutputWriter
from .tilemap import load_tilemap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Output directory for generated images",
)
@click.option(
    "--tile-size",
    "-t",
    default=64,
    type=int,
    help="Size of each tile in pixels (default: 64)",
)
@click.option(
    "--seed",
    "-s",
    default=None,
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--steps",
    default=4,
    type=int,
    help="Number of inference steps (default: 4 for turbo, 30 for base SDXL)",
)
@click.option(
    "--guidance-scale",
    "-g",
    default=0.0,
    type=float,
    help="Guidance scale for generation (default: 0.0 for turbo, 7.5 for base SDXL)",
)
@click.option(
    "--no-individual-tiles",
    is_flag=True,
    help="Skip saving individual tile images",
)
@click.option(
    "--debug-grid",
    is_flag=True,
    help="Also save a debug image with grid overlay",
)
@click.option(
    "--name",
    "-n",
    default=None,
    help="Base name for output files (default: input filename)",
)
@click.option(
    "--device",
    default=None,
    help="Device to run on: cuda, mps, or cpu (default: auto-detect)",
)
@click.option(
    "--fp32",
    is_flag=True,
    help="Use full precision (fp32) instead of fp16",
)
def generate(
    input_file: str,
    output_dir: str,
    tile_size: int,
    seed: Optional[int],
    steps: int,
    guidance_scale: float,
    no_individual_tiles: bool,
    debug_grid: bool,
    name: Optional[str],
    device: Optional[str],
    fp32: bool,
):
    """Generate tile map images from ASCII map + description.

    INPUT_FILE should be a JSON file containing:
    - description: Scene description for the AI
    - tiles: Array of ASCII strings representing the map
    - legend: Mapping of characters to terrain types

    Example:

    \b
    {
        "description": "A mystical forest glade at twilight",
        "tiles": [
            "TTTTTTTTTTTTTTTT",
            "TGGGGGGGGGGGGGGT",
            "TGGGGWWWWGGGGPGT",
            ...
        ],
        "legend": {
            "T": "trees",
            "G": "grass",
            "W": "water",
            "P": "path"
        }
    }
    """
    # Determine output name
    if name is None:
        name = Path(input_file).stem

    click.echo(f"Loading tile map from: {input_file}")

    try:
        tilemap = load_tilemap(input_file)
    except Exception as e:
        click.echo(f"Error loading tile map: {e}", err=True)
        sys.exit(1)

    click.echo(f"Loaded {tilemap.width}x{tilemap.height} tile map")
    click.echo(f"Description: {tilemap.description[:100]}...")

    # Create configuration
    model_config = ModelConfig(
        use_fp16=not fp32,
        device=device,
    )

    gen_config = GenerationConfig(
        tile_size=tile_size,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    prompt_config = PromptConfig()

    # Initialize pipeline
    click.echo(f"Initializing Stable Diffusion pipeline ({model_config.base_model})...")
    pipeline = SDXLPipeline(
        base_model=model_config.base_model,
        inpainting_model=None,  # Use img2img fallback for turbo models
        use_fp16=model_config.use_fp16,
        device=model_config.device,
        is_turbo=model_config.is_turbo,
    )

    # Initialize generator
    tile_generator = TileGenerator(
        pipeline=pipeline,
        generation_config=gen_config,
        prompt_config=prompt_config,
    )

    # Progress callback
    def progress_callback(current: int, total: int):
        percent = (current / total) * 100
        click.echo(f"\rGenerating tiles: {current}/{total} ({percent:.1f}%)", nl=False)

    click.echo(f"Generating {tilemap.width * tilemap.height} tiles...")

    try:
        tiles = tile_generator.generate_tilemap(
            tilemap=tilemap,
            seed=seed,
            progress_callback=progress_callback,
        )
    except Exception as e:
        click.echo(f"\nError during generation: {e}", err=True)
        logger.exception("Generation failed")
        sys.exit(1)

    click.echo()  # New line after progress

    # Save output
    output_writer = OutputWriter(
        output_dir=output_dir,
        tile_size=tile_size,
    )

    click.echo(f"Saving output to: {output_dir}")

    saved = output_writer.save_all(
        tiles=tiles,
        tilemap=tilemap,
        name=name,
        save_individual=not no_individual_tiles,
        save_composite=True,
    )

    if debug_grid:
        grid_path = output_writer.save_debug_grid(tiles, tilemap, name)
        saved["debug_grid"] = grid_path

    # Report results
    click.echo("\nGeneration complete!")
    click.echo(f"  Composite image: {saved.get('composite', 'N/A')}")
    if "tiles_dir" in saved:
        click.echo(f"  Individual tiles: {saved['tiles_dir']}")
    if "debug_grid" in saved:
        click.echo(f"  Debug grid: {saved['debug_grid']}")

    # Cleanup
    pipeline.unload()


def main():
    """Main entry point."""
    generate()


if __name__ == "__main__":
    main()
