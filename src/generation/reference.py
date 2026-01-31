"""Reference image generation for establishing global scene style."""

import logging
from typing import Optional

from PIL import Image

from ..config import GenerationConfig, PromptConfig
from ..types import TileMap, TileType
from .pipeline import SDXLPipeline
from .prompts import PromptBuilder

logger = logging.getLogger(__name__)


class ReferenceGenerator:
    """Generates low-resolution reference images for scene consistency."""

    def __init__(
        self,
        pipeline: SDXLPipeline,
        generation_config: GenerationConfig = None,
        prompt_config: PromptConfig = None,
    ):
        """Initialize the reference generator.

        Args:
            pipeline: SDXL pipeline for image generation
            generation_config: Generation settings
            prompt_config: Prompt template settings
        """
        self.pipeline = pipeline
        self.gen_config = generation_config or GenerationConfig()
        self.prompt_builder = PromptBuilder(prompt_config)

    def generate(
        self,
        tilemap: TileMap,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a low-resolution reference image for the scene.

        This reference establishes the overall style, lighting, and color palette
        that individual tiles will match.

        Args:
            tilemap: The tile map with description and terrain layout
            seed: Random seed for reproducibility

        Returns:
            Low-resolution reference image (e.g., 512x384)
        """
        # Build prompt that describes the overall scene
        prompt = self._build_scene_prompt(tilemap)
        negative_prompt = self.prompt_builder.get_negative_prompt()

        logger.info(f"Generating reference image with prompt: {prompt[:100]}...")

        reference = self.pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.gen_config.reference_width,
            height=self.gen_config.reference_height,
            num_inference_steps=self.gen_config.num_inference_steps,
            guidance_scale=self.gen_config.guidance_scale,
            seed=seed,
        )

        logger.info(
            f"Reference image generated: {reference.width}x{reference.height}"
        )
        return reference

    def _build_scene_prompt(self, tilemap: TileMap) -> str:
        """Build a prompt describing the overall scene composition.

        Args:
            tilemap: The tile map to describe

        Returns:
            Prompt string for reference generation
        """
        # Count terrain types to describe composition
        terrain_counts: dict[TileType, int] = {}
        for row in tilemap.tiles:
            for tile in row:
                terrain_counts[tile] = terrain_counts.get(tile, 0) + 1

        # Sort by count to get dominant terrain
        sorted_terrain = sorted(
            terrain_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Build terrain description
        terrain_parts = []
        for tile_type, count in sorted_terrain[:4]:  # Top 4 terrain types
            percentage = (count / (tilemap.width * tilemap.height)) * 100
            if percentage > 5:  # Only mention significant terrain
                terrain_parts.append(f"{tile_type.value} ({percentage:.0f}%)")

        terrain_desc = ", ".join(terrain_parts) if terrain_parts else "mixed terrain"

        # Combine with scene description
        base_prompt = self.prompt_builder.build_reference_prompt(tilemap.description)
        return f"{base_prompt}, landscape with {terrain_desc}"

    def extract_region(
        self,
        reference: Image.Image,
        x: int,
        y: int,
        tilemap: TileMap,
        tile_size: int,
    ) -> Image.Image:
        """Extract the region of the reference corresponding to a tile position.

        Args:
            reference: The full reference image
            x: Tile x position in grid
            y: Tile y position in grid
            tilemap: The tile map for dimensions
            tile_size: Target tile size

        Returns:
            Cropped and scaled region from reference
        """
        # Calculate scale factor from reference to tile grid
        scale_x = reference.width / tilemap.width
        scale_y = reference.height / tilemap.height

        # Get corresponding region in reference
        ref_x = int(x * scale_x)
        ref_y = int(y * scale_y)
        ref_w = int(scale_x)
        ref_h = int(scale_y)

        # Crop the region (with some padding for context)
        padding = 2
        left = max(0, ref_x - padding)
        top = max(0, ref_y - padding)
        right = min(reference.width, ref_x + ref_w + padding)
        bottom = min(reference.height, ref_y + ref_h + padding)

        region = reference.crop((left, top, right, bottom))

        # Scale to tile size
        return region.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
