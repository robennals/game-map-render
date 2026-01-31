"""Individual tile generation with context-aware inpainting."""

import logging
from typing import Optional

import numpy as np
from PIL import Image

from ..config import GenerationConfig, PromptConfig
from ..types import GeneratedTile, NeighborContext, TileMap, TileType
from .pipeline import SDXLPipeline
from .prompts import PromptBuilder
from .reference import ReferenceGenerator

logger = logging.getLogger(__name__)


class TileGenerator:
    """Generates individual tiles with context from neighbors for seamless blending."""

    def __init__(
        self,
        pipeline: SDXLPipeline,
        generation_config: GenerationConfig = None,
        prompt_config: PromptConfig = None,
    ):
        """Initialize the tile generator.

        Args:
            pipeline: SDXL pipeline for image generation
            generation_config: Generation settings
            prompt_config: Prompt template settings
        """
        self.pipeline = pipeline
        self.gen_config = generation_config or GenerationConfig()
        self.prompt_builder = PromptBuilder(prompt_config)
        self.reference_generator = ReferenceGenerator(
            pipeline, generation_config, prompt_config
        )

    def generate_tilemap(
        self,
        tilemap: TileMap,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[GeneratedTile]:
        """Generate all tiles for a tile map.

        Args:
            tilemap: The tile map to render
            seed: Base random seed (incremented per tile)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of all generated tiles
        """
        tile_size = self.gen_config.tile_size
        total_tiles = tilemap.width * tilemap.height

        logger.info(
            f"Starting tile generation: {tilemap.width}x{tilemap.height} "
            f"= {total_tiles} tiles at {tile_size}x{tile_size}px"
        )

        # Generate reference image for style consistency
        reference = self.reference_generator.generate(tilemap, seed)

        # Initialize grid to track generated tiles
        tile_grid: list[list[Optional[GeneratedTile]]] = [
            [None for _ in range(tilemap.width)] for _ in range(tilemap.height)
        ]

        generated_tiles: list[GeneratedTile] = []
        current = 0

        # Generate tiles in scanline order (top-left to bottom-right)
        for y in range(tilemap.height):
            for x in range(tilemap.width):
                tile_type = tilemap.tiles[y][x]

                # Calculate seed for this tile (if base seed provided)
                tile_seed = None
                if seed is not None:
                    tile_seed = seed + y * tilemap.width + x

                # Get context from already-generated neighbors
                context = self._get_neighbor_context(tile_grid, x, y)

                # Get neighbor types for transition prompts
                neighbor_types = self._get_neighbor_types(tilemap, x, y)

                # Extract reference region for this tile
                ref_region = self.reference_generator.extract_region(
                    reference, x, y, tilemap, tile_size
                )

                # Generate the tile
                tile_image = self._generate_tile(
                    tile_type=tile_type,
                    scene_description=tilemap.description,
                    neighbor_types=neighbor_types,
                    context=context,
                    reference_region=ref_region,
                    seed=tile_seed,
                )

                # Store the generated tile
                generated_tile = GeneratedTile(
                    image=tile_image, position=(x, y), tile_type=tile_type
                )
                tile_grid[y][x] = generated_tile
                generated_tiles.append(generated_tile)

                current += 1
                if progress_callback:
                    progress_callback(current, total_tiles)

                if current % 10 == 0:
                    logger.info(f"Progress: {current}/{total_tiles} tiles generated")

        logger.info(f"Tile generation complete: {len(generated_tiles)} tiles")
        return generated_tiles

    def _generate_tile(
        self,
        tile_type: TileType,
        scene_description: str,
        neighbor_types: list[TileType],
        context: NeighborContext,
        reference_region: Image.Image,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a single tile with context awareness.

        Args:
            tile_type: Type of terrain to generate
            scene_description: Overall scene description
            neighbor_types: Types of neighboring tiles
            context: Images from already-generated neighbors
            reference_region: Corresponding region from reference image
            seed: Random seed for reproducibility

        Returns:
            Generated tile image
        """
        tile_size = self.gen_config.tile_size
        overlap = self.gen_config.overlap

        # Build prompt with transition awareness
        prompt = self.prompt_builder.build_transition_prompt(
            tile_type, neighbor_types, scene_description
        )
        negative_prompt = self.prompt_builder.get_negative_prompt()

        # If we have neighbor context, use inpainting for seamless blending
        if context.has_context():
            return self._generate_with_inpainting(
                prompt=prompt,
                negative_prompt=negative_prompt,
                context=context,
                reference_region=reference_region,
                tile_size=tile_size,
                overlap=overlap,
                seed=seed,
            )
        else:
            # First tile (top-left) - generate from scratch using reference
            return self._generate_from_reference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                reference_region=reference_region,
                tile_size=tile_size,
                seed=seed,
            )

    def _generate_from_reference(
        self,
        prompt: str,
        negative_prompt: str,
        reference_region: Image.Image,
        tile_size: int,
        seed: Optional[int],
    ) -> Image.Image:
        """Generate tile from scratch, guided by reference region.

        For the first tile with no neighbors, we use img2img-style generation
        starting from the reference region.
        """
        # Create a context image from the reference (blurred for variation)
        from PIL import ImageFilter

        context_img = reference_region.copy()
        context_img = context_img.filter(ImageFilter.GaussianBlur(radius=3))

        # Create mask for full generation (all white = regenerate everything)
        mask = Image.new("L", (tile_size, tile_size), 255)

        return self.pipeline.inpaint(
            prompt=prompt,
            image=context_img,
            mask=mask,
            negative_prompt=negative_prompt,
            num_inference_steps=self.gen_config.num_inference_steps,
            guidance_scale=self.gen_config.guidance_scale,
            strength=0.95,  # High strength for more generation freedom
            seed=seed,
        )

    def _generate_with_inpainting(
        self,
        prompt: str,
        negative_prompt: str,
        context: NeighborContext,
        reference_region: Image.Image,
        tile_size: int,
        overlap: int,
        seed: Optional[int],
    ) -> Image.Image:
        """Generate tile using inpainting with neighbor context.

        Creates a context image with overlap regions from neighbors,
        then inpaints the center region.
        """
        # Start with the reference region as base
        context_img = reference_region.resize(
            (tile_size, tile_size), Image.Resampling.LANCZOS
        )

        # Paste overlap regions from neighbors
        if context.left is not None:
            # Take right edge of left neighbor
            left_edge = context.left.crop(
                (tile_size - overlap, 0, tile_size, tile_size)
            )
            context_img.paste(left_edge, (0, 0))

        if context.top is not None:
            # Take bottom edge of top neighbor
            top_edge = context.top.crop((0, tile_size - overlap, tile_size, tile_size))
            context_img.paste(top_edge, (0, 0))

        if context.top_left is not None:
            # Take bottom-right corner of top-left neighbor
            corner = context.top_left.crop(
                (tile_size - overlap, tile_size - overlap, tile_size, tile_size)
            )
            context_img.paste(corner, (0, 0))

        # Create inpainting mask (white = area to generate)
        mask = self._create_inpainting_mask(tile_size, overlap, context)

        return self.pipeline.inpaint(
            prompt=prompt,
            image=context_img,
            mask=mask,
            negative_prompt=negative_prompt,
            num_inference_steps=self.gen_config.num_inference_steps,
            guidance_scale=self.gen_config.guidance_scale,
            strength=0.99,
            seed=seed,
        )

    def _create_inpainting_mask(
        self,
        tile_size: int,
        overlap: int,
        context: NeighborContext,
    ) -> Image.Image:
        """Create a mask for inpainting that preserves overlap regions.

        Args:
            tile_size: Size of the tile
            overlap: Overlap region size
            context: Available neighbor context

        Returns:
            Grayscale mask (255 = generate, 0 = preserve)
        """
        # Start with all white (generate everything)
        mask = np.ones((tile_size, tile_size), dtype=np.uint8) * 255

        # Create gradient for smooth blending
        gradient_size = overlap

        # If we have left neighbor, preserve and blend left edge
        if context.left is not None:
            for i in range(gradient_size):
                # Gradient from 0 (preserve) to 255 (generate)
                value = int((i / gradient_size) * 255)
                mask[:, i] = np.minimum(mask[:, i], value)

        # If we have top neighbor, preserve and blend top edge
        if context.top is not None:
            for i in range(gradient_size):
                value = int((i / gradient_size) * 255)
                mask[i, :] = np.minimum(mask[i, :], value)

        return Image.fromarray(mask, mode="L")

    def _get_neighbor_context(
        self,
        tile_grid: list[list[Optional[GeneratedTile]]],
        x: int,
        y: int,
    ) -> NeighborContext:
        """Get context images from already-generated neighbors.

        Args:
            tile_grid: Grid of generated tiles
            x: Current tile x position
            y: Current tile y position

        Returns:
            NeighborContext with available neighbor images
        """
        context = NeighborContext()

        # Left neighbor
        if x > 0 and tile_grid[y][x - 1] is not None:
            context.left = tile_grid[y][x - 1].image

        # Top neighbor
        if y > 0 and tile_grid[y - 1][x] is not None:
            context.top = tile_grid[y - 1][x].image

        # Top-left diagonal
        if x > 0 and y > 0 and tile_grid[y - 1][x - 1] is not None:
            context.top_left = tile_grid[y - 1][x - 1].image

        # Top-right diagonal
        if x < len(tile_grid[0]) - 1 and y > 0 and tile_grid[y - 1][x + 1] is not None:
            context.top_right = tile_grid[y - 1][x + 1].image

        return context

    def _get_neighbor_types(
        self,
        tilemap: TileMap,
        x: int,
        y: int,
    ) -> list[TileType]:
        """Get tile types of all neighbors for transition prompts.

        Args:
            tilemap: The tile map
            x: Current tile x position
            y: Current tile y position

        Returns:
            List of neighboring tile types
        """
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            neighbor = tilemap.get_tile(x + dx, y + dy)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors
