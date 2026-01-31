"""Tile compositor for assembling final images."""

import logging
from typing import Optional

from PIL import Image

from ..types import GeneratedTile, TileMap

logger = logging.getLogger(__name__)


class Compositor:
    """Assembles individual tiles into composite images."""

    def __init__(self, tile_size: int = 64):
        """Initialize the compositor.

        Args:
            tile_size: Size of individual tiles in pixels
        """
        self.tile_size = tile_size

    def compose(
        self,
        tiles: list[GeneratedTile],
        tilemap: TileMap,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """Assemble tiles into a single composite image.

        Args:
            tiles: List of generated tiles to assemble
            tilemap: The tile map for dimensions
            background_color: RGB background color for any gaps

        Returns:
            Composite image containing all tiles
        """
        width = tilemap.width * self.tile_size
        height = tilemap.height * self.tile_size

        logger.info(f"Compositing {len(tiles)} tiles into {width}x{height} image")

        # Create composite canvas
        composite = Image.new("RGB", (width, height), background_color)

        # Place each tile
        for tile in tiles:
            x_px = tile.x * self.tile_size
            y_px = tile.y * self.tile_size

            # Ensure tile is the right size
            tile_img = tile.image
            if tile_img.size != (self.tile_size, self.tile_size):
                tile_img = tile_img.resize(
                    (self.tile_size, self.tile_size), Image.Resampling.LANCZOS
                )

            # Convert to RGB if necessary
            if tile_img.mode != "RGB":
                tile_img = tile_img.convert("RGB")

            composite.paste(tile_img, (x_px, y_px))

        logger.info("Composition complete")
        return composite

    def compose_with_grid(
        self,
        tiles: list[GeneratedTile],
        tilemap: TileMap,
        grid_color: tuple[int, int, int] = (50, 50, 50),
        grid_width: int = 1,
    ) -> Image.Image:
        """Assemble tiles with a visible grid overlay.

        Useful for debugging and visualization.

        Args:
            tiles: List of generated tiles to assemble
            tilemap: The tile map for dimensions
            grid_color: RGB color for grid lines
            grid_width: Width of grid lines in pixels

        Returns:
            Composite image with grid overlay
        """
        from PIL import ImageDraw

        composite = self.compose(tiles, tilemap)
        draw = ImageDraw.Draw(composite)

        width = tilemap.width * self.tile_size
        height = tilemap.height * self.tile_size

        # Draw vertical lines
        for x in range(0, width + 1, self.tile_size):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=grid_width)

        # Draw horizontal lines
        for y in range(0, height + 1, self.tile_size):
            draw.line([(0, y), (width, y)], fill=grid_color, width=grid_width)

        return composite

    def compose_scaled(
        self,
        tiles: list[GeneratedTile],
        tilemap: TileMap,
        scale: float = 1.0,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
    ) -> Image.Image:
        """Assemble and scale the composite image.

        Args:
            tiles: List of generated tiles to assemble
            tilemap: The tile map for dimensions
            scale: Scale factor (2.0 = double size)
            resample: Resampling algorithm for scaling

        Returns:
            Scaled composite image
        """
        composite = self.compose(tiles, tilemap)

        if scale != 1.0:
            new_width = int(composite.width * scale)
            new_height = int(composite.height * scale)
            composite = composite.resize((new_width, new_height), resample)
            logger.info(f"Scaled composite to {new_width}x{new_height}")

        return composite

    def extract_tile(
        self,
        composite: Image.Image,
        x: int,
        y: int,
    ) -> Image.Image:
        """Extract a single tile from a composite image.

        Args:
            composite: The composite image
            x: Tile x coordinate
            y: Tile y coordinate

        Returns:
            Extracted tile image
        """
        x_px = x * self.tile_size
        y_px = y * self.tile_size

        return composite.crop(
            (x_px, y_px, x_px + self.tile_size, y_px + self.tile_size)
        )
