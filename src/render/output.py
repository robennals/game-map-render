"""Output file writing for generated tile maps."""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from ..types import GeneratedTile, TileMap
from .compositor import Compositor

logger = logging.getLogger(__name__)


class OutputWriter:
    """Handles saving generated tiles and composite images."""

    def __init__(
        self,
        output_dir: str = "./output",
        tile_size: int = 64,
    ):
        """Initialize the output writer.

        Args:
            output_dir: Directory to save output files
            tile_size: Size of individual tiles
        """
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.compositor = Compositor(tile_size)

    def save_all(
        self,
        tiles: list[GeneratedTile],
        tilemap: TileMap,
        name: str = "map",
        save_individual: bool = True,
        save_composite: bool = True,
        save_reference: Optional[Image.Image] = None,
    ) -> dict[str, Path]:
        """Save all output files.

        Args:
            tiles: List of generated tiles
            tilemap: The tile map
            name: Base name for output files
            save_individual: Whether to save individual tile images
            save_composite: Whether to save the composite image
            save_reference: Optional reference image to save

        Returns:
            Dictionary mapping output type to file path
        """
        self._ensure_output_dir()
        saved_files: dict[str, Path] = {}

        # Save individual tiles
        if save_individual:
            tiles_dir = self.output_dir / f"{name}_tiles"
            tiles_dir.mkdir(exist_ok=True)

            for tile in tiles:
                tile_path = tiles_dir / f"tile_{tile.x:02d}_{tile.y:02d}.png"
                self._save_image(tile.image, tile_path)

            saved_files["tiles_dir"] = tiles_dir
            logger.info(f"Saved {len(tiles)} individual tiles to {tiles_dir}")

        # Save composite image
        if save_composite:
            composite = self.compositor.compose(tiles, tilemap)
            composite_path = self.output_dir / f"{name}_composite.png"
            self._save_image(composite, composite_path)
            saved_files["composite"] = composite_path
            logger.info(f"Saved composite image to {composite_path}")

        # Save reference image if provided
        if save_reference is not None:
            reference_path = self.output_dir / f"{name}_reference.png"
            self._save_image(save_reference, reference_path)
            saved_files["reference"] = reference_path
            logger.info(f"Saved reference image to {reference_path}")

        return saved_files

    def save_individual_tiles(
        self,
        tiles: list[GeneratedTile],
        name: str = "map",
    ) -> Path:
        """Save individual tile images to a subdirectory.

        Args:
            tiles: List of generated tiles
            name: Base name for the tiles directory

        Returns:
            Path to the tiles directory
        """
        self._ensure_output_dir()
        tiles_dir = self.output_dir / f"{name}_tiles"
        tiles_dir.mkdir(exist_ok=True)

        for tile in tiles:
            filename = f"tile_{tile.x:02d}_{tile.y:02d}.png"
            tile_path = tiles_dir / filename
            self._save_image(tile.image, tile_path)

        logger.info(f"Saved {len(tiles)} tiles to {tiles_dir}")
        return tiles_dir

    def save_composite(
        self,
        tiles: list[GeneratedTile],
        tilemap: TileMap,
        name: str = "map",
    ) -> Path:
        """Save the composite map image.

        Args:
            tiles: List of generated tiles
            tilemap: The tile map for dimensions
            name: Base name for the output file

        Returns:
            Path to the saved composite image
        """
        self._ensure_output_dir()

        composite = self.compositor.compose(tiles, tilemap)
        composite_path = self.output_dir / f"{name}_composite.png"
        self._save_image(composite, composite_path)

        logger.info(f"Saved composite to {composite_path}")
        return composite_path

    def save_debug_grid(
        self,
        tiles: list[GeneratedTile],
        tilemap: TileMap,
        name: str = "map",
    ) -> Path:
        """Save a composite with grid overlay for debugging.

        Args:
            tiles: List of generated tiles
            tilemap: The tile map for dimensions
            name: Base name for the output file

        Returns:
            Path to the saved debug image
        """
        self._ensure_output_dir()

        grid_composite = self.compositor.compose_with_grid(tiles, tilemap)
        grid_path = self.output_dir / f"{name}_debug_grid.png"
        self._save_image(grid_composite, grid_path)

        logger.info(f"Saved debug grid to {grid_path}")
        return grid_path

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_image(self, image: Image.Image, path: Path):
        """Save an image to disk.

        Args:
            image: PIL Image to save
            path: Destination path
        """
        # Convert to RGB if necessary (e.g., for RGBA images)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        image.save(path, "PNG", optimize=True)
