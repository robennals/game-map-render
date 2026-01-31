"""Convert tilemaps to color-coded segmentation images for ControlNet."""

from typing import Dict, Optional, Tuple

from PIL import Image

from ..types import TileMap, TileType


# Terrain color mapping for segmentation
# These colors are chosen to be visually distinct and semantically meaningful
# for the segmentation ControlNet model
TERRAIN_COLORS: Dict[TileType, Tuple[int, int, int]] = {
    TileType.GRASS: (34, 139, 34),       # Forest green
    TileType.WATER: (0, 100, 200),       # Blue
    TileType.TREES: (0, 80, 0),          # Dark green
    TileType.PATH: (139, 90, 43),        # Brown
    TileType.SAND: (238, 214, 175),      # Sandy beige
    TileType.WALL: (128, 128, 128),      # Gray
    TileType.LAVA: (255, 69, 0),         # Red-orange
    TileType.SNOW: (255, 255, 255),      # White
    TileType.STONE: (105, 105, 105),     # Dim gray
    TileType.DIRT: (139, 69, 19),        # Saddle brown
    TileType.BRIDGE: (160, 82, 45),      # Sienna
    TileType.FLOWERS: (34, 139, 34),     # Same as grass base (flowers on grass)
}


def tilemap_to_segmentation(
    tilemap: TileMap,
    output_size: Optional[Tuple[int, int]] = None,
    tile_size: int = 64,
) -> Image.Image:
    """Convert a tilemap to a color-coded segmentation image.

    Each tile in the tilemap becomes a solid color block in the output image.
    The colors are chosen to help the ControlNet understand terrain types.

    Args:
        tilemap: The tilemap to convert
        output_size: Optional (width, height) for the output image.
                    If None, uses tile_size * tilemap dimensions.
        tile_size: Size of each tile in pixels (used if output_size is None)

    Returns:
        A PIL Image with color-coded segmentation
    """
    # Calculate dimensions
    if output_size is None:
        width = tilemap.width * tile_size
        height = tilemap.height * tile_size
    else:
        width, height = output_size

    # Calculate tile dimensions in the output image
    tile_width = width / tilemap.width
    tile_height = height / tilemap.height

    # Create the segmentation image
    seg_image = Image.new("RGB", (width, height))

    # Fill each tile region with the appropriate color
    for row_idx, row in enumerate(tilemap.tiles):
        for col_idx, tile_type in enumerate(row):
            color = TERRAIN_COLORS.get(tile_type, (128, 128, 128))  # Gray fallback

            # Calculate pixel bounds for this tile
            x1 = int(col_idx * tile_width)
            y1 = int(row_idx * tile_height)
            x2 = int((col_idx + 1) * tile_width)
            y2 = int((row_idx + 1) * tile_height)

            # Fill the tile region
            for y in range(y1, y2):
                for x in range(x1, x2):
                    seg_image.putpixel((x, y), color)

    return seg_image


def get_terrain_composition(tilemap: TileMap) -> Dict[TileType, float]:
    """Calculate the percentage of each terrain type in a tilemap.

    Args:
        tilemap: The tilemap to analyze

    Returns:
        Dictionary mapping TileType to percentage (0.0 to 1.0)
    """
    total_tiles = tilemap.width * tilemap.height
    counts: Dict[TileType, int] = {}

    for row in tilemap.tiles:
        for tile_type in row:
            counts[tile_type] = counts.get(tile_type, 0) + 1

    return {
        tile_type: count / total_tiles
        for tile_type, count in counts.items()
    }


def build_terrain_prompt(tilemap: TileMap, base_description: str) -> str:
    """Build a prompt that includes terrain composition information.

    Args:
        tilemap: The tilemap to describe
        base_description: The base scene description from the tilemap

    Returns:
        Enhanced prompt with terrain details
    """
    composition = get_terrain_composition(tilemap)

    # Sort by percentage, descending
    sorted_terrains = sorted(
        composition.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build terrain description for significant terrains (>5%)
    terrain_parts = []
    for tile_type, pct in sorted_terrains:
        if pct >= 0.05:  # Only include if 5% or more
            terrain_parts.append(tile_type.value)

    terrain_str = ", ".join(terrain_parts)

    # Combine with base description
    prompt = f"{base_description}, featuring {terrain_str}"

    return prompt
