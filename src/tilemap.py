"""ASCII tile map parsing and loading."""

import json
from pathlib import Path
from typing import Union

from .types import TileMap, TileType


def load_tilemap(path: Union[str, Path]) -> TileMap:
    """Load a tile map from a JSON file.

    Expected JSON format:
    {
        "description": "Scene description for the AI",
        "tiles": [
            "TTTTTTTTTTTTTTTT",
            "TGGGGGGGGGGGGGGT",
            ...
        ],
        "legend": {
            "T": "trees",
            "G": "grass",
            ...
        }
    }
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    return parse_tilemap(data)


def parse_tilemap(data: dict) -> TileMap:
    """Parse a tile map from dictionary data."""
    description = data.get("description", "A game map scene")
    tile_rows = data.get("tiles", [])
    legend = data.get("legend", {})

    if not tile_rows:
        raise ValueError("No tile rows provided")

    # Determine dimensions from first row
    width = len(tile_rows[0])
    height = len(tile_rows)

    # Parse each row
    tiles: list[list[TileType]] = []
    for row_idx, row_str in enumerate(tile_rows):
        if len(row_str) != width:
            raise ValueError(
                f"Row {row_idx} has inconsistent width: "
                f"expected {width}, got {len(row_str)}"
            )

        row_tiles: list[TileType] = []
        for col_idx, char in enumerate(row_str):
            tile_name = legend.get(char)
            if tile_name is None:
                raise ValueError(
                    f"Unknown tile character '{char}' at position ({col_idx}, {row_idx}). "
                    f"Add it to the legend."
                )
            try:
                tile_type = TileType.from_string(tile_name)
            except ValueError as e:
                raise ValueError(
                    f"Invalid tile type '{tile_name}' for character '{char}': {e}"
                )
            row_tiles.append(tile_type)

        tiles.append(row_tiles)

    return TileMap(tiles=tiles, description=description, width=width, height=height)


def tilemap_to_ascii(tilemap: TileMap, legend: dict[str, str] = None) -> str:
    """Convert a TileMap back to ASCII representation for debugging."""
    # Build reverse legend (tile type -> character)
    if legend is None:
        # Use first letter of each tile type as default
        reverse_legend = {tt: tt.value[0].upper() for tt in TileType}
    else:
        reverse_legend = {TileType.from_string(v): k for k, v in legend.items()}

    lines = []
    for row in tilemap.tiles:
        line = "".join(reverse_legend.get(tile, "?") for tile in row)
        lines.append(line)

    return "\n".join(lines)
