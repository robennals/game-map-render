"""Type definitions for the game map renderer."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PIL import Image


class TileType(Enum):
    """Supported terrain tile types."""

    GRASS = "grass"
    WATER = "water"
    WALL = "wall"
    TREES = "trees"
    SAND = "sand"
    PATH = "path"
    STONE = "stone"
    LAVA = "lava"
    SNOW = "snow"
    DIRT = "dirt"
    BRIDGE = "bridge"
    FLOWERS = "flowers"

    @classmethod
    def from_string(cls, value: str) -> "TileType":
        """Convert a string to TileType, case-insensitive."""
        value_lower = value.lower()
        for tile_type in cls:
            if tile_type.value == value_lower:
                return tile_type
        raise ValueError(f"Unknown tile type: {value}")


@dataclass
class TileMap:
    """Represents a parsed tile map with terrain grid and description."""

    tiles: list[list[TileType]]  # [row][col] - typically 12x16
    description: str
    width: int = 16
    height: int = 12

    def __post_init__(self):
        """Validate map dimensions."""
        if len(self.tiles) != self.height:
            raise ValueError(f"Expected {self.height} rows, got {len(self.tiles)}")
        for i, row in enumerate(self.tiles):
            if len(row) != self.width:
                raise ValueError(
                    f"Row {i}: expected {self.width} columns, got {len(row)}"
                )

    def get_tile(self, x: int, y: int) -> Optional[TileType]:
        """Get tile at position, or None if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None


@dataclass
class GeneratedTile:
    """A generated tile image with metadata."""

    image: Image.Image
    position: tuple[int, int]  # (x, y) in grid coordinates
    tile_type: TileType

    @property
    def x(self) -> int:
        return self.position[0]

    @property
    def y(self) -> int:
        return self.position[1]


@dataclass
class NeighborContext:
    """Context from neighboring tiles for seamless generation."""

    left: Optional[Image.Image] = None
    top: Optional[Image.Image] = None
    top_left: Optional[Image.Image] = None
    top_right: Optional[Image.Image] = None

    def has_context(self) -> bool:
        """Check if any neighbor context is available."""
        return any([self.left, self.top, self.top_left, self.top_right])
