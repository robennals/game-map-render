"""Prompt templates and building for tile generation."""

from ..config import PromptConfig
from ..types import TileType


class PromptBuilder:
    """Builds prompts for tile generation based on type and scene description."""

    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()

    def build_tile_prompt(
        self,
        tile_type: TileType,
        scene_description: str,
        include_style: bool = True,
    ) -> str:
        """Build a complete prompt for generating a specific tile type.

        Args:
            tile_type: The type of terrain tile to generate
            scene_description: Overall scene description for context
            include_style: Whether to append photorealistic style suffix

        Returns:
            Complete prompt string for the diffusion model
        """
        # Get tile-specific description
        tile_desc = self.config.tile_prompts.get(
            tile_type.value, f"{tile_type.value} terrain"
        )

        # Build the prompt
        parts = [scene_description, tile_desc, "terrain tile"]

        if include_style:
            parts.append(self.config.style_suffix)

        return ", ".join(parts)

    def build_reference_prompt(self, scene_description: str) -> str:
        """Build a prompt for generating the reference overview image.

        Args:
            scene_description: Overall scene description

        Returns:
            Prompt for generating low-res scene overview
        """
        return (
            f"{scene_description}, aerial view, top-down map view, "
            f"game environment, {self.config.style_suffix}"
        )

    def get_negative_prompt(self) -> str:
        """Get the negative prompt for generation."""
        return self.config.negative_prompt

    def build_transition_prompt(
        self,
        tile_type: TileType,
        neighbor_types: list[TileType],
        scene_description: str,
    ) -> str:
        """Build a prompt that accounts for neighboring tile types.

        This helps create natural transitions between different terrain types.

        Args:
            tile_type: The main tile type being generated
            neighbor_types: Types of neighboring tiles
            scene_description: Overall scene description

        Returns:
            Prompt that encourages natural transitions
        """
        base_prompt = self.build_tile_prompt(tile_type, scene_description)

        # Add transition hints for neighboring types
        unique_neighbors = set(nt for nt in neighbor_types if nt != tile_type)
        if unique_neighbors:
            transition_hints = []
            for neighbor in unique_neighbors:
                transition_hints.append(f"blending with {neighbor.value}")
            transition_str = ", ".join(transition_hints)
            return f"{base_prompt}, natural transition, {transition_str}"

        return base_prompt
