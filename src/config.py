"""Configuration settings for the game map renderer."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ControlNetConfig:
    """Configuration for ControlNet-based generation."""

    # ControlNet Union model for segmentation-guided generation
    controlnet_model: str = "xinsir/controlnet-union-sdxl-1.0"

    # Base SDXL model (not turbo - need proper ControlNet support)
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # How strictly to follow the segmentation map (0.0-1.0)
    # Lower = more creative, higher = stricter adherence to map
    conditioning_scale: float = 0.5

    # Number of inference steps (more = better quality, slower)
    num_inference_steps: int = 20

    # Classifier-free guidance scale
    guidance_scale: float = 7.5

    # Use fp16 for reduced memory usage
    use_fp16: bool = True

    # Device to run on (None = auto-detect)
    device: Optional[str] = None

    # Output image dimensions
    output_width: int = 1024
    output_height: int = 768


@dataclass
class ModelConfig:
    """Configuration for Stable Diffusion models (legacy tile-based approach)."""

    # Base model for generation - SDXL Turbo for fast iteration
    base_model: str = "stabilityai/sdxl-turbo"

    # Inpainting model - using SDXL Turbo (no dedicated inpainting version)
    # Falls back to img2img style generation
    inpainting_model: str = "stabilityai/sdxl-turbo"

    # Optional ControlNet for extra consistency
    controlnet_model: Optional[str] = None  # "xinsir/controlnet-tile-sdxl-1.0"

    # Use fp16 for reduced memory usage
    use_fp16: bool = True

    # Device to run on (None = auto-detect)
    device: Optional[str] = None

    # Whether using a turbo/lightning model (affects step count and guidance)
    is_turbo: bool = True


@dataclass
class GenerationConfig:
    """Configuration for image generation."""

    # Tile dimensions
    tile_size: int = 64

    # Overlap with neighbors for seamless blending (pixels)
    overlap: int = 16

    # Number of inference steps (4 for turbo, 30 for base SDXL)
    num_inference_steps: int = 4

    # Guidance scale (0.0 for turbo models, 7.5 for base SDXL)
    guidance_scale: float = 0.0

    # Reference image size (low-res scene overview)
    reference_width: int = 512
    reference_height: int = 384

    # Random seed for reproducibility (None = random)
    seed: Optional[int] = None

    # Batch size for tile generation (higher = faster but more VRAM)
    batch_size: int = 1


@dataclass
class PromptConfig:
    """Configuration for prompt templates."""

    # Style suffix for photorealistic output
    style_suffix: str = (
        "photorealistic, aerial view, top-down photograph, "
        "natural lighting, high detail, seamless texture, 8k uhd, raw photo"
    )

    # Negative prompt to avoid unwanted styles
    negative_prompt: str = (
        "cartoon, illustration, painting, drawing, anime, low quality, blurry, "
        "pixelated, artificial, fake, unrealistic, oversaturated, "
        "text, watermark, signature"
    )

    # Tile-specific prompt additions
    tile_prompts: dict[str, str] = field(
        default_factory=lambda: {
            "grass": "lush green grass, individual blades visible, natural variation",
            "water": "clear water surface, subtle ripples, reflections, depth, blue tones",
            "trees": "dense forest canopy from above, varied greens, shadows between trees",
            "wall": "stone wall texture, weathered, realistic masonry, gray tones",
            "sand": "beach sand, fine grain, natural patterns, warm golden tones",
            "path": "dirt path, worn texture, subtle footprints, earthy brown tones",
            "stone": "rough stone ground, rocky texture, gray and brown tones",
            "lava": "molten lava, glowing orange, dark crusted surface, heat distortion",
            "snow": "fresh white snow, pristine surface, subtle blue shadows",
            "dirt": "bare dirt ground, brown earth, natural soil texture",
            "bridge": "wooden planks, bridge surface from above, weathered wood grain",
            "flowers": "wildflower meadow from above, colorful blooms in green grass",
        }
    )


@dataclass
class Config:
    """Main configuration combining all settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)

    # Output settings
    output_dir: str = "./output"
    save_individual_tiles: bool = True
    save_composite: bool = True


# Default configuration instance
DEFAULT_CONFIG = Config()
