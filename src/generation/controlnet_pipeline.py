"""ControlNet pipeline for generating full scene images from segmentation maps."""

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Control mode values for ControlNet Union
# 0 = openpose, 1 = depth, 2 = thick line (scribble/hed/softedge)
# 3 = thin line (canny/mlsd/lineart), 4 = normal, 5 = segment
CONTROL_MODE_SEGMENT = 5


class ControlNetGenerator:
    """Generator that uses ControlNet Union segmentation to create full scene images."""

    def __init__(
        self,
        controlnet_model: str = "xinsir/controlnet-union-sdxl-1.0",
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the ControlNet pipeline.

        Args:
            controlnet_model: HuggingFace model ID for the ControlNet Union model
            base_model: HuggingFace model ID for the base SDXL model
            use_fp16: Use half precision for reduced memory
            device: Device to run on (None = auto-detect)
        """
        self.controlnet_model_id = controlnet_model
        self.base_model_id = base_model
        self.use_fp16 = use_fp16

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                logger.warning("No GPU detected, using CPU (generation will be slow)")
        else:
            self.device = device

        self.dtype = torch.float16 if use_fp16 and self.device != "cpu" else torch.float32

        # Lazy-loaded pipeline
        self._pipeline = None
        self._controlnet = None

        logger.info(
            f"ControlNetGenerator initialized: device={self.device}, dtype={self.dtype}"
        )

    def _load_pipeline(self):
        """Load the ControlNet Union pipeline (lazy initialization)."""
        if self._pipeline is not None:
            return

        logger.info(f"Loading ControlNet Union: {self.controlnet_model_id}")
        from diffusers import (
            AutoencoderKL,
            ControlNetUnionModel,
            StableDiffusionXLControlNetUnionPipeline,
        )

        self._controlnet = ControlNetUnionModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=self.dtype,
        )

        # Use the fp16-fixed VAE for better quality
        logger.info("Loading VAE: madebyollin/sdxl-vae-fp16-fix")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.dtype,
        )

        logger.info(f"Loading base model: {self.base_model_id}")
        self._pipeline = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            self.base_model_id,
            controlnet=self._controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            variant="fp16" if self.use_fp16 else None,
        )

        # Apply memory optimizations
        if self.device == "mps":
            # MPS-specific optimizations
            self._pipeline.enable_attention_slicing()
            self._pipeline.to(self.device)
        elif self.device == "cuda":
            self._pipeline.enable_model_cpu_offload()
            self._pipeline.enable_attention_slicing()
        else:
            self._pipeline.to(self.device)

        logger.info("ControlNet Union pipeline loaded successfully")

    def generate(
        self,
        prompt: str,
        segmentation_image: Image.Image,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 768,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image using ControlNet Union with a segmentation map.

        Args:
            prompt: Text prompt for generation
            segmentation_image: Color-coded segmentation image
            negative_prompt: Negative prompt to avoid
            width: Output image width (must be multiple of 8)
            height: Output image height (must be multiple of 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: How strictly to follow the segmentation map
                                          (0.0-1.0, higher = stricter)
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        self._load_pipeline()

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Resize segmentation image to match output dimensions
        if segmentation_image.size != (width, height):
            segmentation_image = segmentation_image.resize(
                (width, height),
                Image.Resampling.NEAREST  # Keep sharp edges for segmentation
            )

        # Ensure RGB mode
        if segmentation_image.mode != "RGB":
            segmentation_image = segmentation_image.convert("RGB")

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            f"Generating image: {width}x{height}, steps={num_inference_steps}, "
            f"guidance={guidance_scale}, conditioning={controlnet_conditioning_scale}"
        )

        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            control_image=[segmentation_image],
            control_mode=[CONTROL_MODE_SEGMENT],
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[controlnet_conditioning_scale],
            generator=generator,
        )

        return result.images[0]

    def unload(self):
        """Unload models to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._controlnet is not None:
            del self._controlnet
            self._controlnet = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("ControlNet models unloaded")
