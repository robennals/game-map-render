"""SDXL pipeline wrapper for tile generation."""

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SDXLPipeline:
    """Wrapper for SDXL/SDXL-Turbo pipelines."""

    def __init__(
        self,
        base_model: str = "stabilityai/sdxl-turbo",
        inpainting_model: Optional[str] = None,
        use_fp16: bool = True,
        device: Optional[str] = None,
        is_turbo: bool = True,
    ):
        """Initialize the SDXL pipeline.

        Args:
            base_model: HuggingFace model ID for generation
            inpainting_model: HuggingFace model ID for inpainting (optional, uses img2img if None)
            use_fp16: Use half precision for reduced memory
            device: Device to run on (None = auto-detect)
            is_turbo: Whether using a turbo/lightning model (affects pipeline choice)
        """
        self.base_model_id = base_model
        self.inpainting_model_id = inpainting_model
        self.use_fp16 = use_fp16
        self.is_turbo = is_turbo

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

        # Lazy-loaded pipelines
        self._base_pipeline = None
        self._img2img_pipeline = None
        self._inpainting_pipeline = None

        logger.info(f"SDXLPipeline initialized: device={self.device}, dtype={self.dtype}, turbo={is_turbo}")

    @property
    def base_pipeline(self):
        """Lazy-load the base text-to-image pipeline."""
        if self._base_pipeline is None:
            logger.info(f"Loading base model: {self.base_model_id}")
            from diffusers import AutoPipelineForText2Image

            self._base_pipeline = AutoPipelineForText2Image.from_pretrained(
                self.base_model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.use_fp16 else None,
            )
            self._base_pipeline.to(self.device)

            logger.info("Base model loaded successfully")

        return self._base_pipeline

    @property
    def img2img_pipeline(self):
        """Lazy-load the image-to-image pipeline (from base model)."""
        if self._img2img_pipeline is None:
            logger.info("Loading img2img pipeline from base model")
            from diffusers import AutoPipelineForImage2Image

            # Load from the already-loaded base pipeline if available
            if self._base_pipeline is not None:
                self._img2img_pipeline = AutoPipelineForImage2Image.from_pipe(
                    self._base_pipeline
                )
            else:
                self._img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
                    self.base_model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    variant="fp16" if self.use_fp16 else None,
                )
                self._img2img_pipeline.to(self.device)

            logger.info("Img2img pipeline loaded successfully")

        return self._img2img_pipeline

    @property
    def inpainting_pipeline(self):
        """Lazy-load the inpainting pipeline (if available)."""
        if self._inpainting_pipeline is None:
            if self.inpainting_model_id is None:
                # No dedicated inpainting model, return None to fall back to img2img
                return None

            logger.info(f"Loading inpainting model: {self.inpainting_model_id}")
            from diffusers import AutoPipelineForInpainting

            self._inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
                self.inpainting_model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.use_fp16 else None,
            )
            self._inpainting_pipeline.to(self.device)

            logger.info("Inpainting model loaded successfully")

        return self._inpainting_pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image using the base pipeline.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt to avoid (ignored for turbo models)
            width: Output image width
            height: Output image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale (0.0 for turbo)
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # Only add negative prompt for non-turbo models
        if not self.is_turbo and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        result = self.base_pipeline(**kwargs)
        return result.images[0]

    def img2img(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        strength: float = 0.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image using img2img with a source image.

        Args:
            prompt: Text prompt for generation
            image: Source image to transform
            negative_prompt: Negative prompt to avoid (ignored for turbo models)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale (0.0 for turbo)
            strength: How much to transform (0.0 = no change, 1.0 = complete)
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Ensure image dimensions are multiples of 8
        width = (image.width // 8) * 8
        height = (image.height // 8) * 8
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        kwargs = {
            "prompt": prompt,
            "image": image,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": generator,
        }

        # Only add negative prompt for non-turbo models
        if not self.is_turbo and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        result = self.img2img_pipeline(**kwargs)
        return result.images[0]

    def inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        strength: float = 0.99,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image using inpainting or img2img fallback.

        Args:
            prompt: Text prompt for generation
            image: Context image with surrounding content
            mask: Binary mask (white = area to generate) - used for blending if no inpainting model
            negative_prompt: Negative prompt to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            strength: How much to transform the masked area
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        # Check if we have a dedicated inpainting model
        if self.inpainting_pipeline is not None:
            return self._inpaint_dedicated(
                prompt, image, mask, negative_prompt,
                num_inference_steps, guidance_scale, strength, seed
            )
        else:
            # Fall back to img2img + manual blending
            return self._inpaint_via_img2img(
                prompt, image, mask, negative_prompt,
                num_inference_steps, guidance_scale, strength, seed
            )

    def _inpaint_dedicated(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
    ) -> Image.Image:
        """Inpaint using dedicated inpainting model."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Ensure images are the right size for SDXL (multiples of 8)
        width = (image.width // 8) * 8
        height = (image.height // 8) * 8

        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)

        kwargs = {
            "prompt": prompt,
            "image": image,
            "mask_image": mask,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": generator,
        }

        if not self.is_turbo and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        result = self.inpainting_pipeline(**kwargs)
        return result.images[0]

    def _inpaint_via_img2img(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
    ) -> Image.Image:
        """Simulate inpainting using img2img + mask blending."""
        # Generate new image via img2img
        generated = self.img2img(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )

        # Ensure same size
        if generated.size != image.size:
            generated = generated.resize(image.size, Image.Resampling.LANCZOS)

        # Blend using the mask (white = use generated, black = use original)
        # Convert mask to same size if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)

        # Ensure mask is in L mode for compositing
        if mask.mode != "L":
            mask = mask.convert("L")

        # Composite: where mask is white, use generated; where black, use original
        result = Image.composite(generated, image.convert("RGB"), mask)
        return result

    def unload(self):
        """Unload models to free memory."""
        if self._base_pipeline is not None:
            del self._base_pipeline
            self._base_pipeline = None
        if self._img2img_pipeline is not None:
            del self._img2img_pipeline
            self._img2img_pipeline = None
        if self._inpainting_pipeline is not None:
            del self._inpainting_pipeline
            self._inpainting_pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("Models unloaded")
