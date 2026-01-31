#!/usr/bin/env python3
"""Step 3: Regional prompts via latent blending.

Simpler approach: generate with each prompt separately, then blend the latents
based on spatial masks. Less elegant but avoids MPS attention issues.
"""

import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np


def create_left_right_masks(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create left-half and right-half masks with soft edges."""
    x = torch.linspace(0, 1, width)
    left_mask = torch.sigmoid((0.5 - x) * 10)
    left_mask = left_mask.unsqueeze(0).expand(height, -1)
    right_mask = 1.0 - left_mask
    return left_mask, right_mask


@torch.no_grad()
def generate_with_regional_blending(
    pipe,
    prompts: list[str],
    masks: list[torch.Tensor],
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 768,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = 42,
    blend_start_step: int = 0,  # Start blending from this step
):
    """Generate an image by blending denoising from multiple prompts.

    At each denoising step, we:
    1. Compute noise prediction for each prompt
    2. Blend the predictions based on spatial masks
    3. Step the diffusion with the blended prediction
    """
    device = pipe.device

    # Encode all prompts
    prompt_embeds_list = []
    pooled_embeds_list = []
    for prompt in prompts:
        prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
        )
        # Concatenate for CFG: [neg, pos]
        combined = torch.cat([neg_embeds, prompt_embeds], dim=0)
        combined_pooled = torch.cat([neg_pooled, pooled], dim=0)
        prompt_embeds_list.append(combined)
        pooled_embeds_list.append(combined_pooled)

    # Prepare masks for latent space (latent is 1/8 of image size)
    latent_h = height // 8
    latent_w = width // 8
    mask_tensors = []
    for mask in masks:
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze()
        mask_tensors.append(resized.to(device))

    # Normalize masks to sum to 1
    mask_sum = sum(mask_tensors)
    mask_tensors = [m / (mask_sum + 1e-8) for m in mask_tensors]

    # Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Create initial latents
    generator = torch.Generator(device=device).manual_seed(seed)
    model_dtype = next(pipe.unet.parameters()).dtype
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=model_dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    # Add time embeddings for SDXL
    # Get text_encoder_projection_dim from pooled embeddings
    text_encoder_projection_dim = pooled_embeds_list[0].shape[-1]
    add_time_ids = pipe._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=model_dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)
    # Duplicate for CFG
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Get noise predictions for each prompt
        noise_preds = []
        for prompt_embeds, pooled_embeds in zip(prompt_embeds_list, pooled_embeds_list):
            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": add_time_ids,
            }
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            noise_preds.append(noise_pred)

        # Blend noise predictions based on masks
        if i >= blend_start_step:
            blended_pred = torch.zeros_like(noise_preds[0])
            for noise_pred, mask in zip(noise_preds, mask_tensors):
                # Expand mask to match noise_pred shape [2, 4, h, w]
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(noise_pred)
                blended_pred = blended_pred + noise_pred * mask_expanded
        else:
            # Before blending starts, use equal weights
            blended_pred = sum(noise_preds) / len(noise_preds)

        # Perform CFG
        noise_pred_uncond, noise_pred_cond = blended_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}/{num_inference_steps}")

    # Decode latents to image
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image


def main():
    print("Step 3: Regional Prompts via Latent Blending")
    print("=" * 50)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pipeline
    print("\nLoading SDXL pipeline...")
    # Use float32 on MPS to avoid dtype mismatch errors
    use_fp32 = device == "mps"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32 if use_fp32 else torch.float16,
        variant=None if use_fp32 else "fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe.to(device)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Define prompts
    base_style = "aerial view, photorealistic, top-down, nature landscape, high detail"
    grass_prompt = f"lush green grass field, {base_style}"
    water_prompt = f"calm blue water pond, {base_style}"
    negative_prompt = "cartoon, blurry, illustration"

    # Create masks (for latent space: 128x96 for 1024x768 image)
    left_mask, right_mask = create_left_right_masks(height=96, width=128)

    print(f"\nPrompts:")
    print(f"  Left: {grass_prompt}")
    print(f"  Right: {water_prompt}")

    # --- BASELINE ---
    print("\n--- Generating BASELINE (single prompt) ---")
    generator = torch.Generator(device=device).manual_seed(42)
    baseline = pipe(
        prompt=f"green grass and blue water, {base_style}",
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    baseline.save(output_dir / "step3_baseline.png")
    print("  Saved step3_baseline.png")

    # --- REGIONAL: grass left, water right ---
    print("\n--- Generating REGIONAL (grass LEFT, water RIGHT) ---")
    regional = generate_with_regional_blending(
        pipe,
        prompts=[grass_prompt, water_prompt],
        masks=[left_mask, right_mask],
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=42,
    )
    regional.save(output_dir / "step3_regional.png")
    print("  Saved step3_regional.png")

    # --- SWAPPED: water left, grass right ---
    print("\n--- Generating SWAPPED (water LEFT, grass RIGHT) ---")
    swapped = generate_with_regional_blending(
        pipe,
        prompts=[water_prompt, grass_prompt],
        masks=[left_mask, right_mask],
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=42,
    )
    swapped.save(output_dir / "step3_swapped.png")
    print("  Saved step3_swapped.png")

    print("\n" + "=" * 50)
    print("Step 3 Complete!")
    print("=" * 50)
    print("\nCompare:")
    print("  - step3_baseline.png: Single prompt")
    print("  - step3_regional.png: Grass LEFT, Water RIGHT")
    print("  - step3_swapped.png:  Water LEFT, Grass RIGHT")


if __name__ == "__main__":
    main()
