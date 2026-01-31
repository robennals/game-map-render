#!/usr/bin/env python3
"""Generate game map images from tilemaps using regional prompt conditioning.

Uses latent blending to apply different prompts to different spatial regions,
guided by a tilemap that specifies where each terrain type should appear.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np


# Terrain type to visualization color (RGB)
TERRAIN_COLORS = {
    "trees": (34, 139, 34),       # Forest green
    "grass": (124, 252, 0),       # Lawn green
    "water": (30, 144, 255),      # Dodger blue
    "path": (139, 90, 43),        # Brown
    "sand": (238, 214, 175),      # Sandy beige
    "wall": (128, 128, 128),      # Gray
    "lava": (255, 69, 0),         # Red-orange
    "snow": (255, 250, 250),      # Snow white
    "stone": (105, 105, 105),     # Dim gray
    "dirt": (139, 69, 19),        # Saddle brown
    "bridge": (160, 82, 45),      # Sienna
    "flowers": (255, 105, 180),   # Hot pink
}

# Terrain type to prompt mapping
TERRAIN_PROMPTS = {
    "trees": "dense forest with tall trees, lush green foliage",
    "grass": "lush green grass field, meadow",
    "water": "calm blue water, pond, lake surface",
    "path": "dirt path, brown walking trail",
    "sand": "sandy beach, golden sand",
    "wall": "stone wall, gray bricks",
    "lava": "molten lava, glowing orange magma",
    "snow": "white snow, snowy ground",
    "stone": "gray stone floor, rocky ground",
    "dirt": "brown dirt, soil",
    "bridge": "wooden bridge planks",
    "flowers": "colorful wildflowers in grass",
}


def load_tilemap(path: Path) -> Tuple[str, List[str], Dict[str, str]]:
    """Load tilemap from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["description"], data["tiles"], data["legend"]


def create_tilemap_visualization(
    tiles: List[str],
    legend: Dict[str, str],
    output_width: int,
    output_height: int,
) -> Image.Image:
    """Create a color-coded visualization of the tilemap."""
    tile_height = len(tiles)
    tile_width = len(tiles[0])

    # Create RGB image at tile resolution
    img_array = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

    for y, row in enumerate(tiles):
        for x, char in enumerate(row):
            terrain = legend.get(char, "unknown")
            color = TERRAIN_COLORS.get(terrain, (255, 0, 255))  # Magenta for unknown
            img_array[y, x] = color

    # Scale up to output resolution
    img = Image.fromarray(img_array)
    img = img.resize((output_width, output_height), Image.Resampling.NEAREST)

    return img


def create_overlay_comparison(
    generated: Image.Image,
    tilemap_vis: Image.Image,
    alpha: float = 0.4,
) -> Image.Image:
    """Create an overlay of the tilemap visualization on the generated image."""
    # Ensure both images are RGB
    generated = generated.convert("RGB")
    tilemap_vis = tilemap_vis.convert("RGB")

    # Blend them
    overlay = Image.blend(generated, tilemap_vis, alpha)
    return overlay


def create_terrain_masks(
    tiles: List[str],
    legend: Dict[str, str],
    output_height: int,
    output_width: int,
) -> Dict[str, torch.Tensor]:
    """Create soft-edged masks for each terrain type.

    Returns a dict mapping terrain name to a [height, width] mask tensor.
    """
    tile_height = len(tiles)
    tile_width = len(tiles[0])

    # Create binary masks for each terrain type at tile resolution
    # Multiple legend entries can map to the same terrain, so accumulate
    terrain_masks = {}
    for char, terrain in legend.items():
        if terrain not in terrain_masks:
            terrain_masks[terrain] = np.zeros((tile_height, tile_width), dtype=np.float32)
        for y, row in enumerate(tiles):
            for x, c in enumerate(row):
                if c == char:
                    terrain_masks[terrain][y, x] = 1.0

    # Remove empty masks
    terrain_masks = {k: v for k, v in terrain_masks.items() if v.sum() > 0}

    # Upscale masks to output resolution with soft edges
    result = {}
    for terrain, mask in terrain_masks.items():
        # Convert to tensor and add batch/channel dims for interpolation
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        # Upscale with bilinear interpolation (creates soft edges)
        upscaled = F.interpolate(
            mask_tensor,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False,
        ).squeeze()

        # Apply slight blur for smoother transitions
        # Using a simple box blur approximation
        kernel_size = max(3, output_width // tile_width // 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padded = F.pad(upscaled.unsqueeze(0).unsqueeze(0),
                       (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2),
                       mode='reflect')
        blurred = F.avg_pool2d(padded, kernel_size, stride=1).squeeze()

        result[terrain] = blurred

    return result


@torch.no_grad()
def generate_from_tilemap(
    pipe,
    description: str,
    terrain_masks: Dict[str, torch.Tensor],
    width: int = 1024,
    height: int = 768,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> Image.Image:
    """Generate an image using regional prompt conditioning based on terrain masks."""
    device = pipe.device
    model_dtype = next(pipe.unet.parameters()).dtype

    # Base style for all prompts
    base_style = f"aerial view, photorealistic, top-down, game map, {description}, high detail, seamless"
    negative_prompt = "cartoon, blurry, illustration, text, watermark, low quality"

    # Build prompts and masks lists
    prompts = []
    masks = []
    for terrain, mask in terrain_masks.items():
        terrain_desc = TERRAIN_PROMPTS.get(terrain, terrain)
        prompt = f"{terrain_desc}, {base_style}"
        prompts.append(prompt)
        masks.append(mask)

    print(f"  Generating with {len(prompts)} terrain regions:")
    for i, (terrain, prompt) in enumerate(zip(terrain_masks.keys(), prompts)):
        coverage = masks[i].mean().item() * 100
        print(f"    {terrain}: {coverage:.1f}% coverage")

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

    # Normalize masks to sum to 1 at each position
    mask_sum = sum(mask_tensors)
    mask_tensors = [m / (mask_sum + 1e-8) for m in mask_tensors]

    # Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Create initial latents
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=model_dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    # Add time embeddings for SDXL
    text_encoder_projection_dim = pooled_embeds_list[0].shape[-1]
    add_time_ids = pipe._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=model_dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)
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
        blended_pred = torch.zeros_like(noise_preds[0])
        for noise_pred, mask in zip(noise_preds, mask_tensors):
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(noise_pred)
            blended_pred = blended_pred + noise_pred * mask_expanded

        # Perform CFG
        noise_pred_uncond, noise_pred_cond = blended_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if (i + 1) % 5 == 0:
            print(f"    Step {i + 1}/{num_inference_steps}")

    # Decode latents to image
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image


def main():
    parser = argparse.ArgumentParser(description="Generate game map from tilemap")
    parser.add_argument("tilemap", type=Path, help="Path to tilemap JSON file")
    parser.add_argument("-o", "--output", type=Path, help="Output image path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--height", type=int, default=768, help="Output height")
    args = parser.parse_args()

    print("Game Map Generator - Regional Prompt Conditioning")
    print("=" * 50)

    # Load tilemap
    print(f"\nLoading tilemap: {args.tilemap}")
    description, tiles, legend = load_tilemap(args.tilemap)
    print(f"  Description: {description}")
    print(f"  Size: {len(tiles[0])}x{len(tiles)} tiles")
    print(f"  Terrain types: {list(legend.values())}")

    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load pipeline
    print("\nLoading SDXL pipeline...")
    use_fp32 = device == "mps"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32 if use_fp32 else torch.float16,
        variant=None if use_fp32 else "fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe.to(device)

    # Create terrain masks
    print("\nCreating terrain masks...")
    terrain_masks = create_terrain_masks(tiles, legend, args.height, args.width)

    # Save visualizations
    output_dir = args.output.parent if args.output else Path("output")
    output_dir.mkdir(exist_ok=True)

    # Create and save tilemap color visualization
    tilemap_vis = create_tilemap_visualization(tiles, legend, args.width, args.height)
    tilemap_vis_path = output_dir / f"{args.tilemap.stem}_tilemap.png"
    tilemap_vis.save(tilemap_vis_path)
    print(f"  Saved tilemap visualization: {tilemap_vis_path}")

    # Save individual terrain masks
    for terrain, mask in terrain_masks.items():
        mask_img = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
        mask_path = output_dir / f"mask_{terrain}.png"
        mask_img.save(mask_path)
        print(f"  Saved mask: {mask_path}")

    # Generate image
    print("\nGenerating image...")
    image = generate_from_tilemap(
        pipe,
        description=description,
        terrain_masks=terrain_masks,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    # Save output
    output_path = args.output or (output_dir / f"{args.tilemap.stem}_generated.png")
    image.save(output_path)
    print(f"\nSaved output: {output_path}")

    # Create and save overlay comparison
    overlay = create_overlay_comparison(image, tilemap_vis, alpha=0.4)
    overlay_path = output_dir / f"{args.tilemap.stem}_overlay.png"
    overlay.save(overlay_path)
    print(f"Saved overlay comparison: {overlay_path}")

    print("\n" + "=" * 50)
    print("Generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
