#!/usr/bin/env python3
"""Step 3: Regional prompt conditioning - the simple way.

Instead of hacking attention weights, we:
1. Encode multiple prompts separately
2. Compute attention to each prompt
3. Blend results based on spatial masks

This is just like normal prompt conditioning, but spatially varying.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention
from PIL import Image
import numpy as np


class RegionalPromptProcessor:
    """Computes attention to multiple prompts and blends based on spatial masks."""

    def __init__(
        self,
        regional_embeds: List[torch.Tensor],  # List of encoder_hidden_states for each region
        regional_masks: List[torch.Tensor],   # List of spatial masks [H, W] for each region
    ):
        self.regional_embeds = regional_embeds  # Each: [batch, seq_len, dim]
        self.regional_masks = regional_masks    # Each: [H, W] with values 0-1
        self._mask_cache = {}

    def _get_masks_for_size(self, spatial_seq_len: int, device: torch.device) -> List[torch.Tensor]:
        """Resize all masks to match current spatial dimensions."""
        if spatial_seq_len in self._mask_cache:
            return [m.to(device) for m in self._mask_cache[spatial_seq_len]]

        # Figure out spatial dimensions
        target_h = target_w = None
        for h in range(1, int(spatial_seq_len ** 0.5) + 20):
            if spatial_seq_len % h == 0:
                w = spatial_seq_len // h
                if 1.2 <= w / h <= 1.5:
                    target_h, target_w = h, w
                    break

        if target_h is None:
            target_h = int(spatial_seq_len ** 0.5)
            target_w = spatial_seq_len // target_h

        resized_masks = []
        for mask in self.regional_masks:
            resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze()
            resized_masks.append(resized.flatten())  # [spatial_seq_len]

        self._mask_cache[spatial_seq_len] = resized_masks
        return [m.to(device) for m in resized_masks]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # For self-attention (encoder_hidden_states is None), just do normal attention
        if encoder_hidden_states is None:
            return self._normal_attention(attn, hidden_states, hidden_states, attention_mask, temb)

        # Cross-attention: blend attention to multiple prompts
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, spatial_seq_len, _ = hidden_states.shape

        # Get masks resized for this spatial resolution
        masks = self._get_masks_for_size(spatial_seq_len, hidden_states.device)

        # Normalize masks so they sum to 1 at each location
        mask_sum = sum(masks)
        masks = [m / (mask_sum + 1e-8) for m in masks]

        # Compute attention output for each regional prompt
        # Use diffusers' built-in attention which handles all the dimension complexity

        blended_output = None
        orig_dtype = hidden_states.dtype

        for region_idx, region_embed in enumerate(self.regional_embeds):
            # Get embeddings for this region's prompt
            region_embed = region_embed.to(hidden_states.device, dtype=orig_dtype)

            if attn.norm_cross:
                region_embed = attn.norm_encoder_hidden_states(region_embed)

            # First project Q, K, V
            query = attn.to_q(hidden_states)
            key = attn.to_k(region_embed)
            value = attn.to_v(region_embed)

            # Query dimensions
            q_inner_dim = query.shape[-1]
            q_head_dim = q_inner_dim // attn.heads

            # Key/value dimensions (may be different due to GQA)
            kv_inner_dim = key.shape[-1]
            # For attention to work, head_dim must match between Q and K
            # So compute kv_heads from the inner dims
            kv_heads = kv_inner_dim // q_head_dim

            # Debug: print dimensions on first call
            if not hasattr(self, '_debug_printed'):
                print(f"DEBUG: attn.heads={attn.heads}, q_inner={q_inner_dim}, q_head_dim={q_head_dim}")
                print(f"DEBUG: kv_inner={kv_inner_dim}, kv_heads={kv_heads}")
                print(f"DEBUG: key.shape={key.shape}, key.numel()={key.numel()}")
                self._debug_printed = True

            # Reshape to [batch, heads, seq, head_dim] for SDPA
            query = query.view(batch_size, -1, attn.heads, q_head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, kv_heads, q_head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, kv_heads, q_head_dim).transpose(1, 2)

            # Expand KV heads to match Q heads for SDPA (it doesn't handle GQA natively)
            if kv_heads != attn.heads:
                n_rep = attn.heads // kv_heads
                key = key.repeat_interleave(n_rep, dim=1)
                value = value.repeat_interleave(n_rep, dim=1)

            # Use PyTorch's scaled_dot_product_attention
            region_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            # Reshape back to [batch, seq, inner_dim]
            region_output = region_output.transpose(1, 2).reshape(batch_size, -1, q_inner_dim)
            # region_output: [batch, spatial_seq_len, inner_dim]

            # Weight by spatial mask
            mask = masks[region_idx]  # [spatial_seq_len]
            mask = mask.view(1, -1, 1)  # [1, spatial_seq_len, 1]

            if blended_output is None:
                blended_output = mask * region_output
            else:
                blended_output = blended_output + mask * region_output

        hidden_states = blended_output

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _normal_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb):
        """Standard attention for self-attention layers."""
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Use PyTorch 2.0 scaled_dot_product_attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def create_left_right_masks(height: int = 96, width: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create left-half and right-half masks with soft edges."""
    x = torch.linspace(0, 1, width)

    # Smooth transition in the middle
    left_mask = torch.sigmoid((0.5 - x) * 10)
    left_mask = left_mask.unsqueeze(0).expand(height, -1)

    right_mask = 1.0 - left_mask

    return left_mask, right_mask


def encode_prompt(pipe, prompt: str, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a prompt using both SDXL text encoders.

    Returns: (positive_embeds, negative_embeds, positive_pooled, negative_pooled)
    """
    # SDXL uses two text encoders
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
        pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
            negative_prompt_2="",
        )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def main():
    print("Step 3: Regional Prompt Conditioning (Simple Approach)")
    print("=" * 50)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pipeline
    print("\nLoading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe.to(device)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Define regional prompts
    base_style = "aerial view, photorealistic, top-down, nature landscape, high detail"
    grass_prompt = f"lush green grass field, {base_style}"
    water_prompt = f"calm blue water pond, {base_style}"

    print(f"\nGrass prompt: {grass_prompt}")
    print(f"Water prompt: {water_prompt}")

    # Encode prompts
    print("\nEncoding prompts...")
    grass_pos, grass_neg, grass_pooled_pos, grass_pooled_neg = encode_prompt(pipe, grass_prompt, device)
    water_pos, water_neg, water_pooled_pos, water_pooled_neg = encode_prompt(pipe, water_prompt, device)

    print(f"  Grass pos embeds shape: {grass_pos.shape}")
    print(f"  Grass neg embeds shape: {grass_neg.shape}")

    # Create masks
    left_mask, right_mask = create_left_right_masks(height=96, width=128)
    print(f"\nMasks: left={left_mask.shape}, right={right_mask.shape}")

    # Save mask visualizations
    Image.fromarray((left_mask.numpy() * 255).astype(np.uint8)).resize((1024, 768)).save(
        output_dir / "step3_mask_left.png"
    )
    Image.fromarray((right_mask.numpy() * 255).astype(np.uint8)).resize((1024, 768)).save(
        output_dir / "step3_mask_right.png"
    )

    # --- BASELINE: single prompt ---
    print("\n--- Generating BASELINE (single prompt: grass and water) ---")
    generator = torch.Generator(device=device).manual_seed(42)
    baseline = pipe(
        prompt=f"green grass on left and blue water on right, {base_style}",
        negative_prompt="cartoon, blurry",
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

    # Set up regional processors
    # For CFG, we need to handle both unconditional and conditional
    # The batch during inference is [uncond, cond], so we need embeds for both
    # Each regional embed should be [2, 77, 2048] = [neg, pos] concatenated
    regional_embeds = [
        torch.cat([grass_neg, grass_pos], dim=0),  # Region 0 (left): grass
        torch.cat([grass_neg, water_pos], dim=0),  # Region 1 (right): water
    ]
    print(f"  Regional embeds shapes: {[e.shape for e in regional_embeds]}")
    regional_masks = [left_mask, right_mask]

    processor = RegionalPromptProcessor(regional_embeds, regional_masks)

    # Apply to all cross-attention layers
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:  # Cross-attention
            attn_procs[name] = processor
        else:  # Self-attention - use default
            from diffusers.models.attention_processor import AttnProcessor2_0
            attn_procs[name] = AttnProcessor2_0()
    pipe.unet.set_attn_processor(attn_procs)

    generator = torch.Generator(device=device).manual_seed(42)

    # We need to bypass the normal prompt encoding since we're injecting our own
    # Use a dummy prompt and manually set the embeds
    regional = pipe(
        prompt="nature landscape",  # Dummy, will be overridden by our processor
        negative_prompt="cartoon, blurry",
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    regional.save(output_dir / "step3_regional.png")
    print("  Saved step3_regional.png")

    # --- SWAPPED: grass right, water left ---
    print("\n--- Generating SWAPPED (grass RIGHT, water LEFT) ---")

    regional_embeds_swapped = [
        torch.cat([grass_neg, water_pos], dim=0),  # Region 0 (left): water
        torch.cat([grass_neg, grass_pos], dim=0),  # Region 1 (right): grass
    ]

    processor_swapped = RegionalPromptProcessor(regional_embeds_swapped, regional_masks)

    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name] = processor_swapped
        else:
            from diffusers.models.attention_processor import AttnProcessor2_0
            attn_procs[name] = AttnProcessor2_0()
    pipe.unet.set_attn_processor(attn_procs)

    generator = torch.Generator(device=device).manual_seed(42)
    swapped = pipe(
        prompt="nature landscape",
        negative_prompt="cartoon, blurry",
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
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
