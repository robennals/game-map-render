#!/usr/bin/env python3
"""Step 3: Test if modifying cross-attention actually controls the output.

Simple test: boost "grass" attention on left half, "water" on right half.
If output shows grass-left/water-right, the approach works.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention
from PIL import Image
import numpy as np


class RegionalPromptAttnProcessor:
    """Attention processor that applies spatial masks to boost certain tokens in certain regions."""

    def __init__(
        self,
        token_masks: Dict[int, torch.Tensor],  # token_idx -> spatial mask
        boost_factor: float = 20.0,  # Additive bias strength for attention logits
    ):
        """
        Args:
            token_masks: Maps token indices to spatial masks.
                         Masks should be [height, width] with values 0-1.
            boost_factor: Additive bias applied to attention logits.
                         Positive in masked regions, negative outside.
        """
        self.token_masks = token_masks
        self.boost_factor = boost_factor
        self._mask_cache = {}  # Cache resized masks

    def _get_mask_for_size(self, token_idx: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Get mask resized to match the current attention spatial dimensions."""
        if token_idx not in self.token_masks:
            return None

        cache_key = (token_idx, seq_len)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key].to(device)

        mask = self.token_masks[token_idx]  # [H, W]
        h, w = mask.shape

        # Figure out target dimensions from seq_len
        # Try to find h, w such that h*w = seq_len with ~4:3 ratio
        target_h = target_w = None
        for th in range(1, int(seq_len ** 0.5) + 20):
            if seq_len % th == 0:
                tw = seq_len // th
                if 1.2 <= tw / th <= 1.5:
                    target_h, target_w = th, tw
                    break

        if target_h is None:
            # Fallback for square-ish
            target_h = int(seq_len ** 0.5)
            target_w = seq_len // target_h
            if target_h * target_w != seq_len:
                return None  # Can't figure out dimensions

        # Resize mask
        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze()  # [target_h, target_w]

        # Flatten to [seq_len]
        mask_flat = mask_resized.flatten()
        self._mask_cache[cache_key] = mask_flat

        return mask_flat.to(device)

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
        is_cross_attention = encoder_hidden_states is not None

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Compute attention scores
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        # Shape: [batch, heads, spatial_seq_len, text_seq_len]

        # Apply spatial masks to boost/suppress certain tokens in certain regions
        # Using ADDITIVE bias on logits (before softmax) for stronger effect
        if is_cross_attention and self.token_masks:
            spatial_seq_len = attn_weights.shape[2]

            for token_idx, _ in self.token_masks.items():
                mask = self._get_mask_for_size(token_idx, spatial_seq_len, attn_weights.device)
                if mask is not None:
                    # mask shape: [spatial_seq_len], values 0-1
                    # Where mask is high, boost attention to this token
                    # Where mask is low, suppress attention to this token
                    # Convert mask from [0,1] to [-boost, +boost] range
                    bias = (mask * 2.0 - 1.0) * self.boost_factor  # [-boost, +boost]
                    bias = bias.view(1, 1, -1)  # [1, 1, spatial]

                    # Add bias to attention logits for this token
                    attn_weights[:, :, :, token_idx] = attn_weights[:, :, :, token_idx] + bias

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        hidden_states = torch.matmul(attn_weights, value)

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def create_left_right_masks(height: int = 96, width: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Create simple left-half and right-half masks with soft edges."""
    # Create coordinate grid
    x = torch.linspace(0, 1, width)

    # Left mask: 1 on left, fades to 0 on right
    # Using sigmoid for smooth transition
    left_mask = torch.sigmoid((0.5 - x) * 10)  # Sharp-ish transition at middle
    left_mask = left_mask.unsqueeze(0).expand(height, -1)

    # Right mask: opposite
    right_mask = 1.0 - left_mask

    return left_mask, right_mask


def set_regional_processors(unet, token_masks: Dict[int, torch.Tensor], boost_factor: float = 3.0):
    """Set all attention processors to use regional prompting."""
    attn_procs = {}
    for name in unet.attn_processors.keys():
        # Only apply to cross-attention layers (attn2)
        if "attn2" in name:
            attn_procs[name] = RegionalPromptAttnProcessor(token_masks, boost_factor)
        else:
            # Use default processor for self-attention
            from diffusers.models.attention_processor import AttnProcessor2_0
            attn_procs[name] = AttnProcessor2_0()
    unet.set_attn_processor(attn_procs)


def main():
    print("Step 3: Test Regional Attention Modification")
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

    # Our test prompt - we want grass on left, water on right
    prompt = "aerial view of green grass and blue water, photorealistic, top-down, nature landscape"
    negative_prompt = "cartoon, illustration, blurry, text"

    # Find token indices for grass and water
    tokenizer = pipe.tokenizer
    tokens = tokenizer.encode(prompt)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {token_strings}")

    grass_idx = None
    water_idx = None
    for i, tok in enumerate(token_strings):
        if "grass" in tok.lower():
            grass_idx = i
            print(f"  'grass' at index {i}")
        if "water" in tok.lower():
            water_idx = i
            print(f"  'water' at index {i}")

    if grass_idx is None or water_idx is None:
        print("ERROR: Couldn't find grass or water tokens!")
        return

    # Create left/right masks
    # Latent size for 1024x768 is 128x96
    left_mask, right_mask = create_left_right_masks(height=96, width=128)
    print(f"\nMasks created: left={left_mask.shape}, right={right_mask.shape}")

    # Save mask visualizations
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    left_vis = (left_mask.numpy() * 255).astype(np.uint8)
    right_vis = (right_mask.numpy() * 255).astype(np.uint8)
    Image.fromarray(left_vis).resize((1024, 768)).save(output_dir / "step3_mask_left.png")
    Image.fromarray(right_vis).resize((1024, 768)).save(output_dir / "step3_mask_right.png")
    print("Saved mask visualizations")

    # First: generate WITHOUT regional attention (baseline)
    print("\n--- Generating BASELINE (no regional attention) ---")
    generator = torch.Generator(device=device).manual_seed(42)
    start = time.time()
    baseline_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    print(f"Baseline generated in {time.time() - start:.1f}s")
    baseline_image.save(output_dir / "step3_baseline.png")

    # Second: generate WITH regional attention (grass=left, water=right)
    print("\n--- Generating WITH regional attention (grass LEFT, water RIGHT) ---")
    token_masks = {
        grass_idx: left_mask,
        water_idx: right_mask,
    }
    set_regional_processors(pipe.unet, token_masks, boost_factor=20.0)

    generator = torch.Generator(device=device).manual_seed(42)
    start = time.time()
    regional_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    print(f"Regional generated in {time.time() - start:.1f}s")
    regional_image.save(output_dir / "step3_regional.png")

    # Third: generate with SWAPPED regions (grass=right, water=left)
    print("\n--- Generating WITH regional attention (grass RIGHT, water LEFT) ---")
    token_masks_swapped = {
        grass_idx: right_mask,
        water_idx: left_mask,
    }
    set_regional_processors(pipe.unet, token_masks_swapped, boost_factor=20.0)

    generator = torch.Generator(device=device).manual_seed(42)
    start = time.time()
    swapped_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    print(f"Swapped generated in {time.time() - start:.1f}s")
    swapped_image.save(output_dir / "step3_swapped.png")

    print("\n" + "=" * 50)
    print("Step 3 Complete!")
    print("=" * 50)
    print("\nCompare the images:")
    print("  - step3_baseline.png: No regional control")
    print("  - step3_regional.png: Grass boosted LEFT, water boosted RIGHT")
    print("  - step3_swapped.png: Grass boosted RIGHT, water boosted LEFT")
    print("\nIf regional control works, grass/water positions should swap between")
    print("regional and swapped images!")


if __name__ == "__main__":
    main()
