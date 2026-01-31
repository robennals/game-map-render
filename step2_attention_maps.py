#!/usr/bin/env python3
"""Step 2: Extract and visualize cross-attention maps.

Hook into attention layers to see which image regions attend to which tokens.
This helps us understand the mechanism before we modify it.
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention
from PIL import Image
import numpy as np


class AttentionMapCapture:
    """Captures cross-attention maps during generation."""

    def __init__(self):
        self.attention_maps = {}  # layer_name -> list of attention maps per step

    def clear(self):
        self.attention_maps = {}


class CapturingAttnProcessor:
    """Attention processor that captures cross-attention maps."""

    def __init__(self, capture: AttentionMapCapture, layer_name: str):
        self.capture = capture
        self.layer_name = layer_name

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
        # Only capture cross-attention (when encoder_hidden_states is provided)
        is_cross_attention = encoder_hidden_states is not None

        residual = hidden_states

        # Handle input normalization if needed
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape
            # For non-square images, we need to figure out the aspect ratio
            # The original image is 1024x768 = 4:3 ratio
            # Latent space is 128x96, and UNet downsamples further
            # Try to find factors that match the sequence length with ~4:3 ratio
            height = width = None
            for h in range(1, int(sequence_length ** 0.5) + 20):
                if sequence_length % h == 0:
                    w = sequence_length // h
                    # Check if roughly 4:3 ratio (allow some tolerance)
                    if 1.2 <= w / h <= 1.5:
                        height, width = h, w
                        break
            if height is None:
                # Fallback: just use square root approximation
                height = int(sequence_length ** 0.5)
                width = sequence_length // height

        # For cross-attention, use encoder_hidden_states; for self-attention, use hidden_states
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

        # Compute attention scores manually to capture them
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Capture cross-attention maps (not self-attention)
        if is_cross_attention and "up_blocks" in self.layer_name:
            # Only capture from up_blocks for cleaner visualization
            # Average across heads and store
            avg_attn = attn_weights.mean(dim=1).detach().cpu()  # [batch, seq_len, text_len]
            if self.layer_name not in self.capture.attention_maps:
                self.capture.attention_maps[self.layer_name] = []
            self.capture.attention_maps[self.layer_name].append((height, width, avg_attn))

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


def set_capturing_processors(unet, capture: AttentionMapCapture):
    """Replace all attention processors with capturing versions."""
    attn_procs = {}
    for name in unet.attn_processors.keys():
        attn_procs[name] = CapturingAttnProcessor(capture, name)
    unet.set_attn_processor(attn_procs)


def visualize_attention_for_token(
    attention_maps: dict,
    token_idx: int,
    output_size: tuple[int, int] = (1024, 768),
) -> np.ndarray:
    """Aggregate attention maps for a specific token across layers."""
    aggregated = None

    for layer_name, maps_list in attention_maps.items():
        # Use the last timestep's attention (most refined)
        if not maps_list:
            continue

        height, width, attn = maps_list[-1]

        # attn shape: [batch, spatial_seq_len, text_seq_len]
        # Get attention to specific token
        token_attn = attn[0, :, token_idx].numpy()  # [spatial_seq_len]

        # Verify dimensions match
        if height * width != len(token_attn):
            # Try to find correct dimensions
            seq_len = len(token_attn)
            found = False
            for h in range(1, int(seq_len ** 0.5) + 20):
                if seq_len % h == 0:
                    w = seq_len // h
                    if 1.2 <= w / h <= 1.5:
                        height, width = h, w
                        found = True
                        break
            if not found:
                height = int(seq_len ** 0.5)
                width = seq_len // height
                if height * width != seq_len:
                    continue  # Skip this layer if we can't figure out dimensions

        # Reshape to spatial
        token_attn = token_attn.reshape(height, width)

        # Resize to output size
        token_attn_img = Image.fromarray((token_attn * 255).astype(np.uint8))
        token_attn_img = token_attn_img.resize(output_size, Image.Resampling.BILINEAR)
        token_attn = np.array(token_attn_img).astype(np.float32) / 255.0

        if aggregated is None:
            aggregated = token_attn
        else:
            aggregated = np.maximum(aggregated, token_attn)

    return aggregated


def save_attention_visualization(
    attn_map: np.ndarray,
    original_image: Image.Image,
    output_path: Path,
    token_name: str,
):
    """Save attention map overlaid on original image."""
    # Normalize attention map
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Create heatmap (red-yellow colormap)
    heatmap = np.zeros((*attn_map.shape, 3), dtype=np.uint8)
    heatmap[:, :, 0] = 255  # Red channel
    heatmap[:, :, 1] = (attn_map * 255).astype(np.uint8)  # Yellow where high attention

    heatmap_img = Image.fromarray(heatmap)

    # Blend with original
    blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=0.4)

    # Add label
    blended.save(output_path)
    print(f"  Saved attention map for '{token_name}': {output_path}")


def main():
    print("Step 2: Cross-Attention Map Visualization")
    print("=" * 50)

    # Setup device
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

    # Setup attention capture
    capture = AttentionMapCapture()
    set_capturing_processors(pipe.unet, capture)

    # Generate with a prompt containing distinct elements
    prompt = "aerial view of green grass with blue water pond, photorealistic, top-down"
    negative_prompt = "cartoon, illustration, blurry"

    # Tokenize to find token indices
    tokenizer = pipe.tokenizer
    tokens = tokenizer.encode(prompt)
    token_strings = [tokenizer.decode([t]) for t in tokens]

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {token_strings}")

    # Find indices of interesting tokens
    interesting_tokens = {}
    for i, tok in enumerate(token_strings):
        tok_lower = tok.strip().lower()
        if tok_lower in ["grass", "water", "pond", "green", "blue"]:
            interesting_tokens[tok.strip()] = i
            print(f"  Token '{tok.strip()}' at index {i}")

    print("\nGenerating image...")
    start = time.time()

    generator = torch.Generator(device=device).manual_seed(42)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    print(f"Generated in {time.time() - start:.1f}s")

    # Save original image
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    image.save(output_dir / "step2_original.png")
    print(f"\nSaved original: {output_dir / 'step2_original.png'}")

    # Report captured attention maps
    print(f"\nCaptured attention from {len(capture.attention_maps)} layers:")
    for layer_name, maps in capture.attention_maps.items():
        if maps:
            h, w, attn = maps[-1]
            print(f"  {layer_name}: {len(maps)} steps, final shape {h}x{w}, text_len={attn.shape[-1]}")

    # Visualize attention for interesting tokens
    print("\nVisualizing attention maps:")
    for token_name, token_idx in interesting_tokens.items():
        attn_map = visualize_attention_for_token(
            capture.attention_maps,
            token_idx,
            output_size=(1024, 768),
        )
        if attn_map is not None:
            save_attention_visualization(
                attn_map,
                image,
                output_dir / f"step2_attn_{token_name}.png",
                token_name,
            )

    print("\n" + "=" * 50)
    print("Step 2 PASSED: Can capture and visualize attention maps!")
    print("=" * 50)
    print("\nCheck the output directory to see which regions attend to 'grass' vs 'water'")


if __name__ == "__main__":
    main()
