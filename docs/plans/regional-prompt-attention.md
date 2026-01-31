# Regional Prompt Attention for Tilemap Generation

## Problem Statement

We want to generate game level images from tilemaps where:
- Each tile (e.g., grass, water, trees) should look like its terrain type
- Transitions between terrain types should be natural and seamless
- The overall image should be coherent

## Why Previous Approaches Failed

### Tile-by-tile generation
- 64x64 tiles are too small for SDXL (needs 512x512 minimum)
- Even with blending, visible seams appear at tile boundaries

### ControlNet Segmentation
- Uses hard-edged color regions to define "this area = grass"
- Model either follows boundaries too strictly (artificial look) or ignores them
- No natural blending between terrain types
- Requires specific ADE20K color palette that may not map well to game terrains

## The Solution: Regional Prompt Attention

### Core Idea
Instead of using a segmentation map, we directly manipulate the cross-attention mechanism:
- Each tile area has its own text prompt (e.g., "lush green grass", "clear blue water")
- The influence of each prompt is spatially weighted using soft masks
- Masks are strongest at tile centers, fade toward edges (Gaussian falloff)
- This allows natural blending where terrains meet

### How Cross-Attention Works in Stable Diffusion

In the UNet's cross-attention layers:
1. Image latents form **queries (Q)** - "what should this pixel look like?"
2. Text embeddings form **keys (K)** and **values (V)** - "here's what the prompt describes"
3. Attention scores = softmax(Q @ K^T / sqrt(d))
4. Output = attention_scores @ V

The attention scores determine how much each spatial location "listens to" each text token.

### Our Modification

We create a custom `AttentionProcessor` that:
1. Encodes multiple prompts (one per terrain type)
2. Computes attention scores for each prompt
3. Weights each prompt's contribution by spatial masks
4. Blends the results

```
For each spatial location (x, y):
    total_attention = 0
    for each terrain_type:
        mask_weight = gaussian_mask[terrain_type][x, y]
        attention_to_terrain = cross_attention(Q[x,y], K[terrain_type], V[terrain_type])
        total_attention += mask_weight * attention_to_terrain
    output[x, y] = total_attention
```

### Mask Design

For a 16x12 tilemap generating a 1024x768 image:
- Latent space is 128x96 (image_size / 8)
- Each tile covers ~8x8 latent pixels
- Gaussian mask centered on each tile, sigma ~4-6 pixels
- Masks overlap at boundaries, naturally blend

```
mask[tile](x, y) = exp(-((x - tile_center_x)^2 + (y - tile_center_y)^2) / (2 * sigma^2))
```

Normalize masks so they sum to 1 at each location.

## Implementation Plan

### Step 1: Minimal SDXL generation (verify base works)
- Load SDXL, generate a simple image with one prompt
- Verify MPS works, get baseline timing
- **Verification**: Image generates successfully

### Step 2: Extract and visualize cross-attention maps
- Hook into attention layers, capture attention scores
- Visualize which image regions attend to which tokens
- **Verification**: Can see attention heatmaps for words like "grass", "water"

### Step 3: Create soft spatial masks for tiles
- Given a tilemap, generate per-tile Gaussian masks
- Masks are in latent space dimensions (64x48 for 1024x768 output)
- **Verification**: Visualize masks, confirm smooth falloff at edges

### Step 4: Encode multiple prompts
- Encode one prompt per terrain type ("lush green grass", "blue water", etc.)
- Store the text embeddings for each
- **Verification**: Print embedding shapes, confirm they're compatible

### Step 5: Custom AttentionProcessor with regional prompts
- Create processor that blends attention across prompts weighted by masks
- Start simple: just 2 regions (e.g., grass + water)
- **Verification**: Generated image shows grass/water in roughly correct regions

### Step 6: Full tilemap integration
- Extend to all terrain types from tilemap
- Add base scene prompt that applies everywhere (but weaker)
- **Verification**: Generate forest.json, visually check terrain placement

### Step 7: Tune blending parameters
- Adjust mask falloff (sharper vs softer transitions)
- Adjust base prompt vs regional prompt strength
- **Verification**: Transitions look natural, no harsh boundaries

## Technical References

### Diffusers AttentionProcessor
Custom attention processors can be created by implementing `__call__`:

```python
class RegionalPromptAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,        # Image latents (queries)
        encoder_hidden_states: torch.Tensor, # Text embeddings (keys/values)
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Compute attention scores
        # ... modify here based on spatial masks ...

        return output
```

Set processor via: `unet.set_attn_processor(processor_dict)`

### Key Resources
- [Diffusers attention_processor.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
- [Cross Attention Control](https://github.com/bloc97/CrossAttentionControl)
- [Prompt-to-Prompt paper](https://prompt-to-prompt.github.io/)
- [Training-free Regional Prompting](https://arxiv.org/html/2411.02395v1)

## Parameters to Tune

| Parameter | Description | Starting Value |
|-----------|-------------|----------------|
| `mask_sigma` | Gaussian falloff in latent pixels | 4-6 |
| `base_prompt_weight` | How much the scene prompt affects all regions | 0.3 |
| `regional_prompt_weight` | How much tile prompts affect their regions | 0.7 |
| `blend_schedule` | Whether to vary blending over diffusion steps | None initially |

## Potential Challenges

1. **Attention shape mismatch**: Different prompts have different token counts. May need padding or separate attention computations.

2. **Multi-head attention**: Need to apply masks correctly across all attention heads.

3. **Multiple attention layers**: UNet has many cross-attention layers at different resolutions. Need to handle all of them.

4. **SDXL dual text encoders**: SDXL uses CLIP + OpenCLIP. Need to handle both sets of embeddings.

5. **Performance**: Computing attention for multiple prompts is more expensive. May need optimization.
