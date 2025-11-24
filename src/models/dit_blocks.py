"""
Core DiT (Diffusion Transformer) blocks for ArtFlow.

This module implements three variants of DiT blocks:
1. UnconditionalDiTBlock: Standard DiT block with AdaLN modulation (timestep only).
2. DoubleStreamDiTBlock: MMDiT-style block with separate weights for image and text streams.
3. SingleStreamDiTBlock: FLUX-style block with fused image+text processing and single modulation.

Common components like TimestepEmbeddings, MSRoPE (Multimodal Scalable RoPE), and FeedForward
are shared across implementations.
"""

from typing import Tuple, Optional
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings"""

    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,)
        Returns:
            embedding: (B, N)
        """
        half = self.hidden_size // 2
        exponent = (
            -math.log(self.max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )

        emb = torch.exp(exponent)
        emb = timesteps.unsqueeze(1).float() * emb.unsqueeze(0)

        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings
    Args:
        x: Input tensor [B, S, H, D]
        freqs_cis: Complex frequency tensor [S, D/2]
    Returns:
        Tensor with rotary embeddings applied [B, S, H, D]
    """
    # Reshape x to [B, S, H, D/2, 2] and view as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape for broadcasting: [S, D/2] -> [1, S, 1,  D/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    x_rotated = x_complex * freqs_cis

    # View as real and flatten the last two dimensions
    x_out = torch.view_as_real(x_rotated).flatten(3)

    return x_out.type_as(x)


class MSRoPE(nn.Module):
    """Multimodal Scalable RoPE for 2D images and text"""

    def __init__(
        self,
        theta: int = 10000,
        axes_dim: list = [64, 64],
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

        # Precompute frequency tables for both spatial axes
        pos_index = torch.arange(2560)  # maximum position index
        # Store as real to avoid safetensors issues with complex numbers
        self.register_buffer(
            "pos_freqs", torch.view_as_real(self._build_frequency_table(pos_index))
        )

        # LRU cache for image frequency computation
        self._cache = {}

    def rope_params(self, index: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Generate RoPE parameters for given dimensions
        Args:
            index: Position indices [S]
            dim: Embedding dimension for this axis
        Returns:
            Complex frequency tensor [S, dim]
        """
        assert dim % 2 == 0

        # Apply scaling
        if self.scaling_type == "linear":
            index = index / self.scaling_factor
            theta = self.theta
        elif self.scaling_type == "ntk":
            theta = self.theta * (self.scaling_factor ** (dim / (dim - 2)))
        else:
            theta = self.theta

        # Compute frequency components: 1/ theta^(2i/dim)
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )

        # Convert to complex polar form: e^(i*freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _build_frequency_table(self, pos_index: torch.Tensor) -> torch.Tensor:
        """Build frequency table for both height and width axes"""
        height_freqs = self.rope_params(pos_index, self.axes_dim[0])
        width_freqs = self.rope_params(pos_index, self.axes_dim[1])

        # Concatenate height and width frequencies: [S, height_dim+width_dim]
        return torch.cat([height_freqs, width_freqs], dim=1)

    def forward(
        self, img_hw: Tuple[int, int], txt_seq_len: int, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute frequency tensors for image and text sequences
        Args:
            img_hw: Image dimensions (height, width)
            txt_seq_len: Length of text sequence
            device: Device for tensor placement
        Returns:
            Tuple of (image_freqs, text_freqs) tensors
        """
        height, width = img_hw

        # Check if we need to expand the frequency table
        max_needed = max(height, width) + txt_seq_len
        current_max = self.pos_freqs.shape[0]

        if max_needed > current_max:
            # Expand by 2x or to max_needed, whichever is larger
            new_max = max(max_needed, current_max * 2)
            # print(f"Expanding RoPE frequency table from {current_max} to {new_max}")
            pos_index = torch.arange(new_max, device=device)
            self.pos_freqs = torch.view_as_real(self._build_frequency_table(pos_index))

        # Ensure frequencies are on correct device
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)

        # Get cached image frequencies or compute new ones
        cache_key = (height, width)
        if cache_key in self._cache:
            img_freqs = self._cache[cache_key]
        else:
            img_freqs = self._compute_image_freqs(height, width)
            self._cache[cache_key] = img_freqs
        img_freqs = img_freqs.to(device)

        # Text frequencies start after maximum image position
        max_img_pos = max(height, width)
        # View as complex for slicing
        pos_freqs_complex = torch.view_as_complex(self.pos_freqs)
        txt_freqs = pos_freqs_complex[
            max_img_pos : max_img_pos + txt_seq_len, :
        ]  # placing text tokens on a diagonal in the 2D position space

        return img_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_image_freqs(self, height: int, width: int) -> torch.Tensor:
        """
        Compute frequency tensor for 2D image grid
        Args:
            height: Height of image
            width: Width of image
        Returns:
            Frequency tensor [height*width, total_dim] (complex)
        """
        # Split precomputed frequencies by axis
        # pos_freqs is [S, D, 2] (real)
        h_dim, w_dim = self.axes_dim
        h_freqs, w_freqs = self.pos_freqs.split([h_dim // 2, w_dim // 2], dim=1)

        # Select frequencies for the current height and width
        h_freqs = h_freqs[:height, :]  # [H, h_dim//2, 2]
        w_freqs = w_freqs[:width, :]  # [W, w_dim//2, 2]

        # Broadcast to create the grid
        # Expand to [H, W, dim, 2]
        h_freqs_grid = h_freqs.unsqueeze(1).expand(
            height, width, -1, -1
        )  # [height, width, h_dim//2, 2]
        w_freqs_grid = w_freqs.unsqueeze(0).expand(
            height, width, -1, -1
        )  # [height, width, w_dim//2, 2]

        # Concatenate
        freqs = torch.cat(
            [h_freqs_grid, w_freqs_grid], dim=2
        )  # [height, width, (h_dim+w_dim)//2, 2]

        # Flatten spatial dimensions
        freqs = freqs.flatten(0, 1)  # [height*width, (h_dim+w_dim)//2, 2]

        # Convert back to complex
        freqs = torch.view_as_complex(freqs)

        return freqs


class GatedFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim * 2)
        self.down_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gated, x_linear = self.up_proj(x).chunk(2, dim=-1)
        x = F.silu(x_gated) * x_linear
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout_out(x)


class StandardFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj_in = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj_out(x)
        return self.dropout_out(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_img = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_txt = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm_img = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm_img = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.q_norm_txt = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm_txt = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.rope = MSRoPE(
            theta=rope_theta,
            axes_dim=rope_axes_dim,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
        )

        self.proj_img = nn.Linear(dim, dim)
        self.proj_txt = nn.Linear(dim, dim)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        img_hw: Tuple[int, int],
        txt_seq_len: int,
        txt_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S_img, C = img_tokens.shape
        _, S_txt, _ = txt_tokens.shape

        # Image QKV
        qkv_img = (
            self.qkv_img(img_tokens)
            .reshape(B, S_img, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q_img, k_img, v_img = qkv_img.unbind(0)  # [B, H, S, D]

        # Text QKV
        qkv_txt = (
            self.qkv_txt(txt_tokens)
            .reshape(B, S_txt, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q_txt, k_txt, v_txt = qkv_txt.unbind(0)  # [B, H, S, D]

        # QK Norm
        q_img = self.q_norm_img(q_img)
        k_img = self.k_norm_img(k_img)
        q_txt = self.q_norm_txt(q_txt)
        k_txt = self.k_norm_txt(k_txt)

        # RoPE
        img_freqs, txt_freqs = self.rope(img_hw, txt_seq_len, img_tokens.device)

        # Apply RoPE (need to transpose to [B, S, H, D] for apply_rotary_emb)
        q_img = apply_rotary_emb(q_img.transpose(1, 2), img_freqs).transpose(1, 2)
        k_img = apply_rotary_emb(k_img.transpose(1, 2), img_freqs).transpose(1, 2)

        q_txt = apply_rotary_emb(q_txt.transpose(1, 2), txt_freqs).transpose(1, 2)
        k_txt = apply_rotary_emb(k_txt.transpose(1, 2), txt_freqs).transpose(1, 2)

        # Concat
        q = torch.cat([q_img, q_txt], dim=2)
        k = torch.cat([k_img, k_txt], dim=2)
        v = torch.cat([v_img, v_txt], dim=2)

        # Prepare attention mask
        attn_mask = None
        if txt_attention_mask is not None:
            # Create mask for concatenated sequence [img + txt]
            B, _ = q.shape[0], q.shape[1]

            # Image tokens always have attention
            img_mask = torch.ones(
                B,
                S_img,
                device=txt_attention_mask.device,
                dtype=txt_attention_mask.dtype,
            )
            full_mask = torch.cat([img_mask, txt_attention_mask], dim=1)

            # Convert to attention mask format [B, 1, 1, S_total]
            # Invert mask: 1 (valid) -> False (unmasked), 0 (padding) -> True (masked)
            attn_mask = (full_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(dtype=torch.bool)

        # Attention with mask
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)

        # Split
        x_img = x[:, :, :S_img, :]
        x_txt = x[:, :, S_img:, :]

        # Reshape and Project
        x_img = x_img.transpose(1, 2).reshape(B, S_img, C)
        x_txt = x_txt.transpose(1, 2).reshape(B, S_txt, C)

        x_img = self.proj_img(x_img)
        x_txt = self.proj_txt(x_txt)

        return x_img, x_txt


class DoubleStreamDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        c_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
        modulation_share: str = "none",
        ffn_type: str = "gated",
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.modulation_share = modulation_share

        # Modulation
        if modulation_share == "none":
            # Separate Image/Text, Separate MSA/MLP -> 12 * dim
            self.modulation_img = nn.Sequential(
                nn.SiLU(), nn.Linear(c_dim, 6 * dim, bias=True)
            )
            self.modulation_txt = nn.Sequential(
                nn.SiLU(), nn.Linear(c_dim, 6 * dim, bias=True)
            )
        elif modulation_share == "stream":
            # Share Image/Text, Separate MSA/MLP -> 6 * dim (broadcast across streams)
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(c_dim, 6 * dim, bias=True)
            )
        elif modulation_share == "layer":
            # Separate Image/Text, Share MSA/MLP -> 6 * dim (3 for img, 3 for txt)
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(c_dim, 6 * dim, bias=True)
            )
        elif modulation_share == "all":
            # Share Image/Text, Share MSA/MLP -> 3 * dim (broadcast everything)
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(c_dim, 3 * dim, bias=True)
            )
        else:
            raise ValueError(f"Unknown modulation_share strategy: {modulation_share}")

        # Attention
        self.norm1_img = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_txt = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = DoubleStreamAttention(
            dim,
            num_heads,
            qkv_bias,
            rope_theta,
            rope_axes_dim,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
        )

        # MLP
        self.norm2_img = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if ffn_type == "gated":
            self.mlp_img = GatedFeedForward(dim, mlp_hidden_dim)
            self.mlp_txt = GatedFeedForward(dim, mlp_hidden_dim)
        elif ffn_type == "standard":
            self.mlp_img = StandardFeedForward(dim, mlp_hidden_dim)
            self.mlp_txt = StandardFeedForward(dim, mlp_hidden_dim)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        c: torch.Tensor,
        img_hw: Tuple[int, int],
        txt_seq_len: int,
        txt_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Modulation
        # c: [B, c_dim]
        if self.modulation_share == "none":
            (
                shift_msa_img,
                scale_msa_img,
                gate_msa_img,
                shift_mlp_img,
                scale_mlp_img,
                gate_mlp_img,
            ) = self.modulation_img(c).chunk(6, dim=1)
            (
                shift_msa_txt,
                scale_msa_txt,
                gate_msa_txt,
                shift_mlp_txt,
                scale_mlp_txt,
                gate_mlp_txt,
            ) = self.modulation_txt(c).chunk(6, dim=1)
        elif self.modulation_share == "stream":
            params = self.modulation(c)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                params.chunk(6, dim=1)
            )
            # Broadcast to both streams
            shift_msa_img = shift_msa_txt = shift_msa
            scale_msa_img = scale_msa_txt = scale_msa
            gate_msa_img = gate_msa_txt = gate_msa
            shift_mlp_img = shift_mlp_txt = shift_mlp
            scale_mlp_img = scale_mlp_txt = scale_mlp
            gate_mlp_img = gate_mlp_txt = gate_mlp
        elif self.modulation_share == "layer":
            params = self.modulation(c)
            shift_img, scale_img, gate_img, shift_txt, scale_txt, gate_txt = (
                params.chunk(6, dim=1)
            )
            # Broadcast to MSA and MLP
            shift_msa_img = shift_mlp_img = shift_img
            scale_msa_img = scale_mlp_img = scale_img
            gate_msa_img = gate_mlp_img = gate_img
            shift_msa_txt = shift_mlp_txt = shift_txt
            scale_msa_txt = scale_mlp_txt = scale_txt
            gate_msa_txt = gate_mlp_txt = gate_txt
        elif self.modulation_share == "all":
            params = self.modulation(c)
            shift, scale, gate = params.chunk(3, dim=1)
            # Broadcast to everything
            shift_msa_img = shift_mlp_img = shift_msa_txt = shift_mlp_txt = shift
            scale_msa_img = scale_mlp_img = scale_msa_txt = scale_mlp_txt = scale
            gate_msa_img = gate_mlp_img = gate_msa_txt = gate_mlp_txt = gate

        # 1. Attention Block
        img_norm = modulate(self.norm1_img(img_tokens), shift_msa_img, scale_msa_img)
        txt_norm = modulate(self.norm1_txt(txt_tokens), shift_msa_txt, scale_msa_txt)

        img_attn, txt_attn = self.attn(
            img_norm, txt_norm, img_hw, txt_seq_len, txt_attention_mask
        )

        img_tokens = img_tokens + gate_msa_img.unsqueeze(1) * img_attn
        txt_tokens = txt_tokens + gate_msa_txt.unsqueeze(1) * txt_attn

        # 2. MLP Block
        img_norm = modulate(self.norm2_img(img_tokens), shift_mlp_img, scale_mlp_img)
        txt_norm = modulate(self.norm2_txt(txt_tokens), shift_mlp_txt, scale_mlp_txt)

        img_tokens = img_tokens + gate_mlp_img.unsqueeze(1) * self.mlp_img(img_norm)
        txt_tokens = txt_tokens + gate_mlp_txt.unsqueeze(1) * self.mlp_txt(txt_norm)

        return img_tokens, txt_tokens


class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.rope = MSRoPE(
            theta=rope_theta,
            axes_dim=rope_axes_dim,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # [B, H, S, D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q = apply_rotary_emb(q.transpose(1, 2), freqs).transpose(1, 2)
        k = apply_rotary_emb(k.transpose(1, 2), freqs).transpose(1, 2)

        # Attention
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )

        x = x.transpose(1, 2).reshape(B, S, C)

        return x


class SingleStreamDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        c_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
        ffn_type: str = "gated",
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, 3 * dim, bias=True))

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SingleStreamAttention(
            dim,
            num_heads,
            qkv_bias,
            rope_theta,
            rope_axes_dim,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
        )

        if ffn_type == "gated":
            self.mlp = GatedFeedForward(dim, self.mlp_hidden_dim)
        elif ffn_type == "standard":
            self.mlp = StandardFeedForward(dim, self.mlp_hidden_dim)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.proj_out = nn.Linear(dim + dim, dim)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        c: torch.Tensor,
        img_hw: Tuple[int, int],
        txt_seq_len: int,
        txt_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S_img, C = img_tokens.shape
        _, S_txt, _ = txt_tokens.shape

        x = torch.cat([img_tokens, txt_tokens], dim=1)

        # Prepare RoPE frequencies
        img_freqs, txt_freqs = self.attn.rope(img_hw, txt_seq_len, x.device)
        freqs = torch.cat([img_freqs, txt_freqs], dim=0)

        # Modulation
        shift_msa, scale_msa, gate_msa = self.modulation(c).chunk(3, dim=1)
        norm_x = modulate(self.norm(x), shift_msa, scale_msa)

        # Prepare attention mask
        attn_mask = None
        if txt_attention_mask is not None:
            img_mask = torch.ones(
                B,
                S_img,
                device=txt_attention_mask.device,
                dtype=txt_attention_mask.dtype,
            )
            full_mask = torch.cat([img_mask, txt_attention_mask], dim=1)
            attn_mask = (full_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(dtype=torch.bool)

        # Attention
        attn_out = self.attn(norm_x, freqs, attn_mask)

        # MLP
        mlp_out = self.mlp(norm_x)

        # Combine
        hidden = torch.cat([attn_out, mlp_out], dim=-1)
        out = gate_msa.unsqueeze(1) * self.proj_out(hidden)
        x = x + out

        img_tokens = x[:, :S_img, :]
        txt_tokens = x[:, S_img:, :]

        return img_tokens, txt_tokens


class UnconditionalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.rope = MSRoPE(
            theta=rope_theta,
            axes_dim=rope_axes_dim,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
        B, S, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # [B, H, S, D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Calculate frequencies for RoPE
        # We don't have text here, so we just use a dummy 0 for text length
        img_freqs, _ = self.rope(img_hw, 0, x.device)

        # Apply RoPE
        q = apply_rotary_emb(q.transpose(1, 2), img_freqs).transpose(1, 2)
        k = apply_rotary_emb(k.transpose(1, 2), img_freqs).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)

        return x


class UnconditionalDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        c_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Modulation: 6 parameters (shift, scale, gate for both attn and mlp)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, 6 * dim, bias=True))

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = UnconditionalAttention(
            dim,
            num_heads,
            qkv_bias,
            rope_theta,
            rope_axes_dim,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GatedFeedForward(dim, mlp_hidden_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, img_hw: Tuple[int, int]
    ) -> torch.Tensor:
        # c: [B, c_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(c).chunk(6, dim=1)
        )

        # 1. Attention Block
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(norm_x, img_hw)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. MLP Block
        norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    def get_params_flops(model, inputs):
        params = sum(p.numel() for p in model.parameters())
        flops = FlopCountAnalysis(model, inputs)
        return params, flops.total()

    print("\n" + "=" * 60)
    print("DiT Block Analysis")
    print("=" * 60)

    B, S, C = 1, 256, 1024
    num_heads = 16
    c_dim = 256
    img_hw = (16, 16)
    txt_seq_len = 77
    rope_axes_dim = [32, 32]  # sums to 64 = head_dim

    # 1. Unconditional DiT Block
    block_uncond = UnconditionalDiTBlock(
        dim=C, num_heads=num_heads, c_dim=c_dim, rope_axes_dim=rope_axes_dim
    )
    x = torch.randn(B, S, C)
    c = torch.randn(B, c_dim)
    inputs_uncond = (x, c, img_hw)
    params_uncond, flops_uncond = get_params_flops(block_uncond, inputs_uncond)

    # 2. Single Stream DiT Block
    block_single = SingleStreamDiTBlock(
        dim=C, num_heads=num_heads, c_dim=c_dim, rope_axes_dim=rope_axes_dim
    )
    img_tokens = torch.randn(B, S, C)
    txt_tokens = torch.randn(B, txt_seq_len, C)
    inputs_single = (img_tokens, txt_tokens, c, img_hw, txt_seq_len)
    params_single, flops_single = get_params_flops(block_single, inputs_single)

    # 3. Double Stream DiT Block
    block_double = DoubleStreamDiTBlock(
        dim=C, num_heads=num_heads, c_dim=c_dim, rope_axes_dim=rope_axes_dim
    )
    inputs_double = (img_tokens, txt_tokens, c, img_hw, txt_seq_len)
    params_double, flops_double = get_params_flops(block_double, inputs_double)

    print(f"{'Model':<25} | {'Params':<15} | {'FLOPs':<15}")
    print("-" * 60)
    print(f"{'Unconditional':<25} | {params_uncond:<15,d} | {flops_uncond:<15.2e}")
    print(f"{'Single Stream':<25} | {params_single:<15,d} | {flops_single:<15.2e}")
    print(f"{'Double Stream':<25} | {params_double:<15,d} | {flops_double:<15.2e}")
    print("-" * 60)
