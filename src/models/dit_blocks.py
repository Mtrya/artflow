"""Attention, positional features and native ArtFlow transformer blocks."""

from typing import Tuple, Optional, Sequence
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .varlen_attention import packed_attention

# Optional SDPA backend restriction for the masked (padded-text) attention
# path; None keeps the library default.
_SDPA_BACKENDS = None


class RMSNorm(nn.RMSNorm):
    """Fuse Ascend normalization while retaining FP32 arithmetic and gains."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.device.type != "npu"
            or self.weight is None
            or len(self.normalized_shape) != 1
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return super().forward(x)
        import torch_npu

        # PyTorch's mixed-dtype RMSNorm promotes the input to FP32 and applies
        # the learned gain before casting back. Casting the gain to BF16 just
        # to use a fused kernel would change that computation.
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        y, _ = torch_npu.npu_rms_norm(
            x.float(), self.weight.float(), epsilon=eps
        )
        return y.to(x.dtype)


def sdpa_with_pad_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """SDPA with an optional bool key-padding mask of shape [B, 1, 1, S_k].

    A bool mask forces PyTorch off the flash path; the math fallback then
    materializes B*H*S^2 attention weights per layer (saved for backward),
    which at seq ~2K is tens of GiB per layer and OOMs long-caption batches.
    We convert the mask to an additive bias in q.dtype
    and prefer the memory-efficient kernel (never materializes S^2 scores);
    mask=None keeps the default (flash) path untouched.
    """
    if attn_mask is None:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    return sdpa_with_bias(q, k, v, pad_bias_from_mask(attn_mask, q.dtype))


def set_sdpa_backends(names: Optional[Sequence[str]]) -> None:
    """Restrict SDPA backends for masked (padded-text) attention.

    ``None`` keeps the library default (memory-efficient first). Otherwise
    masked attention is limited to the named ``SDPBackend`` members, with the
    math backend appended as a fallback. Training keeps the default: on a
    single 4090 at 256p (micro-batch 16, 128-192 text tokens) cuDNN attention
    is ~30% slower than memory-efficient on the forward+backward training
    step, despite a faster forward-only kernel with an additive pad bias.
    """
    global _SDPA_BACKENDS
    if names is None:
        _SDPA_BACKENDS = None
        return
    from torch.nn.attention import SDPBackend

    resolved = []
    for name in names:
        member = getattr(SDPBackend, name.strip().upper(), None)
        if member is None:
            raise ValueError(f"unknown SDPA backend {name!r}")
        resolved.append(member)
    _SDPA_BACKENDS = tuple(resolved)


def _sdpa_backends():
    if _SDPA_BACKENDS is not None:
        return list(_SDPA_BACKENDS)
    from torch.nn.attention import SDPBackend

    return [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def sdpa_with_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """SDPA with a precomputed additive bias of shape [B, 1, 1, S_k]."""
    if q.device.type == "npu":
        # torch_npu's F.sdpa dispatch with a float additive bias materializes
        # per-head S^2 bias + P matrix per layer (measured: +2.8 GiB at
        # B8/S2304), which OOMs long-caption batches. npu_fusion_attention with
        # a bool (B,1,S,S) drop-mask (True = drop) is true flash attention:
        # +0.06 GiB saved for backward, maxdiff 2e-3 vs the float-bias path at
        # S=2304 bf16. Two gotchas, both measured: the scale argument defaults
        # to 1.0 (NOT 1/sqrt(D)) so it must be passed explicitly, and the mask
        # must be full (B,1,S,S) - aclnn rejects broadcastable (B,1,1,S).
        import torch_npu

        b, h, s_q, d = q.shape
        s_k = k.shape[2]
        # bias == -inf: identical to torch.isneginf for float tensors, but
        # aten.eq.Scalar has a torchair GE converter while aten.isneginf does
        # not (ascend-ta-bench3 NotImplementedError).
        drop = bias == float("-inf")
        mask = drop.expand(b, 1, s_q, s_k).contiguous()
        return torch_npu.npu_fusion_attention(
            q, k, v, h, "BNSD", atten_mask=mask, keep_prob=1.0, scale=d**-0.5
        )[0]
    from torch.nn.attention import sdpa_kernel

    backends = _sdpa_backends()
    if q.device.type != "cuda":
        from torch.nn.attention import SDPBackend

        backends = [SDPBackend.MATH]
    with sdpa_kernel(backends):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)


def pad_bias_from_mask(attn_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a bool keep-mask [B, 1, 1, S_k] into an additive SDPA bias."""
    bias = torch.zeros(attn_mask.shape, dtype=dtype, device=attn_mask.device)
    return bias.masked_fill(~attn_mask, float("-inf"))


class TimestepEmbeddings(nn.Module):
    """Sinusoidal features; scaling applies here, never to the flow path."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("factor", torch.tensor(1000, dtype=torch.int64))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        factor = state_dict.get(prefix + "factor")
        if factor is None or factor.item() != 1000:
            error_msgs.append(f"{prefix}expected timestep feature factor 1000")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,)
        Returns:
            embedding: (B, N)
        """
        half = self.hidden_size // 2
        exponent = (
            -math.log(10000)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )

        emb = torch.exp(exponent)
        times = timesteps.float() * 1000
        emb = times.unsqueeze(1) * emb.unsqueeze(0)

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


def apply_rotary_emb_real(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Equivalent FP32 rotation expressed without complex multiplication.

    This lets Inductor fuse surrounding casts/layout operations. Compiled
    rounding may differ; the complex implementation remains the default.
    """
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs.unbind(-1)
    cosine, sine = torch.view_as_real(freqs_cis)[None, :, None].unbind(-1)
    rotated = torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), dim=-1
    )
    return rotated.flatten(3).type_as(x)


def apply_rotary_emb_realfreq(
    x: torch.Tensor, freqs_real: torch.Tensor
) -> torch.Tensor:
    """apply_rotary_emb_real taking an already-real frequency tensor.

    freqs_real is [S, D/2, 2] float32 (cos, sin pairs) — the view_as_real of
    the complex table, materialized OUTSIDE compiled regions so no
    complex-typed tensor crosses the graph boundary (torchair/GE has no
    complex dtype support).
    """
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs.unbind(-1)
    cosine, sine = freqs_real[None, :, None].unbind(-1)
    rotated = torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), dim=-1
    )
    return rotated.flatten(3).type_as(x)


def set_real_rope(model: nn.Module, enabled: bool) -> None:
    """Set an instance-local execution policy before compiling the model.

    This is not checkpoint state: loading weights must not override the launch
    policy. Other live models and the complex frequency tables are untouched.
    """
    if type(enabled) is not bool:
        raise ValueError("real RoPE policy must be boolean")
    for module in model.modules():
        if isinstance(module, (DoubleStreamAttention, SingleStreamAttention)):
            module.real_rope = enabled


class MSRoPE(nn.Module):
    """Multimodal Scalable RoPE for 2D images and text"""

    def __init__(
        self,
        theta: int = 10000,
        axes_dim: list = [64, 64],
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
        centered: bool = False,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        # centered: image coords symmetric around 0 ([-(n-1)/2, (n-1)/2]) instead of
        # 0-based from the corner; text stays pinned to a fixed diagonal beyond the
        # image range in both modes.
        self.centered = centered

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
        # View as complex for slicing (must be float32 — view_as_complex doesn't support bf16)
        pos_freqs_complex = torch.view_as_complex(self.pos_freqs.float())
        txt_freqs = pos_freqs_complex[
            max_img_pos : max_img_pos + txt_seq_len, :
        ]  # placing text tokens on a diagonal in the 2D position space

        return img_freqs, txt_freqs

    def prepare_freqs(
        self, img_hw: Tuple[int, int], txt_seq_len: int, device
    ) -> torch.Tensor:
        """Concatenated [S_img + S_txt, D/2] complex freqs, reusable across blocks.

        Identical to applying ``forward()`` and concatenating the two halves;
        hoisting it out of the block loop removes one table lookup and two
        concatenations per layer.
        """
        img_freqs, txt_freqs = self.forward(img_hw, txt_seq_len, device)
        return torch.cat([img_freqs, txt_freqs], dim=0)

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
        # pos_freqs is [S, D, 2] (real) — cast to float32 for view_as_complex
        h_dim, w_dim = self.axes_dim

        if self.centered:
            # Symmetric (possibly half-integer) coordinates around the origin.
            h_idx = torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0
            w_idx = torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0
            # rope_params returns complex [N, dim//2]; store as real pairs
            h_freqs = torch.view_as_real(self.rope_params(h_idx, h_dim))
            w_freqs = torch.view_as_real(self.rope_params(w_idx, w_dim))
        else:
            pos_freqs = self.pos_freqs.float()
            h_freqs, w_freqs = pos_freqs.split([h_dim // 2, w_dim // 2], dim=1)

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
        x = self.up_proj(x)
        if x.device.type == "npu":
            import torch_npu

            x = torch_npu.npu_swiglu(x, dim=-1)
        else:
            x_gated, x_linear = x.chunk(2, dim=-1)
            x = F.silu(x_gated) * x_linear
        x = self.dropout(x)
        x = self.down_proj(x)
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
        rope_centered: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.real_rope = False

        self.qkv_img = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_txt = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm_img = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm_img = RMSNorm(self.head_dim, eps=1e-6)
        self.q_norm_txt = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm_txt = RMSNorm(self.head_dim, eps=1e-6)

        self.rope = MSRoPE(
            theta=rope_theta,
            axes_dim=rope_axes_dim,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
            centered=rope_centered,
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
        attn_metadata=None,
        rope_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
        img_freqs, txt_freqs = (
            self.rope(img_hw, txt_seq_len, img_tokens.device)
            if rope_freqs is None
            else rope_freqs
        )

        # Apply RoPE (need to transpose to [B, S, H, D] for apply_rotary_emb)
        rotate = apply_rotary_emb_real if self.real_rope else apply_rotary_emb
        q_img = rotate(q_img.transpose(1, 2), img_freqs).transpose(1, 2)
        k_img = rotate(k_img.transpose(1, 2), img_freqs).transpose(1, 2)

        q_txt = rotate(q_txt.transpose(1, 2), txt_freqs).transpose(1, 2)
        k_txt = rotate(k_txt.transpose(1, 2), txt_freqs).transpose(1, 2)

        # Concat
        q = torch.cat([q_img, q_txt], dim=2)
        k = torch.cat([k_img, k_txt], dim=2)
        v = torch.cat([v_img, v_txt], dim=2)

        # Prepare attention mask
        attn_mask = None
        if txt_attention_mask is not None and attn_metadata is None:
            # Image tokens always have attention
            img_mask = torch.ones(
                B,
                S_img,
                device=txt_attention_mask.device,
                dtype=txt_attention_mask.dtype,
            )
            full_mask = torch.cat([img_mask, txt_attention_mask], dim=1)

            # Convert to attention mask format [B, 1, 1, S_total]
            # Mask: 1 (valid) -> True (attend), 0 (padding) -> False (ignore)
            attn_mask = (full_mask > 0).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(dtype=torch.bool)

        # Attention with mask
        x = (
            packed_attention(q, k, v, attn_metadata)
            if attn_metadata is not None
            else sdpa_with_pad_mask(q, k, v, attn_mask)
        )

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
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Normalize branch outputs before learned residual gates.
        self.norm_msa_img_out = RMSNorm(dim, eps=1e-6)
        self.norm_msa_txt_out = RMSNorm(dim, eps=1e-6)
        self.norm_mlp_img_out = RMSNorm(dim, eps=1e-6)
        self.norm_mlp_txt_out = RMSNorm(dim, eps=1e-6)

        # Modulation
        self.modulation_img = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 6 * dim, bias=True)
        )
        self.modulation_txt = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 6 * dim, bias=True)
        )

        # Attention
        self.norm1_img = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_txt = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = DoubleStreamAttention(
            dim,
            num_heads,
            True,
            rope_theta,
            rope_axes_dim,
            rope_scaling_type="none",
            rope_scaling_factor=1.0,
            rope_centered=True,
        )

        # MLP
        self.norm2_img = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_img = GatedFeedForward(dim, mlp_hidden_dim)
        self.mlp_txt = GatedFeedForward(dim, mlp_hidden_dim)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        c: torch.Tensor,
        img_hw: Tuple[int, int],
        txt_seq_len: int,
        txt_attention_mask: Optional[torch.Tensor] = None,
        attn_metadata=None,
        rope_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Modulation
        # c: [B, c_dim]
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

        # 1. Attention Block
        img_norm = modulate(self.norm1_img(img_tokens), shift_msa_img, scale_msa_img)
        txt_norm = modulate(self.norm1_txt(txt_tokens), shift_msa_txt, scale_msa_txt)

        img_attn, txt_attn = self.attn(
            img_norm,
            txt_norm,
            img_hw,
            txt_seq_len,
            txt_attention_mask,
            attn_metadata=attn_metadata,
            rope_freqs=rope_freqs,
        )

        img_attn = self.norm_msa_img_out(img_attn)
        txt_attn = self.norm_msa_txt_out(txt_attn)
        img_tokens = img_tokens + gate_msa_img.unsqueeze(1) * img_attn
        txt_tokens = txt_tokens + gate_msa_txt.unsqueeze(1) * txt_attn

        # 2. MLP Block
        img_norm = modulate(self.norm2_img(img_tokens), shift_mlp_img, scale_mlp_img)
        txt_norm = modulate(self.norm2_txt(txt_tokens), shift_mlp_txt, scale_mlp_txt)

        mlp_img_out = self.mlp_img(img_norm)
        mlp_txt_out = self.mlp_txt(txt_norm)
        mlp_img_out = self.norm_mlp_img_out(mlp_img_out)
        mlp_txt_out = self.norm_mlp_txt_out(mlp_txt_out)
        img_tokens = img_tokens + gate_mlp_img.unsqueeze(1) * mlp_img_out
        txt_tokens = txt_tokens + gate_mlp_txt.unsqueeze(1) * mlp_txt_out

        return img_tokens, txt_tokens

    def initialize_weights(self):
        # AdaLN-zero: Initialize modulation MLP to zero
        nn.init.constant_(self.modulation_img[-1].weight, 0)
        nn.init.constant_(self.modulation_img[-1].bias, 0)
        nn.init.constant_(self.modulation_txt[-1].weight, 0)
        nn.init.constant_(self.modulation_txt[-1].bias, 0)


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
        rope_centered: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.real_rope = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.rope = MSRoPE(
            theta=rope_theta,
            axes_dim=rope_axes_dim,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
            centered=rope_centered,
        )

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        img_hw: Tuple[int, int],
        txt_seq_len: int,
        txt_attention_mask: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attn_metadata=None,
    ) -> torch.Tensor:
        B, S, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_freqs is not None:
            # Frequencies already cover [img, txt] in that order, so RoPE is a
            # single elementwise pass over the concatenated sequence. Under
            # the real-rope policy the hoisted table arrives in real
            # [S, D/2, 2] layout (converted once at model level, outside any
            # compiled block); otherwise it is complex.
            rotate = apply_rotary_emb_realfreq if self.real_rope else apply_rotary_emb
            q = rotate(q.transpose(1, 2), rope_freqs).transpose(1, 2)
            k = rotate(k.transpose(1, 2), rope_freqs).transpose(1, 2)
        else:
            rotate = apply_rotary_emb_real if self.real_rope else apply_rotary_emb
            # Need to apply different RoPE to img and txt parts
            # Assuming x is [img, txt]
            S_img = img_hw[0] * img_hw[1]

            q_img = q[:, :, :S_img, :]
            k_img = k[:, :, :S_img, :]
            q_txt = q[:, :, S_img:, :]
            k_txt = k[:, :, S_img:, :]

            img_freqs, txt_freqs = self.rope(img_hw, txt_seq_len, x.device)

            q_img = rotate(q_img.transpose(1, 2), img_freqs).transpose(1, 2)
            k_img = rotate(k_img.transpose(1, 2), img_freqs).transpose(1, 2)
            q_txt = rotate(q_txt.transpose(1, 2), txt_freqs).transpose(1, 2)
            k_txt = rotate(k_txt.transpose(1, 2), txt_freqs).transpose(1, 2)

            q = torch.cat([q_img, q_txt], dim=2)
            k = torch.cat([k_img, k_txt], dim=2)

        if attn_metadata is not None:
            x = packed_attention(q, k, v, attn_metadata)
            return self.proj(x.transpose(1, 2).reshape(B, S, C))

        if attn_bias is not None:
            x = sdpa_with_bias(q, k, v, attn_bias)
            x = x.transpose(1, 2).reshape(B, S, C)
            return self.proj(x)

        # Prepare attention mask
        attn_mask = None
        if txt_attention_mask is not None:
            # Image tokens always have attention
            img_mask = torch.ones(
                B,
                S_img,
                device=txt_attention_mask.device,
                dtype=txt_attention_mask.dtype,
            )
            full_mask = torch.cat([img_mask, txt_attention_mask], dim=1)

            # Convert to attention mask format [B, 1, 1, S]
            # Mask : 1 (valid) -> True (attend), 0 (padding) -> False (ignore)
            attn_mask = (full_mask > 0).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(dtype=torch.bool)

        # Attention with mask
        x = sdpa_with_pad_mask(q, k, v, attn_mask)

        x = x.transpose(1, 2).reshape(B, S, C)

        x = self.proj(x)

        return x


class SingleStreamDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        c_dim: int,
        mlp_ratio: float = 4.0,
        rope_theta: int = 10000,
        rope_axes_dim: list = [64, 64],
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Modulation
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, 3 * dim, bias=True))

        # Normalize branch outputs before the residual add. Learned gates and
        # affine norm gains remain outside this control of branch amplitude.
        self.norm_msa_out = RMSNorm(dim, eps=1e-6)
        self.norm_mlp_out = RMSNorm(dim, eps=1e-6)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SingleStreamAttention(
            dim,
            num_heads,
            True,
            rope_theta,
            rope_axes_dim,
            rope_scaling_type="none",
            rope_scaling_factor=1.0,
            rope_centered=True,
        )

        # MLP
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = GatedFeedForward(dim, mlp_hidden_dim)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        c: torch.Tensor,
        img_hw: Tuple[int, int],
        txt_seq_len: int,
        txt_attention_mask: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attn_metadata=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Modulation
        # c: [B, c_dim]
        params = self.modulation(c)
        shift, scale, gate = params.chunk(3, dim=1)
        shift_msa = shift_mlp = shift
        scale_msa = scale_mlp = scale
        gate_msa = gate_mlp = gate

        # Concatenate for shared MSA/MLP processing
        S_img = img_tokens.shape[1]
        S_txt = txt_tokens.shape[1]
        x = torch.cat([img_tokens, txt_tokens], dim=1)

        # 1. Attention Block
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        x_attn = self.attn(
            x_norm,
            img_hw,
            txt_seq_len,
            txt_attention_mask,
            rope_freqs=rope_freqs,
            attn_bias=attn_bias,
            attn_metadata=attn_metadata,
        )

        x_attn = self.norm_msa_out(x_attn)
        x = x + gate_msa.unsqueeze(1) * x_attn

        # 2. MLP Block
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        mlp_out = self.norm_mlp_out(mlp_out)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        img_tokens, txt_tokens = x.split([S_img, S_txt], dim=1)

        return img_tokens, txt_tokens

    def initialize_weights(self):
        # AdaLN-zero: Initialize modulation MLP to zero
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)
