"""
Unified ArtFlow DiT model for Stage 1 Architecture Ablation.
Supports configurable conditioning (Pure/Fused), block schedules (Double/Single stream),
and modulation strategies.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .dit_blocks import (
        DoubleStreamDiTBlock,
        SingleStreamDiTBlock,
        TimestepEmbeddings,
    )
except ImportError:
    from dit_blocks import (
        DoubleStreamDiTBlock,
        SingleStreamDiTBlock,
        TimestepEmbeddings,
    )


class ArtFlow(nn.Module):
    """
    Unified ArtFlow model.

    Configuration:
    - conditioning_scheme: "pure" (timestep only) or "fused" (timestep + pooled text)
    - double_stream_depth: Number of double-stream blocks (at the start)
    - single_stream_depth: Number of single-stream blocks (at the end)
    - modulation_share: Strategy for DoubleStream blocks ("none", "stream", "layer", "all")
    - qkv_bias: Whether to use bias in QKV projections
    - ffn_type: "gated" or "standard"
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        txt_in_features: int = 2048,
        # Configuration
        hidden_size: int = 1152,
        num_heads: int = 16,
        double_stream_depth: int = 28,
        single_stream_depth: int = 0,
        mlp_ratio: float = 2.67,
        conditioning_scheme: str = "pure",
        qkv_bias: bool = True,
        double_stream_modulation: str = "none",
        single_stream_modulation: str = "none",
        ffn_type: str = "gated",
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.conditioning_scheme = conditioning_scheme

        # 1. Input Embeddings
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.txt_embedder = nn.Linear(txt_in_features, hidden_size)

        # 2. Conditioning (Timestep + Optional Text)
        self.t_embedder = TimestepEmbeddings(hidden_size)

        if conditioning_scheme == "pure":
            # MLP(t)
            self.c_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        elif conditioning_scheme == "fused":
            # MLP(cat(t, txt_pooled))
            self.txt_pooled_proj = nn.Linear(txt_in_features, hidden_size)
            self.c_mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown conditioning_scheme: {conditioning_scheme}")

        # 3. Blocks
        head_dim = hidden_size // num_heads
        assert head_dim % 2 == 0, "Head dimension must be divisible by 2 for RoPE"
        rope_axes_dim = [head_dim // 2, head_dim // 2]

        self.blocks = nn.ModuleList()

        # Add Double Stream Blocks
        for _ in range(double_stream_depth):
            self.blocks.append(
                DoubleStreamDiTBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    c_dim=hidden_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    rope_axes_dim=rope_axes_dim,
                    modulation_share=double_stream_modulation,
                    ffn_type=ffn_type,
                    rope_scaling_type=rope_scaling_type,
                    rope_scaling_factor=rope_scaling_factor,
                )
            )

        # Add Single Stream Blocks
        for _ in range(single_stream_depth):
            self.blocks.append(
                SingleStreamDiTBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    c_dim=hidden_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    rope_axes_dim=rope_axes_dim,
                    modulation_share=single_stream_modulation,
                    ffn_type=ffn_type,
                    rope_scaling_type=rope_scaling_type,
                    rope_scaling_factor=rope_scaling_factor,
                )
            )

        # 4. Final Layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(
                hidden_size, patch_size * patch_size * self.out_channels, bias=True
            ),
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize blocks (AdaLN-zero)
        for block in self.blocks:
            if hasattr(block, "initialize_weights"):
                block.initialize_weights()

        # Zero-out output layer
        nn.init.constant_(self.final_layer[1].weight, 0)
        nn.init.constant_(self.final_layer[1].bias, 0)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        txt: torch.Tensor,
        txt_pooled: Optional[torch.Tensor] = None,
        txt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (N, C, H, W)
        t: (N,)
        txt: (N, L_txt, D_txt)
        txt_pooled: (N, D_txt) - Required if conditioning_scheme="fused"
        txt_mask: (N, L_txt)
        """
        _, _, H, W = x.shape
        x = self.x_embedder(x)  # (N, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (N, L_img, D)

        txt = self.txt_embedder(txt)  # (N, L_txt, D)

        t_emb = self.t_embedder(t)  # (N, D)

        if self.conditioning_scheme == "pure":
            c = self.c_mlp(t_emb)
        else:  # fused
            if txt_pooled is None:
                raise ValueError("txt_pooled is required for fused conditioning")
            txt_pooled_emb = self.txt_pooled_proj(txt_pooled)
            c = self.c_mlp(torch.cat([t_emb, txt_pooled_emb], dim=1))

        img_hw = (H // self.patch_size, W // self.patch_size)
        txt_seq_len = txt.shape[1]

        for block in self.blocks:
            x, txt = block(x, txt, c, img_hw, txt_seq_len, txt_mask)

        x = self.final_layer(x)
        x = self.unpatchify(x, H, W)

        return x


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    def test_variant(name, config):
        print(f"\nTesting {name}...")
        try:
            model = ArtFlow(**config)
            x = torch.randn(1, 16, 64, 64)  # Smaller size for quick test
            t = torch.randint(0, 1000, (1,))
            txt = torch.randn(1, 77, 2048)
            txt_pooled = (
                torch.randn(1, 2048)
                if config.get("conditioning_scheme") == "fused"
                else None
            )

            inputs = (x, t, txt, txt_pooled)

            params = sum(p.numel() for p in model.parameters())
            #flops = FlopCountAnalysis(model, inputs).total()

            print(f"Params: {params:,}")
            #print(f"FLOPs: {flops:.2e}")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False

    base_config = {
        "hidden_size": 640,
        "num_heads": 8,
        "double_stream_depth": 2,
        "single_stream_depth": 14,
        "mlp_ratio": 2.67,
        "conditioning_scheme": "pure",
        "qkv_bias": False,
        "double_stream_modulation": "layer",
        "single_stream_modulation": "layer",
        "ffn_type": "gated",
    }
    test_variant("Base", base_config)

    config = {
        "hidden_size": 1120,
        "num_heads": 14,
        "double_stream_depth": 0,
        "single_stream_depth": 28,
        "mlp_ratio": 2.667,
        "conditioning_scheme": "pure",
        "double_stream_modulation": "layer",
        "single_stream_modulation": "none",
        "ffn_type": "gated"
    }
    test_variant("Test", config)
