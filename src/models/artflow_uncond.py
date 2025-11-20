"""
Unconditional DiT model for ArtFlow (Stage 0: Algorithm Ablation).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    from .dit_blocks import UnconditionalDiTBlock, TimestepEmbeddings
except ImportError:
    from dit_blocks import UnconditionalDiTBlock, TimestepEmbeddings

class ArtFlowUncond(nn.Module):
    """
    Unconditional DiT model for ArtFlow.
    Uses RoPE for spatial information, allowing variable resolution.
    """
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 1. Input Embeddings
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # 2. Timestep Embeddings
        self.t_embedder = TimestepEmbeddings(hidden_size)
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # 3. Blocks
        head_dim = hidden_size // num_heads
        assert head_dim % 2 == 0, "Head dimension must be divisible by 2 for RoPE"
        rope_axes_dim = [head_dim // 2, head_dim // 2]
        
        self.blocks = nn.ModuleList([
            UnconditionalDiTBlock(hidden_size, num_heads, c_dim=hidden_size, mlp_ratio=mlp_ratio, rope_axes_dim=rope_axes_dim)
            for _ in range(depth)
        ])
        
        # 4. Final Layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        
        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        """
        _, _, H, W = x.shape
        x = self.x_embedder(x) # (N, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2) # (N, L, D)
        
        t = self.t_embedder(t) # (N, D)
        c = self.t_mlp(t)      # (N, D)
        
        img_hw = (H // self.patch_size, W // self.patch_size)
        
        for block in self.blocks:
            x = block(x, c, img_hw)
            
        x = self.final_layer(x) # (N, L, p*p*C)
        x = self.unpatchify(x, H, W)  # (N, C, H, W)
        
        return x

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    
    model = ArtFlowUncond(hidden_size=512, depth=8, num_heads=8)
    x = torch.randn(1, 16, 128, 128)
    t = torch.randint(0, 1000, (1,))
    
    flops = FlopCountAnalysis(model, (x, t)).total()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FLOPs: {flops:.2e}")
