"""
Hybrid Double/Single-Stream DiT model for ArtFlow (Stage 1: Architecture Ablation).
Conditioning: MLP(concat(t, text_pooled)).
Architecture: N DoubleStream blocks + M SingleStream blocks.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    from .dit_blocks import DoubleStreamDiTBlock, SingleStreamDiTBlock, TimestepEmbeddings
except ImportError:
    from dit_blocks import DoubleStreamDiTBlock, SingleStreamDiTBlock, TimestepEmbeddings

class ArtFlowHybrid(nn.Module):
    """
    Hybrid DiT with fused conditioning.
    Uses a stack of DoubleStream blocks followed by SingleStream blocks.
    """
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1120,
        double_depth: int = 6,
        single_depth: int = 12,
        num_heads: int = 14,
        double_mlp_ratio: float = 2.67,
        single_mlp_ratio: float = 4.0,
        txt_in_features: int = 2048,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 1. Input Embeddings
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.txt_embedder = nn.Linear(txt_in_features, hidden_size)
        
        # 2. Timestep + Text Embeddings
        self.t_embedder = TimestepEmbeddings(hidden_size)
        self.c_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # 3. Blocks
        head_dim = hidden_size // num_heads
        assert head_dim % 2 == 0, "Head dimension must be divisible by 2 for RoPE"
        rope_axes_dim = [head_dim // 2, head_dim // 2]

        self.double_blocks = nn.ModuleList([
            DoubleStreamDiTBlock(hidden_size, num_heads, c_dim=hidden_size, mlp_ratio=double_mlp_ratio, rope_axes_dim=rope_axes_dim)
            for _ in range(double_depth)
        ])
        
        self.single_blocks = nn.ModuleList([
            SingleStreamDiTBlock(hidden_size, num_heads, c_dim=hidden_size, mlp_ratio=single_mlp_ratio, rope_axes_dim=rope_axes_dim)
            for _ in range(single_depth)
        ])
        
        # 4. Final Layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, txt: torch.Tensor, txt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (N, C, H, W)
        t: (N,)
        txt: (N, L_txt, D_txt)
        """
        _, _, H, W = x.shape
        x = self.x_embedder(x) # (N, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2) # (N, L_img, D)
        
        txt = self.txt_embedder(txt) # (N, L_txt, D)
        
        t_emb = self.t_embedder(t) # (N, D)
        c = self.c_mlp(t_emb) # (N, D)
        
        img_hw = (H // self.patch_size, W // self.patch_size)
        txt_seq_len = txt.shape[1]
        
        # Double Stream Blocks
        for block in self.double_blocks:
            x, txt = block(x, txt, c, img_hw, txt_seq_len, txt_mask)
            
        # Single Stream Blocks
        # Concatenate happens inside SingleStreamDiTBlock, but we need to pass them separately
        for block in self.single_blocks:
            x, txt = block(x, txt, c, img_hw, txt_seq_len, txt_mask)
            
        x = self.final_layer(x)
        x = self.unpatchify(x, H, W)
        
        return x

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    
    model = ArtFlowHybrid(hidden_size=1120, double_depth=6, single_depth=12, num_heads=14, double_mlp_ratio=2.8, txt_in_features=2048)
    x = torch.randn(1, 16, 256, 256)
    t = torch.randint(0, 1000, (1,))
    txt = torch.randn(1, 77, 2048)
    
    flops = FlopCountAnalysis(model, (x, t, txt)).total()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FLOPs: {flops:.2e}")
