"""ArtFlow: fused text/time conditioning with normalized residual branches."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from .dit_blocks import (
    DoubleStreamDiTBlock,
    SingleStreamDiTBlock,
    TimestepEmbeddings,
    pad_bias_from_mask,
)


class ArtFlow(nn.Module, PyTorchModelHubMixin):
    """Versioned architecture; only capacity is configurable.

    The Qwen image VAE supplies 16 latent channels, patched 2×2. Text features
    have width 1024. Pooled text and sinusoidal time features jointly condition
    gated, branch-normalized blocks. Double-stream blocks use independent
    attention/MLP modulation; single-stream blocks share it within each layer.
    """

    ARCHITECTURE = "artflow-v2"

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        double_stream_depth: int,
        single_stream_depth: int,
        mlp_ratio: float,
        architecture: str = ARCHITECTURE,
    ):
        super().__init__()
        if architecture != self.ARCHITECTURE:
            raise ValueError(
                f"unsupported architecture {architecture!r}; use its original source revision"
            )
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % (4 * num_heads):
            raise ValueError("hidden_size must be a positive multiple of 4 * num_heads")
        if (
            min(double_stream_depth, single_stream_depth) < 0
            or double_stream_depth + single_stream_depth < 1
        ):
            raise ValueError("at least one transformer block is required")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.patch_size = 2
        self.in_channels = self.out_channels = 16
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.double_stream_depth = double_stream_depth
        self.single_stream_depth = single_stream_depth
        self.mlp_ratio = mlp_ratio
        self.x_embedder = nn.Conv2d(16, hidden_size, kernel_size=2, stride=2)
        self.txt_embedder = nn.Linear(1024, hidden_size)
        self.t_embedder = TimestepEmbeddings(hidden_size)
        self.txt_pooled_proj = nn.Linear(1024, hidden_size)
        self.c_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        head_dim = hidden_size // num_heads
        self.blocks = nn.ModuleList(
            block(
                dim=hidden_size,
                num_heads=num_heads,
                c_dim=hidden_size,
                mlp_ratio=mlp_ratio,
                rope_axes_dim=[head_dim // 2, head_dim // 2],
            )
            for block in (
                [DoubleStreamDiTBlock] * double_stream_depth
                + [SingleStreamDiTBlock] * single_stream_depth
            )
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, 2 * 2 * 16, bias=True),
        )
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)
        for block in self.blocks:
            block.initialize_weights()
        nn.init.zeros_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x.reshape(x.shape[0], h // 2, w // 2, 2, 2, 16)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], 16, h, w)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        txt: torch.Tensor,
        txt_pooled: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict latent velocity from normalized time t in [0, 1]."""
        _, _, height, width = x.shape
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        txt = self.txt_embedder(txt)
        c = self.c_mlp(
            torch.cat([self.t_embedder(t), self.txt_pooled_proj(txt_pooled)], dim=1)
        )
        img_hw = (height // 2, width // 2)
        txt_seq_len = txt.shape[1]

        # Single-stream layers share geometry and padding. Build these tensors
        # once per forward, keeping their math identical to the per-layer path.
        rope_freqs = attn_bias = None
        if self.single_stream_depth:
            first_single = self.blocks[self.double_stream_depth]
            rope_freqs = first_single.attn.rope.prepare_freqs(
                img_hw, txt_seq_len, x.device
            )
            if first_single.attn.real_rope:
                rope_freqs = torch.view_as_real(rope_freqs)
            if txt_mask is not None:
                keep = torch.cat(
                    [
                        torch.ones(
                            txt_mask.shape[0],
                            img_hw[0] * img_hw[1],
                            device=txt_mask.device,
                            dtype=txt_mask.dtype,
                        ),
                        txt_mask,
                    ],
                    dim=1,
                )
                attn_bias = pad_bias_from_mask(
                    (keep > 0).unsqueeze(1).unsqueeze(2), x.dtype
                )
        for block in self.blocks:
            if isinstance(block, SingleStreamDiTBlock):
                x, txt = block(
                    x,
                    txt,
                    c,
                    img_hw,
                    txt_seq_len,
                    txt_mask,
                    rope_freqs=rope_freqs,
                    attn_bias=attn_bias,
                )
            else:
                x, txt = block(x, txt, c, img_hw, txt_seq_len, txt_mask)
        return self.unpatchify(self.final_layer(x), height, width)

    def get_config(self) -> dict:
        """Complete capacity and architecture metadata, independent of weights."""
        return dict(
            architecture=self.ARCHITECTURE,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            double_stream_depth=self.double_stream_depth,
            single_stream_depth=self.single_stream_depth,
            mlp_ratio=self.mlp_ratio,
        )

    @classmethod
    def from_single_file(cls, checkpoint_path: str) -> "ArtFlow":
        """Load weights with their adjacent transformer_config.json.

        Old checkpoints without metadata require their original source revision;
        head counts, modulation and positional geometry cannot be guessed safely.
        """
        if checkpoint_path.startswith("hf://"):
            parts = checkpoint_path.removeprefix("hf://").split("/")
            repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
            config_path = hf_hub_download(
                repo_id, str(Path(filename).parent / "transformer_config.json")
            )
            checkpoint_path = hf_hub_download(repo_id, filename)
        else:
            config_path = Path(checkpoint_path).with_name("transformer_config.json")
        config = json.loads(Path(config_path).read_text())
        if "architecture" not in config:
            raise ValueError(
                "checkpoint lacks architecture metadata; use its original source revision"
            )
        model = cls(**config)
        if str(checkpoint_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
