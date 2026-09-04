"""
End-of-arm KID evaluation: fixed-seed fakes vs the full held-out real set.

Replaces the old 200-sample FID/KID (statistically meaningless at that size and
computed on a rotating subset). KID is the small-sample-appropriate metric
(unbiased); 2K fakes vs 2.4K real with subset_size=100 gives a stable ranking
signal for arm comparisons. FID is deliberately not computed here — it needs
>=10K samples and is revisited in the heavy eval of stages 4/5.
"""

import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..flow.solvers import sample_ode
from ..utils.encode_text import encode_text
from ..utils.vae_codec import get_vae_stats
from .metrics import aspect_resize_crop, calculate_kid


def _group_by_shape(dataset, indices: List[int]) -> List[List[int]]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i in indices:
        z = dataset[i]["latents"]
        groups.setdefault((z.shape[-2], z.shape[-1]), []).append(i)
    return list(groups.values())


@torch.no_grad()
def run_kid_eval(
    accelerator,
    model: torch.nn.Module,
    vae_path: str,
    save_path: str,
    current_step: int,
    text_encoder,
    tokenizer,
    pooling: bool,
    exit_layer: Optional[int] = None,
    dataset_path: str = "./precomputed_dataset/light-eval@256p",
    num_fake: int = 2000,
    batch_size: int = 20,
    ode_steps: int = 50,
    seed: int = 123,
) -> Dict[str, float]:
    from datasets import load_from_disk
    from diffusers import AutoencoderKLQwenImage

    print(f"Running end-of-arm KID evaluation at step {current_step}...")
    was_training = model.training
    model.eval()

    device = accelerator.device
    num_processes = getattr(accelerator, "num_processes", 1)
    process_index = getattr(accelerator, "process_index", 0)

    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    vae_mean, vae_std = get_vae_stats(vae_path, device=device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

    dataset = load_from_disk(dataset_path)  # unshuffled: fixed index semantics
    num_fake = min(num_fake, len(dataset))

    def to_uint8(images: torch.Tensor) -> torch.Tensor:
        images = torch.clamp((images + 1) / 2, 0, 1).float().cpu()
        images = aspect_resize_crop(images, 299)
        return (images * 255).to(torch.uint8)

    def decode_real(indices: List[int]) -> torch.Tensor:
        outs = []
        for group in _group_by_shape(dataset, indices):
            for start in range(0, len(group), batch_size):
                idxs = group[start : start + batch_size]
                latents = torch.stack(
                    [torch.as_tensor(dataset[i]["latents"]) for i in idxs]
                ).to(device, torch.bfloat16)
                imgs = vae.decode(latents.unsqueeze(2)).sample.squeeze(2)
                outs.append(to_uint8(imgs))
        return torch.cat(outs) if outs else torch.empty(0, dtype=torch.uint8)

    def gen_fake(indices: List[int]) -> torch.Tensor:
        outs = []
        for group in _group_by_shape(dataset, indices):
            for start in range(0, len(group), batch_size):
                idxs = group[start : start + batch_size]
                caps = []
                for i in idxs:
                    c = dataset[i]["captions"]
                    caps.append(c[1] if len(c) > 1 else c[0])
                shapes = torch.as_tensor(dataset[idxs[0]]["latents"]).shape
                noise = torch.stack(
                    [
                        torch.randn(
                            tuple(shapes),
                            generator=torch.Generator(device="cpu").manual_seed(
                                seed + i
                            ),
                        )
                        for i in idxs
                    ]
                ).to(device, torch.bfloat16)
                txt, txt_mask, txt_pooled = encode_text(
                    caps, text_encoder, tokenizer, pooling, exit_layer=exit_layer
                )

                def model_fn(x, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask):
                    if isinstance(t, float):
                        t = torch.tensor(t, device=x.device).expand(x.shape[0])
                    return model(x, t, txt, txt_pooled, txt_mask)

                with accelerator.autocast():
                    samples = sample_ode(
                        model_fn, noise, steps=ode_steps, t_start=0.0, t_end=1.0
                    )
                samples = samples.to(torch.bfloat16) * vae_std + vae_mean
                imgs = vae.decode(samples.unsqueeze(2)).sample.squeeze(2)
                outs.append(to_uint8(imgs))
        return torch.cat(outs) if outs else torch.empty(0, dtype=torch.uint8)

    # Shard rows across ranks
    real_local = decode_real(list(range(len(dataset)))[process_index::max(1, num_processes)])
    fake_local = gen_fake(list(range(num_fake))[process_index::max(1, num_processes)])

    if num_processes > 1 and hasattr(accelerator, "gather_object"):
        real_parts = accelerator.gather_object(real_local)
        fake_parts = accelerator.gather_object(fake_local)
    else:
        real_parts, fake_parts = [real_local], [fake_local]

    metrics: Dict[str, float] = {}
    if accelerator.is_main_process:
        real = torch.cat([p for p in real_parts if len(p) > 0])
        fake = torch.cat([p for p in fake_parts if len(p) > 0])
        kid_mean, kid_std = calculate_kid(
            real, fake, subset_size=100, device=device, batch_size=64, return_std=True
        )
        metrics = {
            "kid/mean": kid_mean,
            "kid/std": kid_std,
            "kid/num_real": float(len(real)),
            "kid/num_fake": float(len(fake)),
        }
        accelerator.log(metrics, step=current_step)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"kid_step_{current_step:06d}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"KID at step {current_step}: {kid_mean:.5f} ± {kid_std:.5f}")

    del vae
    gc.collect()
    torch.cuda.empty_cache()
    if was_training:
        model.train()
    return metrics
