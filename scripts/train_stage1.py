import os
import argparse
import sys
import warnings
import math
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project modules
from src.models.artflow import ArtFlow
from src.dataset.dataloader_utils import ResolutionBucketSampler, collate_fn
from src.utils.encode_text import encode_text
from src.utils.precompute_engine import sample_caption
from src.flow.paths import FlowMatchingOT
from src.utils.evaluation import run_evaluation_light
from transformers.optimization import get_scheduler

# Suppress specific warning about RMSNorm dtype mismatch in mixed precision
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")


@torch.no_grad()
def update_ema_model(
    ema_model: torch.nn.Module, current_model: torch.nn.Module, decay: float
) -> None:
    for ema_param, param in zip(ema_model.parameters(), current_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), current_model.buffers()):
        ema_buffer.copy_(buffer)


def build_linear_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_learning_rate: float,
    base_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(num_warmup_steps, 0)
    total_steps = max(num_training_steps, warmup_steps + 1)
    if base_learning_rate <= 0:
        raise ValueError("Base learning rate must be positive for cosine scheduler")
    clamped_min_lr = min(min_learning_rate, base_learning_rate)
    min_ratio = clamped_min_lr / base_learning_rate

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ArtFlow Stage 1")
    parser.add_argument(
        "--run_name", type=str, default="artflow_run", help="SwanLab run name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluation Config
    parser.add_argument(
        "--vae_path",
        type=str,
        default="REPA-E/e2e-qwenimage-vae",
        help="VAE path (for eval)",
    )  # Note: VAE used for eval decoding
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1000, help="Run evaluation every N steps"
    )
    parser.add_argument("--num_eval_samples", type=int, default=16, help="Evaluate on M samples")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size used in evaluation")

    # Training Config
    parser.add_argument(
        "--precomputed_dataset_path",
        type=str,
        required=True,
        help="Path to precomputed dataset",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Text encoder path",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear_cosine",
        choices=[
            "linear_cosine",
            "constant",
            "constant_with_warmup",
            "linear",
            "cosine",
            "cosine_with_restarts",
        ],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine decay",
    )
    parser.add_argument(
        "--max_steps", type=int, default=50000, help="Total training steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable exponential moving average tracking",
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="EMA decay factor"
    )
    parser.add_argument(
        "--ema_update_interval",
        type=int,
        default=1,
        help="Steps between EMA updates",
    )

    # Model Config
    parser.add_argument("--hidden_size", type=int, default=1152, help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of heads")
    parser.add_argument(
        "--double_stream_depth", type=int, default=28, help="Double stream blocks"
    )
    parser.add_argument(
        "--single_stream_depth", type=int, default=0, help="Single stream blocks"
    )
    parser.add_argument(
        "--conditioning_scheme",
        type=str,
        default="pure",
        choices=["pure", "fused"],
        help="Conditioning scheme",
    )
    parser.add_argument(
        "--modulation_share",
        type=str,
        default="none",
        choices=["none", "stream", "layer", "all"],
        help="Modulation sharing",
    )
    parser.add_argument(
        "--ffn_type",
        type=str,
        default="gated",
        choices=["gated", "standard"],
        help="FFN type",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="swanlab",
        project_config=project_config,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers(
            project_name="artflow-stage1",
            config=vars(args),
            init_kwargs={"swanlab": {"experiment_name": args.run_name}},
        )

    # Load Text Encoder (Frozen, on GPU)
    accelerator.print(f"Loading Text Encoder: {args.text_encoder_path}")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.text_encoder_path,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32,
        device_map=accelerator.device,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(args.text_encoder_path)

    # Load Model
    accelerator.print("Initializing ArtFlow Model...")
    model = ArtFlow(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        double_stream_depth=args.double_stream_depth,
        single_stream_depth=args.single_stream_depth,
        conditioning_scheme=args.conditioning_scheme,
        modulation_share=args.modulation_share,
        ffn_type=args.ffn_type,
        # Default params
        patch_size=2,
        in_channels=16,
        txt_in_features=2048,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.lr_scheduler_type == "linear_cosine":
        lr_scheduler = build_linear_cosine_scheduler(
            optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_steps,
            min_learning_rate=args.min_learning_rate,
            base_learning_rate=args.learning_rate,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_steps,
        )

    ema_model = deepcopy(model) if args.use_ema else None

    # Dataset
    accelerator.print(f"Loading dataset from {args.precomputed_dataset_path}")
    dataset = load_from_disk(args.precomputed_dataset_path)

    # Sampler & Loader
    sampler = ResolutionBucketSampler(
        dataset,
        batch_size=args.batch_size,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Prepare
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if ema_model is not None:
        dtype = next(accelerator.unwrap_model(model).parameters()).dtype
        ema_model.to(accelerator.device, dtype=dtype)
        ema_model.eval()
        ema_model.load_state_dict(accelerator.unwrap_model(model).state_dict())
        for param in ema_model.parameters():
            param.requires_grad_(False)

    # Probability Path
    algorithm = FlowMatchingOT()

    # Training Loop
    global_step = 0
    progress_bar = tqdm(
        range(args.max_steps), disable=not accelerator.is_local_main_process
    )

    train_iter = iter(dataloader)

    model.train()

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            # 1. Text Encoding (On-the-fly)
            captions_list = batch["captions"]  # List[List[str]]

            # Curriculum Sampling
            stage = global_step / (
                args.max_steps * 2
            )  # for a small model, we want to prioritize learning the easy examples
            selected_captions = [
                sample_caption(caps, stage=stage) for caps in captions_list
            ]

            # Encode
            txt, txt_mask, txt_pooled = encode_text(
                selected_captions,
                text_encoder,
                processor,
                pooling=(args.conditioning_scheme == "fused"),
            )

            # 2. Prepare Inputs
            latents = batch["latents"]
            t = torch.rand(latents.shape[0], device=latents.device)

            # Flow Matching
            z1 = latents
            z0 = torch.randn_like(z1)
            z_t = algorithm.sample_zt(z0, z1, t)

            # Forward
            model_output = model(
                z_t, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask
            )

            # Loss
            loss = algorithm.compute_loss(model_output, z0, z1, t)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            lr_scheduler.step()
            global_step += 1

            if ema_model is not None and global_step % args.ema_update_interval == 0:
                update_ema_model(
                    ema_model, accelerator.unwrap_model(model), args.ema_decay
                )

            progress_bar.update(1)

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "train/loss": loss.item(),
                        "train/stage": stage,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            # Checkpointing
            if global_step % args.checkpoint_interval == 0:
                save_path = os.path.join(
                    args.output_dir,
                    f"{args.run_name}/checkpoint_step_{round(global_step / 1000)}k",
                )
                os.makedirs(save_path, exist_ok=True)
                accelerator.save_state(save_path)
                if ema_model is not None and accelerator.is_main_process:
                    torch.save(
                        ema_model.state_dict(),
                        os.path.join(save_path, "ema_weights.pt"),
                    )

            # Evaluation
            if global_step % args.eval_interval == 0:
                eval_model = (
                    ema_model
                    if ema_model is not None
                    else accelerator.unwrap_model(model)
                )
                run_evaluation_light(
                    accelerator=accelerator,
                    model=eval_model,
                    vae_path=args.vae_path,
                    save_path=f"{args.output_dir}/{args.run_name}",
                    current_step=global_step,
                    text_encoder=text_encoder,
                    processor=processor,
                    pooling=(args.conditioning_scheme == "fused"),
                    num_samples=args.num_eval_samples,
                    batch_size=args.eval_batch_size
                )
                model.train()

    accelerator.end_training()
    print("Training finished.")


if __name__ == "__main__":
    main()
