"""
ArtFlow Gradio App - HF Spaces deployment.

ZeroGPU-compatible inference for text-to-image generation.
"""

import torch
import gradio as gr
from PIL import Image

# Graceful import of spaces (HF Spaces only)
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Create a dummy decorator for local testing
    class DummyGPU:
        def __call__(self, fn):
            return fn
    spaces = type('obj', (object,), {'GPU': DummyGPU()})()

from artflow import ArtFlowPipeline


REPO_ID = "kaupane/artflow"

# Load pipeline at module level (CPU) — ZeroGPU gives GPU only inside @spaces.GPU
print(f"Loading ArtFlow pipeline from {REPO_ID}...")
pipe = ArtFlowPipeline.from_pretrained(REPO_ID, offload=True)
print("Pipeline loaded!")


@spaces.GPU
def generate_image(
    prompt: str,
    seed: int,
    steps: int,
    height: int,
    width: int,
    guidance_scale: float = 3.0,
    negative_prompt: str = "",
) -> Image.Image:
    """Generate image using the ArtFlow pipeline."""
    # Handle random seed
    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**32, (1,)).item())

    print(f"Generating: '{prompt}' | seed: {seed} | steps: {steps} | {height}x{width}")

    # Generate
    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt if negative_prompt else None,
        seed=seed,
    )

    return result.images[0]


# --- Gradio UI ---
with gr.Blocks(title="ArtFlow") as app:
    gr.Markdown("# ArtFlow Image Generation")
    gr.Markdown("Text-to-image generation using flow matching DiT")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                value="impressionist landscape with water lilies",
                lines=3,
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (optional)",
                value="",
                lines=2,
            )

            with gr.Row():
                seed_input = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0,
                )
                steps_input = gr.Slider(
                    label="Steps",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=1,
                )

            with gr.Row():
                height_input = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1024,
                    value=640,
                    step=16,
                )
                width_input = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1024,
                    value=640,
                    step=16,
                )

            guidance_input = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=10.0,
                value=3.0,
                step=0.5,
            )

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            seed_input,
            steps_input,
            height_input,
            width_input,
            guidance_input,
            negative_prompt_input,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    app.launch()
