"""
ArtFlow Gradio App - HF Spaces deployment.

ZeroGPU-compatible inference for text-to-image generation.
"""

import os
import random
import gradio as gr

import spaces
from artflow import ArtFlowPipeline


MODEL_PATH = os.environ.get("ARTFLOW_MODEL", "kaupane/ArtFlow")

# Example prompts
EXAMPLE_PROMPTS = [
    ["impressionist landscape with flowers"],
    ["a serene Japanese garden with cherry blossoms"],
    ["vintage portrait of a woman in renaissance style"],
    ["a cozy cottage in the woods during autumn"],
]


# Load pipeline at module level (CPU) — ZeroGPU gives GPU only inside @spaces.GPU
print(f"Loading ArtFlow pipeline from {MODEL_PATH}...")
pipe = ArtFlowPipeline.from_pretrained(MODEL_PATH, offload=True)
print("Pipeline loaded!")


def upsample_prompt(prompt: str) -> str:
    """Upsample prompt using DeepSeek API."""
    if not prompt or not prompt.strip():
        return prompt

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return prompt

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        system_prompt = """你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。
你的工作流程严格遵循一个逻辑序列：
首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。
接着，你会判断提示词是否需要"生成式推理"。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。
然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。
最后，你将为画面选择一种最适合表现该提示词的艺术风格，这将成为作品的灵魂。你可以从印象派水彩的柔和光晕与流动笔触、古典油画的厚重质感与明暗对比、浪漫主义的绚丽色彩与激情笔触、水墨意境的留白写意与墨色层次、或是其他任何合适的艺术风格中选择一种，确保该风格的视觉特征在描述中得到充分体现，使最终画面既有艺术感染力又能准确传达用户提示词所暗示的独特气质。
你的最终描述必须客观、具象、使用自然、流畅的中文，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令，保持在60～100字以内。
仅严格输出最终的修改后的prompt，不要输出任何其他内容。"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户输入 prompt: {prompt}"},
            ],
            max_tokens=200,
            temperature=0.7,
        )

        upsampled = response.choices[0].message.content.strip()
        return upsampled if upsampled else prompt

    except Exception as e:
        print(f"Prompt upsampling failed: {e}")
        return prompt


@spaces.GPU
def generate(
    prompt,
    width=640,
    height=640,
    seed=42,
    steps=50,
    guidance_scale=1.0,
    random_seed=True,
    enable_upsample=False,
    gallery_images=None,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate an image using the ArtFlow model.
    """
    if random_seed:
        new_seed = random.randint(1, 1000000)
    else:
        new_seed = seed if seed != -1 else random.randint(1, 1000000)

    try:
        if pipe is None:
            raise gr.Error("Model not loaded.")

        if not prompt or not prompt.strip():
            raise gr.Error("Please enter a prompt.")

        # Upsample prompt if enabled
        final_prompt = prompt
        if enable_upsample:
            progress(0.05, desc="Upsampling prompt...")
            final_prompt = upsample_prompt(prompt)
            print(f"Upsampled prompt: {final_prompt}")

        progress(0.1, desc="Generating...")

        print(
            f"Generating: '{final_prompt}' | seed: {new_seed} | steps: {steps} | {width}x{height} | cfg: {guidance_scale}"
        )

        # Progress callback
        def step_callback(step, total_steps):
            progress_pct = 0.1 + 0.8 * (step / total_steps)
            progress(progress_pct, desc=f"Denoising step {step}/{total_steps}...")

        result = pipe(
            prompt=final_prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=new_seed,
            progress_callback=step_callback,
        )

        progress(0.95, desc="Decoding image...")
        image = result.images[0]
        progress(1.0, desc="Done!")

    except Exception as e:
        print(f"Generation failed: {e}")
        raise gr.Error(f"Generation failed: {str(e)}")

    if gallery_images is None:
        gallery_images = []

    # Latest output at the top
    gallery_images = [image] + gallery_images

    return gallery_images, new_seed


# ==================== Gradio UI ====================

with gr.Blocks(title="ArtFlow") as demo:
    gr.Markdown(
        """<div align="center">
# 🎨 ArtFlow Image Generation
*Flow-matching DiT for artistic image generation*
</div>"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                lines=3,
                placeholder="Describe what you want to generate...",
            )

            with gr.Row():
                enable_upsample = gr.Checkbox(
                    label="✨ Upsample Prompt (DeepSeek)",
                    value=False,
                    info="Enhance prompt with AI",
                )

            with gr.Row():
                width = gr.Slider(
                    label="Width", minimum=256, maximum=1024, value=640, step=16
                )
                height = gr.Slider(
                    label="Height", minimum=256, maximum=1024, value=640, step=16
                )

            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                random_seed = gr.Checkbox(label="Random Seed", value=True)

            with gr.Row():
                steps = gr.Slider(
                    label="Steps", minimum=10, maximum=100, value=50, step=1
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.5,
                )

            generate_btn = gr.Button("🚀 Generate", variant="primary")

            # Example prompts
            gr.Markdown("### 📝 Example Prompts")
            gr.Examples(examples=EXAMPLE_PROMPTS, inputs=prompt_input, label=None)

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="Generated Images",
                columns=2,
                rows=2,
                height=600,
                object_fit="contain",
                format="png",
                interactive=False,
            )
            used_seed = gr.Textbox(label="Seed Used", interactive=False)
            seed_state = gr.State()

    # Event handlers
    generate_btn.click(
        generate,
        inputs=[
            prompt_input,
            width,
            height,
            seed,
            steps,
            guidance_scale,
            random_seed,
            enable_upsample,
            output_gallery,
        ],
        outputs=[output_gallery, seed_state],
    ).then(
        lambda s: (str(s), int(s)),
        inputs=[seed_state],
        outputs=[used_seed, seed],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()
