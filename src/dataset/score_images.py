"""
Manual image scoring interface with integrated VLM prompt generation.

Usage:
    python -m src.dataset.score_images \\
        --dataset_name "kaupane/wikiart-captions" \\
        --image_field "image" \\
        --target_samples 5000 \\
        --output_dir "./scored_images" \\
        --min_resolution 640

Features:
- Gradio interface for 1-5 star scoring
- Automatic hash-based deduplication and resume
- Aspect ratio and resolution filtering
- VLM prompt generation for 5-star images
- Export scored images to HuggingFace dataset format
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
import requests
from datasets import Dataset, load_dataset
from PIL import Image
import io


class ImageScorer:
    def __init__(
        self,
        dataset_name: str,
        image_field: str,
        target_samples: int,
        output_dir: str,
        min_resolution: int = 640,
        aspect_ratio_min: float = 0.8,
        aspect_ratio_max: float = 1.25,
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.dataset_name = dataset_name
        self.image_field = image_field
        self.target_samples = target_samples
        self.output_dir = Path(output_dir)
        self.min_resolution = min_resolution
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max

        self.dataset_configs = dataset_configs or [
            {"name": dataset_name, "image_field": image_field, "weight": 1.0}
        ]

        # Create directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.scores_file = self.output_dir / "scores.jsonl"
        self.prompts_file = self.output_dir / "prompts.jsonl"

        # State
        self.datasets: List[Dict[str, Any]] = []
        self.total_weight = 0.0
        self.cumulative_weights: List[float] = []
        self.scored_hashes = self._load_scored_hashes()
        self.current_item = None
        self.current_image: Optional[Image.Image] = None  # Keep image in memory
        self.pending_prompts: Optional[List[str]] = None

        # API
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            print("Warning: SILICONFLOW_API_KEY not found in environment")

    def _load_scored_hashes(self) -> set:
        """Load already scored image hashes to support resume."""
        scored = set()
        if self.scores_file.exists():
            with open(self.scores_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    scored.add(data["hash"])
        return scored

    def _generate_hash(self, dataset_name: str, index: int) -> str:
        """Generate hash from dataset name and index."""
        content = f"{dataset_name}_{index}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_image_exists(self, hash_str: str) -> bool:
        """Check if image with this hash already exists."""
        return (self.images_dir / f"{hash_str}.png").exists()

    def _fetch_image(self, image_data: Any) -> Optional[Image.Image]:
        """Fetch image from various sources."""
        # PIL Image
        if isinstance(image_data, Image.Image):
            return image_data

        # URL string
        if isinstance(image_data, str) and (
            image_data.startswith("http://") or image_data.startswith("https://")
        ):
            try:
                response = requests.get(image_data, timeout=10)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            except Exception as e:
                print(f"Failed to fetch image from URL: {e}")
                return None

        # Dict with URL or image
        if isinstance(image_data, dict):
            if "url" in image_data:
                return self._fetch_image(image_data["url"])
            if "bytes" in image_data:
                return Image.open(io.BytesIO(image_data["bytes"]))

        return None

    def _check_image_quality(self, img: Image.Image) -> Tuple[bool, str]:
        """Check if image meets quality requirements."""
        w, h = img.size
        ratio = w / h

        # Check aspect ratio
        if not (self.aspect_ratio_min <= ratio <= self.aspect_ratio_max):
            return False, f"Aspect ratio {ratio:.2f} out of range [{self.aspect_ratio_min}, {self.aspect_ratio_max}]"

        # Check resolution
        min_side = min(w, h)
        if min_side < self.min_resolution:
            return False, f"Min side {min_side} < {self.min_resolution}"

        return True, "OK"

    def _resize_and_save(self, img: Image.Image, hash_str: str) -> Path:
        """Resize image to 640x640 and save."""
        img_resized = img.resize((640, 640), Image.LANCZOS)
        save_path = self.images_dir / f"{hash_str}.png"
        img_resized.save(save_path)
        return save_path
    
    def _resize_for_vlm(self, img: Image.Image, target_size: int = 256) -> Image.Image:
        """Resize image to smaller size for VLM API to reduce cost."""
        return img.resize((target_size, target_size), Image.LANCZOS)

    def _call_vlm_api(self, image: Image.Image) -> Optional[List[str]]:
        """Call VLM API to generate prompts inspired by the image."""
        if not self.api_key:
            return None

        try:
            import base64

            # Resize to reduce API cost (384x384 instead of 640x640)
            image_small = self._resize_for_vlm(image, target_size=384)
            
            # Encode image
            buffer = io.BytesIO()
            image_small.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # API request
            url = "https://api.siliconflow.cn/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            instruction = """你是一位二十世纪的艺术策展人。请仔细观察这张图像,它会给你带来什么灵感?

请基于这张图像的启发,想象3个完全不同的艺术场景。每个场景都应该:
1. 是独特的想象,而不是描述这张图片本身
2. 包含详细的视觉细节(风格、纹理、光影、色调等)
3. 控制在77个token以内
4. 用中文描述
5. 不包含过于现代的意象

请以JSON格式返回,包含一个prompts数组,每个元素是一个场景描述字符串。

示例格式:
{"prompts": ["描述1", "描述2", "描述3"]}"""

            payload = {
                "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.8,
                "response_format": {"type": "json_object"},
            }

            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            prompts_data = json.loads(content)
            return prompts_data.get("prompts", [])

        except Exception as e:
            print(f"VLM API call failed: {e}")
            return None

    def _save_score(self, hash_str: str, dataset_name: str, index: int, score: float):
        """Save score to JSONL file."""
        with open(self.scores_file, "a") as f:
            data = {
                "hash": hash_str,
                "dataset_name": dataset_name,
                "index": index,
                "score": score,
                "image_path": str(self.images_dir / f"{hash_str}.png"),
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.scored_hashes.add(hash_str)

    def _save_prompts(self, hash_str: str, prompts: List[str]):
        """Save prompts to JSONL file."""
        with open(self.prompts_file, "a") as f:
            data = {
                "hash": hash_str,
                "prompts": prompts,
                "image_path": str(self.images_dir / f"{hash_str}.png"),
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_dataset(self):
        """Load all datasets and prepare weighted random sampling."""
        self.datasets = []
        self.total_weight = 0.0
        self.cumulative_weights = []

        for cfg in self.dataset_configs:
            name = cfg["name"]
            image_field = cfg["image_field"]
            weight = float(cfg.get("weight", 1.0))
            if weight <= 0:
                continue

            print(f"Loading dataset: {name}")
            dataset = load_dataset(name, split="train")
            length = len(dataset)
            print(f"Dataset loaded: {length} examples")

            self.datasets.append(
                {
                    "name": name,
                    "image_field": image_field,
                    "weight": weight,
                    "dataset": dataset,
                    "length": length,
                }
            )
            self.total_weight += weight
            self.cumulative_weights.append(self.total_weight)

        if not self.datasets:
            raise ValueError("No valid datasets loaded")

    def _choose_dataset(self) -> Dict[str, Any]:
        """Select a dataset according to configured weights."""
        if not self.datasets:
            raise ValueError("Datasets not loaded")

        r = random.random() * self.total_weight
        for entry, cumulative in zip(self.datasets, self.cumulative_weights):
            if r <= cumulative:
                return entry
        return self.datasets[-1]

    def get_next_image(self) -> Tuple[Optional[Image.Image], Optional[Dict]]:
        """Get next valid image for scoring (keeps in memory, doesn't save yet)."""
        if not self.datasets:
            return None, {"error": "Datasets not loaded"}

        max_attempts = 2000
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            if len(self.scored_hashes) >= self.target_samples:
                return None, {"status": f"Target reached: {self.target_samples} images scored"}

            entry = self._choose_dataset()
            dataset = entry["dataset"]
            image_field = entry["image_field"]
            dataset_name = entry["name"]
            length = entry["length"]

            index = random.randint(0, length - 1)
            hash_str = self._generate_hash(dataset_name, index)

            if hash_str in self.scored_hashes:
                continue

            if self._check_image_exists(hash_str):
                continue

            item = dataset[index]
            image_data = item.get(image_field)

            if image_data is None:
                continue

            img = self._fetch_image(image_data)
            if img is None:
                continue

            valid, reason = self._check_image_quality(img)
            if not valid:
                continue

            img_resized = img.resize((640, 640), Image.LANCZOS)

            self.current_item = {
                "hash": hash_str,
                "dataset_name": dataset_name,
                "index": index,
            }
            self.current_image = img_resized

            info = {
                "hash": hash_str,
                "index": index,
                "dataset": dataset_name,
                "scored": len(self.scored_hashes),
                "target": self.target_samples,
                "progress": f"{len(self.scored_hashes)}/{self.target_samples}",
                "reason": reason,
            }

            return img_resized, info

        return None, {"error": f"Failed to find valid image after {max_attempts} attempts"}

    def score_image(self, score: float) -> Tuple[Optional[Image.Image], Dict, str]:
        """Score current image. For 5-star images, keep item in memory for prompt generation."""
        if self.current_item is None or self.current_image is None:
            return None, {"error": "No current image"}, "normal"

        hash_str = self.current_item["hash"]
        dataset_name = self.current_item["dataset_name"]
        index = self.current_item["index"]

        # NOW save the image to disk (only when scored!)
        save_path = self._resize_and_save(self.current_image, hash_str)

        # Save score
        self._save_score(hash_str, dataset_name, index, score)

        # If score < 5.0, just move to next
        if score < 5.0:
            self.current_item = None
            self.current_image = None
            self.pending_prompts = None

            # Get next image
            next_img, info = self.get_next_image()
            return next_img, info, "normal"

        # Score = 5.0: keep in memory and show "Create Prompt" button
        return self.current_image, {"status": "Click 'Create Prompt' to generate prompts"}, "create_prompt"

    def create_prompts(self) -> Tuple[Optional[str], bool]:
        """Generate prompts for current 5-star image."""
        if self.current_item is None or self.current_image is None:
            return "No current image", False

        prompts = self._call_vlm_api(self.current_image)

        if prompts:
            self.pending_prompts = prompts
            prompts_text = "\n\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts)])
            return prompts_text, True
        else:
            return "Prompt generation failed", False

    def skip_prompt_generation(self) -> Tuple[Optional[Image.Image], Dict]:
        """Skip prompt generation and move to next image."""
        # Clean up and get next
        self.current_item = None
        self.current_image = None
        self.pending_prompts = None
        return self.get_next_image()

    def save_prompts_decision(self, selected_indices: str) -> Tuple[Optional[Image.Image], Dict]:
        """Save selected prompts and get next image."""
        if self.pending_prompts is None or self.current_item is None:
            return None, {"error": "No pending prompts"}

        # Parse selected indices
        if selected_indices.strip():
            try:
                indices = [int(x.strip()) - 1 for x in selected_indices.split(",")]
                selected = [self.pending_prompts[i] for i in indices if 0 <= i < len(self.pending_prompts)]
                if selected:
                    self._save_prompts(self.current_item["hash"], selected)
            except Exception as e:
                print(f"Failed to parse indices: {e}")

        # Clean up and get next
        self.current_item = None
        self.current_image = None
        self.pending_prompts = None
        return self.get_next_image()


def create_gradio_interface(scorer: ImageScorer):
    """Create Gradio interface for image scoring."""

    with gr.Blocks(title="ArtFlow Image Scorer") as demo:
        gr.Markdown("# ArtFlow Image Scorer\nScore images from 1-5 stars for reward model training")

        with gr.Row():
            with gr.Column(scale=2):
                image_display = gr.Image(label="Current Image", type="pil", interactive=False)

            with gr.Column(scale=1):
                info_display = gr.JSON(label="Info")

                with gr.Row():
                    score_0 = gr.Button("❌ 0", variant="stop")
                    score_1 = gr.Button("⭐ 1", variant="secondary")
                    score_2 = gr.Button("⭐⭐ 2", variant="secondary")
                    score_3 = gr.Button("⭐⭐⭐ 3", variant="secondary")
                    score_4 = gr.Button("⭐⭐⭐⭐ 4", variant="secondary")
                    score_5 = gr.Button("⭐⭐⭐⭐⭐ 5", variant="primary")

        # Create Prompt button (always visible, disabled until 5-star)
        with gr.Row(visible=True) as create_prompt_section:
            create_prompt_btn = gr.Button("🎨 Create Prompt", variant="primary", size="lg", interactive=False)
            skip_prompt_btn = gr.Button("Skip → Next Image", variant="secondary", interactive=False)

        # Prompt review section (hidden by default)
        with gr.Row(visible=False) as prompt_section:
            with gr.Column():
                prompts_display = gr.Textbox(
                    label="Generated Prompts (Review and select)",
                    lines=10,
                    interactive=False,
                )
                prompt_input = gr.Textbox(
                    label="Select prompts to keep (comma-separated, e.g., '1,2,3')",
                    placeholder="1,2,3",
                    interactive=False,
                )
                submit_prompts_btn = gr.Button("Submit Selection & Next Image", interactive=False)

        # State to track UI mode: "normal", "create_prompt", or "review_prompts"
        ui_mode = gr.State("normal")

        def button_states(
            scoring_enabled: bool,
            create_enabled: bool,
            skip_enabled: bool,
            submit_enabled: bool,
        ) -> List[Any]:
            scoring_update = gr.update(interactive=scoring_enabled)
            return [
                scoring_update,
                scoring_update,
                scoring_update,
                scoring_update,
                scoring_update,
                scoring_update,
                gr.update(interactive=create_enabled),
                gr.update(interactive=skip_enabled),
                gr.update(interactive=submit_enabled),
            ]

        def pack_output(
            img,
            info,
            mode,
            create_visible,
            prompt_visible,
            prompts_text,
            prompt_input_update,
            buttons: List[Any],
        ) -> List[Any]:
            return [
                img,
                info,
                mode,
                create_visible,
                prompt_visible,
                prompts_text,
                prompt_input_update,
                *buttons,
            ]

        def load_first_image():
            img, info = scorer.get_next_image()
            buttons = button_states(scoring_enabled=True, create_enabled=False, skip_enabled=False, submit_enabled=False)
            return pack_output(
                img,
                info,
                "normal",
                gr.Row(visible=True),
                gr.Row(visible=False),
                "",
                gr.update(value="", interactive=False),
                buttons,
            )

        def handle_score(score_val):
            img, info, mode = scorer.score_image(score_val)

            if mode == "create_prompt":
                # Show "Create Prompt" button
                buttons = button_states(
                    scoring_enabled=False,
                    create_enabled=True,
                    skip_enabled=True,
                    submit_enabled=False,
                )
                return pack_output(
                    img,
                    info,
                    "create_prompt",
                    gr.Row(visible=True),
                    gr.Row(visible=False),
                    "",
                    gr.update(value="", interactive=False),
                    buttons,
                )
            else:
                buttons = button_states(
                    scoring_enabled=True,
                    create_enabled=False,
                    skip_enabled=False,
                    submit_enabled=False,
                )
                return pack_output(
                    img,
                    info,
                    "normal",
                    gr.Row(visible=True),
                    gr.Row(visible=False),
                    "",
                    gr.update(value="", interactive=False),
                    buttons,
                )

        def handle_create_prompt():
            # Step 1: disable all while generating
            busy_buttons = button_states(False, False, False, False)
            yield pack_output(
                gr.update(),
                {"status": "Generating prompts..."},
                "create_prompt",
                gr.Row(visible=True),
                gr.Row(visible=False),
                "Generating prompts...",
                gr.update(value="", interactive=False),
                busy_buttons,
            )

            prompts_text, success = scorer.create_prompts()
            if success:
                ready_buttons = button_states(False, False, False, True)
                yield pack_output(
                    gr.update(),
                    {"status": "Review generated prompts"},
                    "review_prompts",
                    gr.Row(visible=True),
                    gr.Row(visible=True),
                    prompts_text,
                    gr.update(value="", interactive=True),
                    ready_buttons,
                )
            else:
                img, info = scorer.skip_prompt_generation()
                normal_buttons = button_states(True, False, False, False)
                yield pack_output(
                    img,
                    info,
                    "normal",
                    gr.Row(visible=True),
                    gr.Row(visible=False),
                    prompts_text,
                    gr.update(value="", interactive=False),
                    normal_buttons,
                )

        def handle_skip_prompt():
            # Skip prompt generation, move to next image
            img, info = scorer.skip_prompt_generation()
            buttons = button_states(True, False, False, False)
            return pack_output(
                img,
                info,
                "normal",
                gr.Row(visible=True),
                gr.Row(visible=False),
                "",
                gr.update(value="", interactive=False),
                buttons,
            )

        def handle_prompt_submission(selected):
            img, info = scorer.save_prompts_decision(selected)
            buttons = button_states(True, False, False, False)
            return pack_output(
                img,
                info,
                "normal",
                gr.Row(visible=True),
                gr.Row(visible=False),
                "",
                gr.update(value="", interactive=False),
                buttons,
            )

        def make_score_handler(score_val: float):
            def _handler():
                return handle_score(score_val)
            return _handler

        # Load first image on start
        demo.load(
            load_first_image,
            outputs=[
                image_display,
                info_display,
                ui_mode,
                create_prompt_section,
                prompt_section,
                prompts_display,
                prompt_input,
                score_0,
                score_1,
                score_2,
                score_3,
                score_4,
                score_5,
                create_prompt_btn,
                skip_prompt_btn,
                submit_prompts_btn,
            ],
        )

        # Score buttons
        for button, value in [
            (score_0, 0.0),
            (score_1, 1.0),
            (score_2, 2.0),
            (score_3, 3.0),
            (score_4, 4.0),
            (score_5, 5.0),
        ]:
            button.click(
                make_score_handler(value),
                outputs=[
                    image_display,
                    info_display,
                    ui_mode,
                    create_prompt_section,
                    prompt_section,
                    prompts_display,
                    prompt_input,
                    score_0,
                    score_1,
                    score_2,
                    score_3,
                    score_4,
                    score_5,
                    create_prompt_btn,
                    skip_prompt_btn,
                    submit_prompts_btn,
                ],
            )

        # Create Prompt button
        create_prompt_btn.click(
            handle_create_prompt,
            outputs=[
                image_display,
                info_display,
                ui_mode,
                create_prompt_section,
                prompt_section,
                prompts_display,
                prompt_input,
                score_0,
                score_1,
                score_2,
                score_3,
                score_4,
                score_5,
                create_prompt_btn,
                skip_prompt_btn,
                submit_prompts_btn,
            ],
        )

        # Skip Prompt button
        skip_prompt_btn.click(
            handle_skip_prompt,
            outputs=[
                image_display,
                info_display,
                ui_mode,
                create_prompt_section,
                prompt_section,
                prompts_display,
                prompt_input,
                score_0,
                score_1,
                score_2,
                score_3,
                score_4,
                score_5,
                create_prompt_btn,
                skip_prompt_btn,
                submit_prompts_btn,
            ],
        )

        # Prompt submission
        submit_prompts_btn.click(
            handle_prompt_submission,
            inputs=[prompt_input],
            outputs=[
                image_display,
                info_display,
                ui_mode,
                create_prompt_section,
                prompt_section,
                prompts_display,
                prompt_input,
                score_0,
                score_1,
                score_2,
                score_3,
                score_4,
                score_5,
                create_prompt_btn,
                skip_prompt_btn,
                submit_prompts_btn,
            ],
        )

    return demo


def export_scores_to_dataset(output_dir: str):
    """Export scored images to HuggingFace dataset format."""
    output_path = Path(output_dir)
    scores_file = output_path / "scores.jsonl"

    if not scores_file.exists():
        print(f"No scores file found at {scores_file}")
        return

    # Load scores
    scores_data = []
    with open(scores_file, "r") as f:
        for line in f:
            scores_data.append(json.loads(line))

    # Create dataset
    dataset = Dataset.from_list(scores_data)

    # Save
    dataset_path = output_path / "scored_dataset"
    dataset.save_to_disk(str(dataset_path))
    print(f"Saved scored dataset to {dataset_path}")


def export_prompts_to_dataset(output_dir: str):
    """Export generated prompts to HuggingFace dataset format."""
    output_path = Path(output_dir)
    prompts_file = output_path / "prompts.jsonl"

    if not prompts_file.exists():
        print(f"No prompts file found at {prompts_file}")
        return

    # Load prompts
    prompts_data = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompts_data.append(json.loads(line))

    # Create dataset
    dataset = Dataset.from_list(prompts_data)

    # Save
    dataset_path = output_path / "prompts_dataset"
    dataset.save_to_disk(str(dataset_path))
    print(f"Saved prompts dataset to {dataset_path}")


def _parse_dataset_arg(raw: str) -> Dict[str, Any]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    data: Dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key.strip()] = value.strip()

    if "name" not in data or "image_field" not in data:
        raise argparse.ArgumentTypeError("--dataset requires name and image_field, e.g., name=...,image_field=...,weight=1.0")

    data["weight"] = float(data.get("weight", 1.0))
    return data


def main():
    parser = argparse.ArgumentParser(description="Manual image scoring with VLM integration")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Score command
    score_parser = subparsers.add_parser("score", help="Start scoring interface")
    score_parser.add_argument("--dataset_name", type=str, required=False, help="HuggingFace dataset name (legacy single dataset)")
    score_parser.add_argument("--image_field", type=str, required=False, help="Field name for images (legacy single dataset)")
    score_parser.add_argument("--dataset", dest="datasets", action="append", type=_parse_dataset_arg, help="Repeatable dataset config: name=...,image_field=...,weight=1.0")
    score_parser.add_argument("--target_samples", type=int, default=5000, help="Target number of samples to score")
    score_parser.add_argument("--output_dir", type=str, default="./scored_images", help="Output directory")
    score_parser.add_argument("--min_resolution", type=int, default=640, help="Minimum resolution (shorter side)")
    score_parser.add_argument("--aspect_ratio_min", type=float, default=0.8, help="Minimum aspect ratio")
    score_parser.add_argument("--aspect_ratio_max", type=float, default=1.25, help="Maximum aspect ratio")
    score_parser.add_argument("--port", type=int, default=7860, help="Gradio port")

    # Export scores command
    export_scores_parser = subparsers.add_parser("export-scores", help="Export scores to dataset")
    export_scores_parser.add_argument("--output_dir", type=str, required=True, help="Scoring output directory")

    # Export prompts command
    export_prompts_parser = subparsers.add_parser("export-prompts", help="Export prompts to dataset")
    export_prompts_parser.add_argument("--output_dir", type=str, required=True, help="Scoring output directory")

    args = parser.parse_args()

    if args.command == "score":
        dataset_configs: Optional[List[Dict[str, Any]]] = None

        if args.datasets:
            dataset_configs = args.datasets
            primary = dataset_configs[0]
            dataset_name = primary["name"]
            image_field = primary["image_field"]
        else:
            if not args.dataset_name or not args.image_field:
                raise ValueError("Provide either --dataset (repeatable) or legacy --dataset_name and --image_field")
            dataset_name = args.dataset_name
            image_field = args.image_field

        scorer = ImageScorer(
            dataset_name=dataset_name,
            image_field=image_field,
            target_samples=args.target_samples,
            output_dir=args.output_dir,
            min_resolution=args.min_resolution,
            aspect_ratio_min=args.aspect_ratio_min,
            aspect_ratio_max=args.aspect_ratio_max,
            dataset_configs=dataset_configs,
        )

        scorer.load_dataset()

        demo = create_gradio_interface(scorer)
        demo.launch(server_port=args.port, share=False)

    elif args.command == "export-scores":
        export_scores_to_dataset(args.output_dir)

    elif args.command == "export-prompts":
        export_prompts_to_dataset(args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
