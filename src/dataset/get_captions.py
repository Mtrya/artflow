"""
Main interface of image captioning
"""

import requests
import base64
import os
from io import BytesIO
import concurrent.futures
from typing import Dict, List, Optional

from PIL import Image

SYSTEM_PROMPT = """You are analyzing artwork. Be accurate, concise, and describe only what you directly observe in the image in English."""

DIRECT_CAPTION_PROMPT = """You are an expert art curator. Write a single, compelling sentence that describes the core essence of this artwork.

Your description must be direct and start immediately with the subject, action, or mood. Do not use introductory phrases like "This is a painting of...", "The image depicts...", or "In this artwork...".

Focus on what's in the image and what makes the image striking. Maintain authenticity and accuracy, avoid generalizations."""

SPATIAL_RELATIONSHIP_PROMPT = """You are an expert writing 'alt text' for web accessibility. Your task is to write a single, objective sentence describing the image's content, focusing on the spatial arrangement of its key elements.

Your description must be direct and start immediately with the subject, action, or mood. Do not use introductory phrases like "This is a painting of...", "The image depicts...", or "In this artwork...".

Focus on the location of objects relative to each other and the frame. Use clear, directional language. (e.g., 'the right foreground', 'behind') Cover emotion or style briefly and stick to a literal, spatial description."""

REVERSE_IMAGE_PROMPT = """You are a prompt engineer for an advanced text-to-image model like Midjourney or DALL-E 3. Your task is to create the perfect prompt that would generate this exact image.

Your output must be a single string of descriptive phrases and comma-separated keywords. Do not write any explanations. Output directly without quotation marks.

Include details on:
- Subject and its specific features (location, texture, etc.)
- Object attributes, vision relations between objects, and environmental details
- Artistic style
- Composition and framing
- Lighting
- Color palette and mood
- Influences from famous artists (only if applicable)"""

CAPTIONERS_QWEN = [
    {
        "name": "qwen-direct",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "api_key_env": "SILICONFLOW_API_KEY",
        "prompt": DIRECT_CAPTION_PROMPT
    },
    {
        "name": "qwen-spatial",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "api_key_env": "SILICONFLOW_API_KEY",
        "prompt": SPATIAL_RELATIONSHIP_PROMPT
    },
    {
        "name": "qwen-reverse",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "api_key_env": "SILICONFLOW_API_KEY",
        "prompt": REVERSE_IMAGE_PROMPT
    },
]

CAPTIONERS_MISTRAL = [
    {
        "name": "mistral",
        "provider": "openrouter",
        "model": "mistralai/mistral-small-3.2-24b-instruct",
        "api_key_env": "OPENROUTER_API_KEY",
        "prompt": DIRECT_CAPTION_PROMPT
    }
]

def pil_to_base64(image) -> str:
    """Convert PIL Image to base64 string for OpenAI-compatible API"""
    # Resize to 3/5 of original dimensions to speed up generation
    width, height = image.size
    new_width = int(width * 0.6)
    new_height = int(height * 0.6)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    buffer = BytesIO()
    resized_image.save(buffer, format="PNG")
    image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_str

def unwrap_quotes(s: str) -> str:
    """Remove surrounding quotes if present"""
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def call_vision_api(base64_image: str, captioner_config: Dict[str,str]):
    """Call vision API"""
    provider = captioner_config["provider"]
    model = captioner_config["model"]
    api_key = os.getenv(captioner_config["api_key_env"])
    prompt = captioner_config["prompt"]

    if provider.lower() == "dashscope":
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif provider.lower() == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
    elif provider.lower() == "siliconflow":
        base_url = "https://api.siliconflow.cn/v1"
    else:
        raise ValueError(f"provider {provider} not supported.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ]
    }

    if provider.lower() == "dashscope" and model.startswith("qwen3-vl"):
        payload["extra_body"] = {"enable_thinking": False}
    elif provider.lower() == "openrouter" and model.startswith("google/gemini-2.5"):
        payload["reasoning"] = {"max_tokens": 4}
    elif provider.lower() == "openrouter" and model.startswith("openai/gpt-5"):
        payload["reasoning"] = {"effort": "low"}
    elif provider.lower() == "siliconflow" and model == "zai-org/GLM-4.5V":
        payload["enable_thinking"] = False

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=45
    )

    response.raise_for_status()
    return response.json()

def call_parallel(images, captioners: Optional[List[Dict[str,str]]]=CAPTIONERS_QWEN, existing_captions: Optional[List[Dict[str,str]]]=None) -> List[Dict[str,str]]:
    """
    Call different VLM models simultaneously.
    If existing_captions is provided, it will only call models for missing captions.
    """
    base64_images = [pil_to_base64(img) for img in images]

    if existing_captions is None:
        existing_captions = [{c["name"]: None for c in captioners} for _ in images]
    
    assert len(base64_images) == len(existing_captions)

    # Flatten all (image_idx, captioner) pairs that need calls
    tasks = []
    for img_idx, base64_image in enumerate(base64_images):
        for captioner in captioners:
            if existing_captions[img_idx].get(captioner["name"]) is None:
                tasks.append((img_idx, base64_image, captioner))

    if not tasks:
        return existing_captions
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map future -> (image_idx, captioner_name)
        future_to_task = {
            executor.submit(call_vision_api, base64_image, captioner): (img_idx, captioner["name"])
            for img_idx, base64_image, captioner in tasks
        }
        
        # Initialize results structure
        results = [{} for _ in images]

        for future in concurrent.futures.as_completed(future_to_task):
            img_idx, captioner_name = future_to_task[future]
            try:
                results[img_idx][captioner_name] = future.result(timeout=50)
            except Exception as exc:
                print(f"{captioner_name} for image {img_idx} failed: {exc}")
                results[img_idx][captioner_name] = None
        
    # Process results and merge with existing captions
    final_captions = [existing.copy() for existing in existing_captions]
    for img_idx, img_results in enumerate(results):
        for captioner_name, result in img_results.items():
            caption = None
            # Extract message content from openai-compatible api response
            if result and 'choices' in result and result['choices']:
                message = result["choices"][0].get("message",{})
                content = message.get("content")
                if content and isinstance(content, str):
                    caption_text = content.strip()
                    caption_text = unwrap_quotes(caption_text)
                    if len(caption_text) >= 10:
                        caption = caption_text

            final_captions[img_idx][captioner_name] = caption

    
    return final_captions

if __name__ == "__main__":
    def test_captions():
        """Test caption generation"""
        from PIL import Image

        images = [Image.open("test_image.png")]

        captions_batch = call_parallel(images, [{}])

        print("\n=== CAPTION RESULTS ===")
        for img_idx, captions in enumerate(captions_batch):
            for captioner, caption in captions.items():
                print(f"\n{captioner} for image {img_idx}:")
                print(f"  {caption}")

    test_captions()