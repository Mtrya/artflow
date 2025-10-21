"""
Main interface of image captioning
"""

import requests
import base64
import os
from io import BytesIO
import concurrent.futures
from typing import Dict

DIRECT_CAPTION_PROMPT = """You are an expert art curator. Write a single, compelling sentence that describes the core essence of this artwork.

Your description must be direct and start immediately with the subject, action, or mood. Do not use introductory phrases like "This is a painting of...", "The image depicts...", or "In this artwork...".

Focus on what's in the image and what makes the image striking."""

REVERSE_IMAGE_PROMPT = """You are a prompt engineer for an advanced text-to-image model like Midjourney or DALL-E 3. Your task is to create the perfect prompt that would generate this exact image.

Your output must be a single string of descriptive phrases and comma-separated keywords. Do not write any explanations.

Include details on:
- Subject and its specific features
- Artistic style (e.g., oil painting, impressionism, ukiyo-e)
- Composition and framing (e.g., wide shot, portrait, rule of thirds)
- Lighting (e.g., dramatic chiaroscuro, soft morning light, golden hour)
- Color palette and mood
- Influences from famous artists if applicable"""

CAPTIONERS = [
    {
        "name": "qwenvl-direct",
        "provider": "dashscope",
        "model": "qwen3-vl-flash",
        "api_key_env": "DASHSCOPE_API_KEY",
        "prompt": DIRECT_CAPTION_PROMPT
    },
    {
        "name": "qwenvl-reverse",
        "provider": "dashscope",
        "model": "qwen3-vl-flash",
        "api_key_env": "DASHSCOPE_API_KEY",
        "prompt": REVERSE_IMAGE_PROMPT
    },
    {
        "name": "gpt-direct",
        "provider": "openrouter",
        "model": "openai/gpt-5-nano",
        "api_key_env": "OPENROUTER_API_KEY",
        "prompt": DIRECT_CAPTION_PROMPT
    },
    {
        "name": "gemini-direct",
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash-lite",
        "api_key_env": "OPENROUTER_API_KEY",
        "prompt": DIRECT_CAPTION_PROMPT
    }
]


def pil_to_base64(image):
    """Convert PIL Image to base64 string for OpenAI-compatible API"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_str

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
    else:
        raise ValueError(f"providers other than dashscope and openrouter not supported")

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
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }

    if provider.lower() == "dashscope" and model == "qwen3-vl-plus":
        payload["extra_body"] = {"enable_thinking": True, "thinking_budget": 64}
    elif provider.lower() == "openrouter" and model == "google/gemini-2.5-flash":
        payload["reasoning"] = {"max_tokens": 64}
    elif provider.lower() == "openrouter" and model == "openai/gpt-5-mini":
        payload["reasoning"] = {"effort": "low"}

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=10
    )

    response.raise_for_status()
    return response.json()

def call_parallel(image, existing_captions: Dict[str,str]={}):
    """
    Call different VLM models simultaneously.
    If existing_captions is provided, it will only call models for missing captions.
    """
    base64_image = pil_to_base64(image)

    if existing_captions is None:
        existing_captions = {c["name"]: None for c in CAPTIONERS}

    captioners_to_call = [
        c for c in CAPTIONERS if existing_captions.get(c["name"]) is None 
        or len(existing_captions.get(c["name"])) < 10
    ]

    if not captioners_to_call:
        return existing_captions
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_captioner = {
            executor.submit(call_vision_api, base64_image, config): config["name"]
            for config in captioners_to_call
        }

        results = {}

        for future in concurrent.futures.as_completed(future_to_captioner):
            captioner_name = future_to_captioner[future]
            try:
                results[captioner_name] = future.result(timeout=15)
            except Exception as exc:
                print(f"{captioner_name} generated an exception: {exc}")
                results[captioner_name] = None

    final_captions = existing_captions.copy()
    for captioner_name, result in results.items():
        caption = None
        if result and 'choices' in result and result['choices']:
            message = result["choices"][0].get("message",{})
            content = message.get("content")
            if content and isinstance(content, str):
                caption_text = content.strip()
                if len(caption_text) >= 10:
                    caption = caption_text
        final_captions[captioner_name] = caption

    return final_captions

if __name__ == "__main__":
    def test_captions():
        """Test caption generation"""
        from PIL import Image
        
        image = Image.open("image.png")

        captions = call_parallel(image,{"qwenvl-reverse": "ocean wave crashing, turbulent foam, golden sunset glow, warm amber and ochre tones, watercolor wash technique, impressionistic brushstrokes, horizontal composition, rule of thirds framing, soft diffused lighting, atmospheric haze, J.M.W. Turner influence, romantic seascape, moody and serene ambiance, textured paper grain, vintage aesthetic"})

        print("\n=== CAPTION RESULTS ===")
        for captioner, caption in captions.items():
            print(f"\n{captioner}:")
            print(f"  {caption}")

    
    test_captions()