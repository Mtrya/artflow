"""Prompt-template constants shared by CPU metadata builders and GPU encoding."""

MAX_SEQUENCE_LENGTH = 2048
DROP_IDX = 38
RETAINED_MIN_LENGTH = 1
SYSTEM_PROMPT = (
    "Describe the image, focusing on its content, artistic style, composition, "
    "lighting, color, texture, and the spatial relationships between objects "
    "and the background:"
)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n{user_prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
