"""Prompt-template constants shared by CPU metadata builders and GPU encoding."""

# Training truncates the prompt here.  The caption prompts ask for much shorter
# targets (the length bands top out at 1,280 retained tokens), but a caption
# that came out longer than asked is kept rather than cut at the request size.
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
