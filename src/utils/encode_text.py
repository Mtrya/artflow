"""
Helper function to encode text using Qwen3-VL
"""

from typing import List, Optional, Tuple

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

def encode_text(
    texts: List[str],
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    pooling: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Encode text to embeddings for DiT conditioning

    Args:
        text: Batch of captions
        model: Frozen Qwen3-VL model
        processor: Qwen processor
        pooling: True or False

    return:
        embeddings: [batch_size, max_seq_len, hidden_dim]
        attention_mask: [batch_size, max_seq_len], 1 for real tokens, 0 for padding
        pooled: [batch_size, hidden_dim], only when pooling is True
    """

    inputs = processor(
        text=texts, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=256
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1] # [b, s, d]

    mask = inputs.attention_mask # [b, s]

    pooled = None
    if pooling:
        # Mean pooling
        mask_unsqueezed = mask.unsqueeze(-1) # [b, s, 1]
        pooled = (embedding * mask_unsqueezed).sum(1) / mask_unsqueezed.sum(1)

    return embedding, mask, pooled

if __name__ == "__main__":
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    texts = ["Impressionism landscape by Claude Monet", "romanticism marina by Van Gogh"]

    embedding, mask, pooled = encode_text(texts, model, processor, True)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Pooled shape: {pooled.shape}")
