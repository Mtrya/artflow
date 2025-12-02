"""Prompt-aware helper to encode text using Qwen3-VL."""

from typing import List, Optional, Tuple

import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen3VLForConditionalGeneration


MAX_SEQUENCE_LENGTH = 1024
DROP_IDX = 38
SYSTEM_PROMPT = "Describe the image, focusing on its content, artistic style, composition, lighting, color, texture, and the spatial relationships between objects and the background:"
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n{user_prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _get_tokenizer(processor: AutoProcessor) -> PreTrainedTokenizerBase:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor must expose a tokenizer for prompt templating.")
    return tokenizer


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return list(torch.split(selected, valid_lengths.tolist(), dim=0))


def _trim_sequence(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.size(0) <= DROP_IDX:
        return sequence.new_zeros((0, sequence.size(1)))
    end = DROP_IDX + MAX_SEQUENCE_LENGTH
    return sequence[DROP_IDX:end]


def _build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=text)


def encode_text(
    texts: List[str],
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    pooling: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Encode captions with the Qwen3-VL chat template for DiT conditioning.

    Returns:
        embeddings: [batch, seq, hidden]
        attention_mask: [batch, seq]
        pooled: [batch, hidden] when pooling is True
    """

    if not texts:
        raise ValueError("texts must contain at least one caption.")

    tokenizer = _get_tokenizer(processor)
    prompts = [_build_prompt(text) for text in texts]

    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH + DROP_IDX,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]

    sequences = _extract_masked_hidden(hidden, inputs.attention_mask)
    trimmed = [_trim_sequence(seq) for seq in sequences]
    max_seq_len = max((seq.size(0) for seq in trimmed), default=0)
    if max_seq_len == 0:
        max_seq_len = 1

    batch_embeddings = []
    batch_masks = []
    for seq in trimmed:
        seq_len = seq.size(0)
        pad_len = max_seq_len - seq_len
        if pad_len > 0:
            pad = seq.new_zeros((pad_len, seq.size(1)))
            seq_padded = torch.cat([seq, pad], dim=0)
        else:
            seq_padded = seq
        batch_embeddings.append(seq_padded)

        mask = seq.new_zeros(max_seq_len, dtype=torch.long)
        mask[:seq_len] = 1
        batch_masks.append(mask)

    embeddings = torch.stack(batch_embeddings).to(model.dtype)
    attention_mask = torch.stack(batch_masks).to(embeddings.device)

    pooled = None
    if pooling:
        weight = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        denom = weight.sum(dim=1).clamp_min(1.0)
        pooled = (embeddings * weight).sum(dim=1) / denom

    return embeddings, attention_mask, pooled


if __name__ == "__main__":
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    texts = [
        "Impressionism landscape by Claude Monet",
        "romanticism marina by Van Gogh",
    ] * 2

    embedding, mask, pooled = encode_text(texts, model, processor, True)

    tokenizer = _get_tokenizer(processor)
    sample_prompt = _build_prompt(texts[0])
    token_info = tokenizer(
        sample_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    ids = token_info.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    prefix_only = PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt="")
    prefix_ids = tokenizer(
        prefix_only,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    ).input_ids[0]
    detected_drop_idx = prefix_ids.shape[0]

    print(f"Configured DROP_IDX={DROP_IDX}, detected prefix length={detected_drop_idx}")
    if detected_drop_idx != DROP_IDX:
        print("WARNING: DROP_IDX does not match detected prefix length!")

    print(f"Embedding shape: {embedding.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print("\nToken inspection (first prompt):")

    sample_embeddings = embedding[0]
    for idx, (tok_id, token) in enumerate(zip(ids.tolist(), tokens)):
        status = "keep" if idx >= DROP_IDX else "drop"
        if status == "keep":
            trimmed_idx = idx - DROP_IDX
            if trimmed_idx < sample_embeddings.size(0):
                emb_vec = sample_embeddings[trimmed_idx]
                emb_preview = ", ".join(f"{v:.4f}" for v in emb_vec[:4])
            else:
                emb_preview = "<truncated>"
        else:
            emb_preview = "-"

        word = tokenizer.decode([tok_id]).strip() or token
        print(
            f"[{idx:03d}] id={tok_id:>6} token={token:<12} word={word:<12} status={status:>4} emb={emb_preview}"
        )
