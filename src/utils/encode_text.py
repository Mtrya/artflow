"""Prompt-aware helper to encode text using a Qwen3 causal LM."""

from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from .prompt_contract import (
    DROP_IDX,
    MAX_SEQUENCE_LENGTH,
    PROMPT_TEMPLATE,
    RETAINED_MIN_LENGTH,
    SYSTEM_PROMPT,
)

EXIT_MODES = ("full_forward_slice", "stop_at_layer")


class _EarlyExit(Exception):
    """Internal control-flow signal: the k-th layer hook fired."""

    def __init__(self, hidden: torch.Tensor):
        super().__init__()
        self.hidden = hidden


def _encode_early_exit(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    exit_layer: int,
    use_cache: bool,
) -> torch.Tensor:
    """Frozen-LM forward that STOPS after layers[exit_layer-1].

    True early exit: skips layers exit_layer..N instead of running
    a full forward and slicing hidden_states[exit_layer]. HF semantics make the
    hook output bit-identical to the slice: hidden_states[i] is the output of
    layers[i-1], before the final norm. No autograd graph exists (no_grad
    caller), so aborting the forward mid-way via an exception in a forward
    hook is safe; the hook fires synchronously right after the layer's output
    is computed.
    """
    layers = model.model.layers
    k = int(exit_layer)
    if k <= 0 or k > len(layers):
        raise ValueError(
            f"exit_layer={k} out of range [1, {len(layers)}] (layers are 1-indexed"
            " for early exit; embedding output is not a valid stop point)"
        )

    captured = {}

    def _hook(module, args, output):
        if isinstance(output, tuple):
            captured["hidden"] = output[0]
        else:
            captured["hidden"] = output
        raise _EarlyExit(captured["hidden"])

    # Registering on layers[k - 1] yields the same tensor the full-forward
    # path returns as hidden_states[k]: HF numbers the embedding as
    # hidden_states[0] and the output of layers[i - 1] as hidden_states[i],
    # both before the final norm. Aborting here is therefore bit-identical to
    # running the remaining layers and slicing afterwards — the skipped
    # layers simply never compute.
    handle = layers[k - 1].register_forward_hook(_hook)
    try:
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
    except _EarlyExit:
        pass  # intended
    finally:
        handle.remove()
    return captured["hidden"]


def _retained_slice(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of the per-sequence ``_trim_sequence`` repack.

    The reference path gathers each row's valid tokens and then trims
    ``[DROP_IDX : DROP_IDX + MAX_SEQUENCE_LENGTH]`` from that row-local
    sequence.  With right padding the valid tokens are a prefix, so the same
    rows and columns are obtained by slicing the padded tensor directly; the
    extra columns it keeps are exactly the ones the reference path fills with
    zeros, and they stay masked out of attention.  Rows with no retained token
    (a dropped caption) get the reference path's single zero token with
    ``mask=1`` so the DiT sees an identical attention denominator.
    """
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("expected hidden [B, S, D] and attention_mask [B, S]")
    batch_size, seq_len, hidden_dim = hidden.shape
    if attention_mask.shape[0] != batch_size:
        raise ValueError("hidden and attention_mask batch sizes must agree")

    if seq_len <= DROP_IDX:
        return (
            hidden.new_zeros((batch_size, RETAINED_MIN_LENGTH, hidden_dim)),
            hidden.new_ones((batch_size, RETAINED_MIN_LENGTH), dtype=torch.long),
        )

    # A single fixed column window is row-exact only because the tokenizer
    # pads on the right: each row's valid tokens form a prefix of the padded
    # row, so columns [DROP_IDX:end) line up identically for every row. With
    # left padding the valid tokens would start at a per-row offset and this
    # window could no longer stand in for the per-row trim below.
    end = min(seq_len, DROP_IDX + MAX_SEQUENCE_LENGTH)
    embeddings = hidden[:, DROP_IDX:end].clone()
    mask = attention_mask[:, DROP_IDX:end].to(torch.long).clone()

    empty = mask.sum(dim=1) == 0
    empty_row = empty.unsqueeze(1)
    embeddings = embeddings.masked_fill(empty_row.unsqueeze(-1), 0.0)
    mask = mask.masked_fill(empty_row, 0)
    mask[:, 0] = torch.where(
        empty, torch.ones_like(mask[:, 0]), mask[:, 0]
    )
    return embeddings, mask


def _encode_full_forward_slice(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    exit_layer: int,
) -> torch.Tensor:
    """Run the complete transformer and return ``hidden_states[exit_layer]``."""
    layers = model.model.layers
    k = int(exit_layer)
    if k <= 0 or k > len(layers):
        raise ValueError(
            f"exit_layer={k} out of range [1, {len(layers)}] (layers are 1-indexed"
            " for hidden-state selection; embedding output is not supported)"
        )

    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) <= k:
        raise RuntimeError("transformer did not return the requested hidden state")
    return hidden_states[k]


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return list(torch.split(selected, valid_lengths.tolist(), dim=0))


def _trim_sequence(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.size(0) <= DROP_IDX:
        return sequence.new_zeros((RETAINED_MIN_LENGTH, sequence.size(1)))
    end = DROP_IDX + MAX_SEQUENCE_LENGTH
    return sequence[DROP_IDX:end]


def _build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=text)


def encode_text(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    pooling: bool,
    exit_layer: Optional[int] = None,
    exit_mode: str = "full_forward_slice",
    fast_slice: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Encode captions with the Qwen3 chat template for DiT conditioning.

    Args:
        exit_layer: if set, select hidden states from this transformer layer
            (the embedding output is index 0).
        exit_mode: ``full_forward_slice`` runs all layers and selects
            ``hidden_states[exit_layer]``; ``stop_at_layer`` stops the forward
            after that layer. The default is ``full_forward_slice``.
        fast_slice: replace the per-sequence gather/pad repack with one slice of
            the padded hidden tensor. Same conditioning features; removes a
            device-to-host sync per call. Requires right padding.

    Returns:
        embeddings: [batch, seq, hidden]
        attention_mask: [batch, seq]
        pooled: [batch, hidden] when pooling is True
    """

    if not texts:
        raise ValueError("texts must contain at least one caption.")

    if exit_mode not in EXIT_MODES:
        raise ValueError(f"exit_mode must be one of {EXIT_MODES}, got {exit_mode!r}")

    if fast_slice and getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("fast_slice requires a right-padding tokenizer")

    prompts = [_build_prompt(text) for text in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH + DROP_IDX,
    ).to(model.device)

    with torch.no_grad():
        if exit_layer is None:
            outputs = model.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            hidden = outputs.last_hidden_state
        elif exit_mode == "full_forward_slice":
            hidden = _encode_full_forward_slice(
                model,
                inputs.input_ids,
                inputs.attention_mask,
                exit_layer,
            )
        else:
            hidden = _encode_early_exit(
                model,
                inputs.input_ids,
                inputs.attention_mask,
                exit_layer,
                use_cache=False,
            )

    if fast_slice:
        embeddings, attention_mask = _retained_slice(hidden, inputs.attention_mask)
        embeddings = embeddings.to(model.dtype)
    else:
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "Qwen/Qwen3-0.6B"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    texts = [
        "Impressionism landscape by Claude Monet",
        "romanticism marina by Van Gogh",
    ] * 2

    embedding, mask, pooled = encode_text(texts, model, tokenizer, True)

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

    sentinel = "__DROP_BOUNDARY__"
    sentinel_prompt = _build_prompt(sentinel)
    sentinel_ids = tokenizer(
        sentinel_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    ).input_ids[0]
    sentinel_token_ids = tokenizer(
        sentinel,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    ).input_ids[0]

    detected_drop_idx = None
    for i in range(0, sentinel_ids.shape[0] - sentinel_token_ids.shape[0] + 1):
        if torch.equal(sentinel_ids[i : i + sentinel_token_ids.shape[0]], sentinel_token_ids):
            detected_drop_idx = i
            break

    print(f"Configured DROP_IDX={DROP_IDX}, detected drop boundary={detected_drop_idx}")
    if detected_drop_idx != DROP_IDX:
        print("WARNING: DROP_IDX does not match detected boundary index!")

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
