"""Utility helpers with lazy imports for data-only environments."""

__all__ = ["encode_text", "encode_image", "decode_latents", "get_vae_stats"]

_LAZY = {
    "encode_text": (".encode_text", "encode_text"),
    "encode_image": (".vae_codec", "encode_image"),
    "decode_latents": (".vae_codec", "decode_latents"),
    "get_vae_stats": (".vae_codec", "get_vae_stats"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attribute = _LAZY[name]
        return getattr(importlib.import_module(module_name, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
