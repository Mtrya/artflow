"""Inko - Flow Matching DiT for Artistic Image Generation"""

__version__ = "0.1.0"
__all__ = ["Inko"]


def __getattr__(name):
    # lazy: keep data-only environments (fetch/clean/label) free of the torch stack
    if name == "Inko":
        from .models.inko import Inko
        return Inko
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
