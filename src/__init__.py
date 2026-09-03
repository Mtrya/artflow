"""ArtFlow - Flow Matching DiT for Artistic Image Generation"""

__version__ = "0.1.0"
__all__ = ["ArtFlow", "ArtFlowPipeline", "ArtFlowPipelineOutput"]


def __getattr__(name):
    # lazy: keep data-only environments (fetch/clean/label) free of the torch stack
    if name == "ArtFlow":
        from .models.artflow import ArtFlow
        return ArtFlow
    if name in ("ArtFlowPipeline", "ArtFlowPipelineOutput"):
        from .pipeline.artflow_pipeline import ArtFlowPipeline, ArtFlowPipelineOutput
        return {"ArtFlowPipeline": ArtFlowPipeline,
                "ArtFlowPipelineOutput": ArtFlowPipelineOutput}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
