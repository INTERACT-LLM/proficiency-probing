"""
Proficiency Probing: A pipeline for embedding texts, fitting a linear probe, 
and testing generalizability of the probe on other distributions.
"""

from .embedder import TextEmbedder
from .probe import OrdinalProbe
from .pipeline import ProficiencyProbingPipeline

__version__ = "0.1.0"
__all__ = ["TextEmbedder", "OrdinalProbe", "ProficiencyProbingPipeline"]
