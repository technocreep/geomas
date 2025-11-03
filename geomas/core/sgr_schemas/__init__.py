"""
SGR (Schema-Guided Reasoning) schemas for geological analysis.

This module contains structured schemas that guide the reasoning process
for various geological analysis tasks using the SGR Deep Research framework.
"""

from .resource_assessment import RESOURCE_ASSESSMENT_SCHEMA
from .risk_analysis import RISK_ANALYSIS_SCHEMA
from .economic_viability import ECONOMIC_VIABILITY_SCHEMA

__all__ = [
    "RESOURCE_ASSESSMENT_SCHEMA",
    "RISK_ANALYSIS_SCHEMA",
    "ECONOMIC_VIABILITY_SCHEMA",
]

