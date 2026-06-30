"""small-council: a heterogeneous, verifier-checked multi-agent coder built on small LLMs."""

from small_council.config import CouncilConfig
from small_council.models import (
    CouncilResult,
    ExampleReport,
    ExampleResult,
    Plan,
    Usage,
    Verdict,
)

__all__ = [
    "CouncilConfig",
    "CouncilResult",
    "ExampleReport",
    "ExampleResult",
    "Plan",
    "Usage",
    "Verdict",
]
__version__ = "0.1.0"
