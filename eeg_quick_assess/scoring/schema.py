from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DimensionResult(BaseModel):
    score: float
    level: str
    evidence: List[str]
    fixed_text: str
    missing_features: Optional[List[str]] = None
    confidence: Optional[float] = None


class ModuleResult(BaseModel):
    module_name: str
    dimensions: Dict[str, DimensionResult]
    module_text_rule: str = ""


class FinalResult(BaseModel):
    final_report_text_llm: Optional[str] = None
    fallback_text: Optional[str] = None
    llm_used: bool = False
    disclaimer: str
