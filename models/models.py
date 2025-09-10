
from pydantic import BaseModel, Field
from typing import Literal, List, Dict


class TriageOutput(BaseModel):
    decision: Literal["AUTO_RESOLVE", "ASK_INFO", "OPEN_TICKET"]
    urgency: Literal["LOW", "MEDIUM", "HIGH"]
    missing_fields: List[str] = Field(default_factory=list)
