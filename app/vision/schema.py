from pydantic import BaseModel, Field

class VisionItem(BaseModel):
    name: str
    portion_estimate: str
    confidence: float
    alternatives: list[str] = Field(default_factory=list)

class VisionObservation(BaseModel):
    items: list[VisionItem]
    plate_context: str | None = None
    overall_confidence: float
    unclear: list[str] = Field(default_factory=list)
