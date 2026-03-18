from pydantic import BaseModel, field_validator
from typing import Literal


class Segment(BaseModel):
    start: float
    end: float

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v, info):
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start")
        return v


class CaptionConfig(BaseModel):
    enabled: bool = True
    style: Literal["simple", "bold", "animated"] = "bold"
    position: Literal["top", "center", "bottom"] = "bottom"


class TextOverlay(BaseModel):
    text: str
    start: float
    end: float
    position: Literal["top", "center", "bottom"] = "top"


class EditPlan(BaseModel):
    segments: list[Segment]
    captions: CaptionConfig
    overlays: list[TextOverlay] = []


class EditPlans(BaseModel):
    plans: list[EditPlan]
