from pydantic import BaseModel


class Detection(BaseModel):
    name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
