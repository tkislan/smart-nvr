from dataclasses import dataclass


@dataclass
class Detection:
    name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
