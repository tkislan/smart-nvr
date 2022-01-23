from typing import List

from ..camera.image import CameraImageContainer
from ..utils.rectangle import Rectangle
from .detection_types import Detection


def adjust_cropped_detection(detection: Detection, dimensions: Rectangle) -> Detection:
    return Detection(
        name=detection.name,
        confidence=detection.confidence,
        x1=detection.rectangle.x1 + dimensions.x1,
        y1=detection.rectangle.y1 + dimensions.y1,
        x2=detection.rectangle.x2 + dimensions.x1,
        y2=detection.rectangle.y2 + dimensions.y1,
    )


class BaseDetectionModel:
    def load(self):
        raise NotImplementedError()

    def detect(
        self, img: CameraImageContainer, threshold: float = 0.5
    ) -> List[Detection]:
        raise NotImplementedError()
