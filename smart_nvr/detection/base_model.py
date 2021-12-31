from typing import List, Tuple

from ..camera.image import CameraImageContainer
from .detection_types import Detection


def adjust_cropped_detection(
    detection: Detection, dimensions: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Detection:
    return Detection(
        name=detection.name,
        confidence=detection.confidence,
        xmin=detection.xmin + dimensions[1][0],
        ymin=detection.ymin + dimensions[0][0],
        xmax=detection.xmax + dimensions[1][0],
        ymax=detection.ymax + dimensions[0][0],
    )


class BaseDetectionModel:
    def load(self):
        raise NotImplementedError()

    def detect(
        self, img: CameraImageContainer, threshold: float = 0.5
    ) -> List[Detection]:
        raise NotImplementedError()
