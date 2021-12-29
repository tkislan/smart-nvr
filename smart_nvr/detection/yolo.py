from functools import reduce
from typing import List, Tuple, cast

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


def detect(model, img: CameraImageContainer, threshold: float = 0.5) -> List[Detection]:
    result = model(img.cropped_images)

    result.print()

    detections_list = [
        [
            adjust_cropped_detection(
                Detection(
                    name=result.names[int(raw_prediction[5])],
                    confidence=float(raw_prediction[4]),
                    xmin=float(raw_prediction[0]),
                    ymin=float(raw_prediction[1]),
                    xmax=float(raw_prediction[2]),
                    ymax=float(raw_prediction[3]),
                ),
                dimensions,
            )
            for raw_prediction in raw_predictions
            if float(raw_prediction[4]) > threshold
        ]
        for dimensions, raw_predictions in zip(img.dimensions, result.pred)
    ]

    detections = reduce(
        lambda acc, value: [*acc, *value], detections_list, cast(List[Detection], [])
    )

    return detections
