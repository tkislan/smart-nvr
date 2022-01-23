from collections import defaultdict
from functools import reduce
from typing import Dict, List

from ..utils.rectangle import Rectangle


class Detection:
    name: str
    confidence: float
    rectangle: Rectangle

    def __init__(
        self, name: str, confidence: float, x1: int, y1: int, x2: int, y2: int
    ):
        self.name = name
        self.confidence = confidence
        self.rectangle = Rectangle(x1, y1, x2, y2)

    def __repr__(self) -> str:
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(kws)})"


def outer_box_detections(detections: List[Detection]) -> Detection:
    name = detections[0].name
    confidence = max([d.confidence for d in detections])

    return Detection(
        name,
        confidence,
        min([d.rectangle.x1 for d in detections]),
        min([d.rectangle.y1 for d in detections]),
        min([d.rectangle.x2 for d in detections]),
        min([d.rectangle.y2 for d in detections]),
    )


def merge_detections(
    detections: List[Detection], overlap_ratio_threshold: float = 0.65
) -> List[Detection]:
    detection_groups_map: Dict[str, List[List[Detection]]] = defaultdict(list)

    for detection in detections:
        detection_grouped = False
        detection_groups = detection_groups_map[detection.name]

        for detection_group in detection_groups:
            for other_detection in detection_group:
                smaller_area = min(
                    detection.rectangle.area, other_detection.rectangle.area
                )
                if (
                    detection.rectangle.overlap_area(other_detection.rectangle)
                    > overlap_ratio_threshold * smaller_area
                ):
                    detection_group.append(detection)
                    detection_grouped = True
                    break

            if detection_grouped:
                break

        if not detection_grouped:
            detection_groups.append([detection])

        detection_groups_map[detection.name] = sorted(
            detection_groups, key=lambda detection_group: len(detection_group)
        )

    detection_groups = reduce(
        lambda acc, detection_groups: [*acc, *detection_groups],
        detection_groups_map.values(),
        [],
    )

    return [
        outer_box_detections(detection_group) for detection_group in detection_groups
    ]
