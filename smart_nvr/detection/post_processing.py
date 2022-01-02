from typing import Iterable, List

from .detection_types import Detection


def filter_detections(
    detections: List[Detection], names: Iterable[str]
) -> List[Detection]:
    names_set = set(names)
    return [detection for detection in detections if detection.name in names_set]
