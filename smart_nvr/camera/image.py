from typing import List, Optional, Tuple

import numpy as np

from ..detection.detection_types import Detection
from ..utils.rectangle import Rectangle
from ..utils.timing import get_current_time_millis


def split_image_into_squares(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width, _ = image.shape
    return (image[:, :height, :], image[:, width - height :, :])


def get_split_image_dimensions(
    image: np.ndarray,
) -> List[Rectangle]:
    height, width, _ = image.shape
    return [
        Rectangle(0, 0, height, height),
        Rectangle(width - height, 0, width, height),
    ]


class CameraImageContainer:
    def __init__(
        self,
        camera_name: str,
        raw_image_np: np.ndarray,
        dimensions: List[Rectangle],
        cropped_images: List[np.ndarray],
        detailed: bool,
        created_at: int,
    ):
        self.camera_name = camera_name
        self.raw_image_np = raw_image_np
        self.dimensions = dimensions
        self.cropped_images = cropped_images
        self.detailed = detailed
        self.created_at = created_at

    @classmethod
    def create(
        cls,
        camera_name: str,
        raw_image_np: np.ndarray,
        dimensions: List[Rectangle],
        detailed: bool = False,
        created_at: Optional[int] = None,
    ) -> "CameraImageContainer":
        if created_at is None:
            created_at = get_current_time_millis()

        cropped_images = [
            raw_image_np[
                crop_image_dimensions.y1 : crop_image_dimensions.y2,
                crop_image_dimensions.x1 : crop_image_dimensions.x2,
                :,
            ]
            for crop_image_dimensions in dimensions
        ]

        return cls(
            camera_name, raw_image_np, dimensions, cropped_images, detailed, created_at
        )


class DetectionCameraImageContainer:
    camera_image_container: CameraImageContainer
    detections: List[Detection]

    def __init__(
        self, camera_image_container: CameraImageContainer, detections: List[Detection]
    ):
        self.camera_image_container = camera_image_container
        self.detections = detections

    def has_detections(self):
        return len(self.detections) > 0
