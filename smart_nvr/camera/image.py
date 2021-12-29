import time
from typing import List, Optional, Tuple

import numpy as np
from ..detection.detection_types import Detection


def split_image_into_squares(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width, _ = image.shape
    return (image[:, :height, :], image[:, width - height :, :])


def get_split_image_dimensions(
    image: np.ndarray,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    height, width, _ = image.shape
    return [
        ((0, height), (0, height)),
        ((0, height), (width - height, width)),
    ]


class CameraImageContainer:
    def __init__(
        self,
        camera_name: str,
        raw_image_np: np.ndarray,
        dimensions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
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
        dimensions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        detailed: bool = False,
        created_at: Optional[int] = None,
    ) -> "CameraImageContainer":
        if created_at is None:
            created_at = int(time.time() * 1000)

        cropped_images = [
            raw_image_np[
                crop_image_dimensions[0][0] : crop_image_dimensions[0][1],
                crop_image_dimensions[1][0] : crop_image_dimensions[1][1],
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
        # return True  # TODO - temporary hack
        return len(self.detections) > 0
