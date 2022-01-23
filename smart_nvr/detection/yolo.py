from functools import reduce
from typing import List, Union, cast

from typing_extensions import Literal

from ..camera.image import CameraImageContainer
from .base_model import BaseDetectionModel, adjust_cropped_detection
from .detection_types import Detection

YoloModelName = Union[
    Literal["yolov5s"], Literal["yolov5m"], Literal["yolov5l"], Literal["yolov5x"]
]


class YoloDetectionModel(BaseDetectionModel):
    def model_name(
        self,
    ) -> YoloModelName:
        raise NotImplementedError()

    def load(self):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("ultralytics/yolov5", self.model_name())
        self._model = self._model.to(device)

    def detect(
        self, img: CameraImageContainer, threshold: float = 0.5
    ) -> List[Detection]:
        result = self._model(img.cropped_images)

        result.print()

        detections_list = [
            [
                adjust_cropped_detection(
                    Detection(
                        name=result.names[int(raw_prediction[5])],
                        confidence=float(raw_prediction[4]),
                        x1=int(raw_prediction[0]),
                        y1=int(raw_prediction[1]),
                        x2=int(raw_prediction[2]),
                        y2=int(raw_prediction[3]),
                    ),
                    dimensions,
                )
                for raw_prediction in raw_predictions
                if float(raw_prediction[4]) > threshold
            ]
            for dimensions, raw_predictions in zip(img.dimensions, result.pred)
        ]

        detections = reduce(
            lambda acc, value: [*acc, *value],
            detections_list,
            cast(List[Detection], []),
        )

        return detections


class YoloSDetectionModel(YoloDetectionModel):
    def model_name(self) -> YoloModelName:
        return "yolov5s"
