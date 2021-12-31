from typing import Dict, Type

from ..app_config import ModelNameEnum
from ..detection.base_model import BaseDetectionModel
from ..detection.opencv import (
    OpenCVTensorflowSSDLiteMobilenetDetectionModel,
    OpenCVTensorflowSSDMobilenetDetectionModel,
)
from ..detection.yolo import YoloSDetectionModel

MODEL_MAP: Dict[ModelNameEnum, Type[BaseDetectionModel]] = {
    ModelNameEnum.yolo_v5_s: YoloSDetectionModel,
    ModelNameEnum.tf_ssd_mobilenet_v2: OpenCVTensorflowSSDMobilenetDetectionModel,
    ModelNameEnum.tf_ssdlite_mobilenet_v2: OpenCVTensorflowSSDLiteMobilenetDetectionModel,
}
