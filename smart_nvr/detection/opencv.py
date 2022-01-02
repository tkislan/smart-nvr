import os.path
import time
from typing import List, Tuple, Union

from typing_extensions import Literal

from ..camera.image import CameraImageContainer
from .base_model import BaseDetectionModel, adjust_cropped_detection
from .detection_types import Detection

SSDModelName = Union[
    Literal["ssd_mobilenet_v2_coco_2018_03_29"],
    Literal["ssdlite_mobilenet_v2_coco_2018_05_09"],
]

MODEL_DIR = os.path.join("data", "models")

# fmt: off
COCO_LABELS = {
    0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
    8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter",
    15: "zebra", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant",
    23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}
# fmt: on


class OpenCVTensorflowDetectionModel(BaseDetectionModel):
    def model_name(self) -> SSDModelName:
        raise NotImplementedError()

    def model_image_size(self) -> Tuple[int, int]:
        raise NotImplementedError()

    def load(self):
        import cv2

        model_path = os.path.join(
            MODEL_DIR, self.model_name(), "frozen_inference_graph.pb"
        )
        config_path = os.path.join(
            MODEL_DIR, self.model_name(), "frozen_inference_graph.pbtxt"
        )

        self._model = cv2.dnn.readNetFromTensorflow(model_path, config_path)

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("OpenCV DNN Using Cuda")
            self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(
        self, img: CameraImageContainer, threshold: float = 0.5
    ) -> List[Detection]:
        import cv2

        detections: List[Detection] = []

        for dimensions, cropped_image in zip(img.dimensions, img.cropped_images):
            t = int(time.perf_counter() * 1000)
            model_img = cv2.resize(cropped_image, self.model_image_size())
            # print(f"Img resize took {int(time.perf_counter() * 1000) - t} ms")

            t = int(time.perf_counter() * 1000)
            self._model.setInput(
                cv2.dnn.blobFromImage(
                    model_img, size=self.model_image_size(), swapRB=True
                )
            )
            # print(f"Model set input took {int(time.perf_counter() * 1000) - t} ms")

            t = int(time.perf_counter() * 1000)
            output = self._model.forward()
            # print(f"Model infer took {int(time.perf_counter() * 1000) - t} ms")

            for raw_prediction in output[0, 0, :, :]:
                confidence = float(raw_prediction[2])

                if confidence > threshold:
                    detections.append(
                        adjust_cropped_detection(
                            Detection(
                                name=COCO_LABELS.get(int(raw_prediction[1]), "unknown"),
                                confidence=confidence,
                                xmin=float(raw_prediction[3] * cropped_image.shape[1]),
                                ymin=float(raw_prediction[4] * cropped_image.shape[0]),
                                xmax=float(raw_prediction[5] * cropped_image.shape[1]),
                                ymax=float(raw_prediction[6] * cropped_image.shape[0]),
                            ),
                            dimensions,
                        )
                    )

        return detections


class OpenCVTensorflowSSDMobilenetDetectionModel(OpenCVTensorflowDetectionModel):
    def model_name(self) -> SSDModelName:
        return "ssd_mobilenet_v2_coco_2018_03_29"

    def model_image_size(self) -> Tuple[int, int]:
        return 300, 300


class OpenCVTensorflowSSDLiteMobilenetDetectionModel(OpenCVTensorflowDetectionModel):
    def model_name(self) -> SSDModelName:
        return "ssdlite_mobilenet_v2_coco_2018_05_09"

    def model_image_size(self) -> Tuple[int, int]:
        return 300, 300
