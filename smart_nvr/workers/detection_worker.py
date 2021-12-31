import queue

from ..camera.feed_multiplexer import CameraFeedMultiplexer
from ..camera.image import DetectionCameraImageContainer
from ..detection.base_model import BaseDetectionModel
from ..detection.post_processing import filter_detections
from .base_worker import BaseWorker


class DetectionWorker(BaseWorker):
    _detection_queue: "queue.Queue[DetectionCameraImageContainer]"

    def __init__(
        self,
        model: BaseDetectionModel,
        image_queue: CameraFeedMultiplexer,
    ):
        super().__init__(name="DetectionWorker")
        self._model = model
        self._image_queue = image_queue
        self._detection_queue = queue.Queue(10)
        self._detection_names = ["person", "car", "cat"]

    @property
    def detection_queue(self) -> "queue.Queue[DetectionCameraImageContainer]":
        return self._detection_queue

    def run_processing(self):
        try:
            img = self._image_queue.get(block=True, timeout=1)

            detections = self._model.detect(img)

            detections = filter_detections(
                detections,
                self._detection_names,
            )

            self._detection_queue.put(DetectionCameraImageContainer(img, detections))
        except queue.Empty:
            pass
