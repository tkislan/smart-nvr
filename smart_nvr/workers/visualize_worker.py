import queue

from ..camera.image import DetectionCameraImageContainer
from ..detection.visualizer import Visualizer
from .base_worker import BaseWorker


class VisualizeWorker(BaseWorker):
    _output_queue: "queue.Queue[DetectionCameraImageContainer]"

    def __init__(
        self,
        detection_queue: "queue.Queue[DetectionCameraImageContainer]",
    ):
        super().__init__(name="VisualizeWorker")
        self._detection_queue = detection_queue
        self._output_queue = queue.Queue(10)

    @property
    def output_queue(self) -> "queue.Queue[DetectionCameraImageContainer]":
        return self._output_queue

    def run_processing(self):
        try:
            img: DetectionCameraImageContainer = self._detection_queue.get(
                block=True, timeout=1
            )

            for detection in img.detections:
                Visualizer.draw_detection(
                    img.camera_image_container.raw_image_np, detection
                )

            self._output_queue.put(img)
        except queue.Empty:
            pass
