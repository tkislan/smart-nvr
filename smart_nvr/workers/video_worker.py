import queue

from ..camera.image import DetectionCameraImageContainer
from ..video.video_writer import VideoWriterManager
from .base_worker import BaseWorker


class VideoWorker(BaseWorker):
    _file_path_queue: "queue.Queue[str]"

    def __init__(
        self,
        output_directory: str,
        detection_queue: "queue.Queue[DetectionCameraImageContainer]",
    ):
        super().__init__(name="VideoWorker")
        self._video_writer_manager = VideoWriterManager(output_directory)
        self._detection_queue = detection_queue
        self._file_path_queue = queue.Queue(10)

    @property
    def file_path_queue(self) -> "queue.Queue[str]":
        return self._file_path_queue

    def run_processing(self):
        try:
            img: DetectionCameraImageContainer = self._detection_queue.get(
                block=True, timeout=1
            )

            self._video_writer_manager.write_image(img)
        except queue.Empty:
            pass

        file_paths = self._video_writer_manager.close_eligible_video_writers()

        for file_path in file_paths:
            print("Closed video writer:", file_path)
            self._file_path_queue.put(file_path)

    def teardown(self):
        file_paths = self._video_writer_manager.close_all_video_writers()

        for file_path in file_paths:
            print("Closed video writer:", file_path)
            self._file_path_queue.put(file_path)
