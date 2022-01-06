import queue

from ..camera.image import DetectionCameraImageContainer
from ..video.output_file import OutputFile
from ..video.video_writer_manager import VideoWriterManager
from .base_worker import BaseWorker


class VideoWorker(BaseWorker):
    _output_file_queue: "queue.Queue[OutputFile]"

    def __init__(
        self,
        output_directory: str,
        detection_queue: "queue.Queue[DetectionCameraImageContainer]",
    ):
        super().__init__(name="VideoWorker")
        self._video_writer_manager = VideoWriterManager(output_directory)
        self._detection_queue = detection_queue
        self._output_file_queue = queue.Queue(10)

    @property
    def file_path_queue(self) -> "queue.Queue[OutputFile]":
        return self._output_file_queue

    def run_processing(self):
        try:
            img: DetectionCameraImageContainer = self._detection_queue.get(
                block=True, timeout=1
            )

            image_output_file = self._video_writer_manager.write_image(img)
            if image_output_file is not None:
                self._output_file_queue.put_nowait(image_output_file)
        except queue.Empty:
            pass
        except queue.Full:
            pass

        output_files = self._video_writer_manager.close_eligible_video_outputs()

        for output_file in output_files:
            print("Closed video writer:", output_file)
            try:
                self._output_file_queue.put_nowait(output_file)
            except queue.Full:
                pass

    def teardown(self):
        output_files = self._video_writer_manager.close_all_video_outputs()

        for output_file in output_files:
            print("Closed video writer:", output_file)
            self._output_file_queue.put(output_file)
