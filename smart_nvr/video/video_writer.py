import datetime
import os.path
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import av

from ..camera.image import DetectionCameraImageContainer

VIDEO_MAX_TIME_MILLIS = 15 * 1000  # 15 seconds
VIDEO_MAX_NO_DETECTION_TIME_MILLIS = 5 * 1000  # 5 seconds


class VideoOutputFile:
    _first_frame_time: Optional[int]
    _last_detection_time: Optional[int]

    def __init__(self, file_path: str, width: int, height: int):
        self._file_path = file_path
        self._container, self._stream = self.create_video_output(
            file_path, width, height
        )
        self._first_frame_time = None
        self._last_detection_time = None

    @property
    def file_path(self) -> str:
        return self._file_path

    def is_max_time_running(self) -> bool:
        return (
            self._first_frame_time is not None
            and (time.time() * 1000) - self._first_frame_time >= VIDEO_MAX_TIME_MILLIS
        )

    def is_max_time_no_detection_running(self) -> bool:
        return (
            self._last_detection_time is not None
            and (time.time() * 1000) - self._last_detection_time
            >= VIDEO_MAX_NO_DETECTION_TIME_MILLIS
        )

    def write_image(self, img: DetectionCameraImageContainer):
        frame = av.VideoFrame.from_ndarray(
            img.camera_image_container.raw_image_np, format="rgb24"
        )

        if self._first_frame_time is None:
            frame.pts = 0
            self._first_frame_time = img.camera_image_container.created_at
        else:
            frame.pts = int(
                (
                    img.camera_image_container.created_at / 1000
                    - self._first_frame_time / 1000
                )
                / self._stream.codec_context.time_base
            )

        for packet in self._stream.encode(frame):
            self._container.mux(packet)

        if img.has_detections():
            self._last_detection_time = img.camera_image_container.created_at

    def close(self):
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    @staticmethod
    def create_video_output(file_path: str, width: int, height: int):
        container = av.open(file_path, mode="w")
        stream = container.add_stream("mpeg4", rate=25)  # alibi frame rate
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        stream.codec_context.time_base = Fraction(1, 1000)

        return container, stream


class VideoWriterManager:
    _output_directory: str
    _video_writers: Dict[str, VideoOutputFile]

    def __init__(self, output_directory: str):
        self._output_directory = output_directory
        self._video_writers = {}

    def write_image(self, img: DetectionCameraImageContainer):
        if not img.has_detections() and not self.has_video_writer(
            img.camera_image_container.camera_name
        ):
            return

        video_writer = self.get_video_writer(img)

        video_writer.write_image(img)

    def has_video_writer(self, camera_name: str) -> bool:
        return camera_name in self._video_writers

    def get_video_writer(self, img: DetectionCameraImageContainer) -> VideoOutputFile:
        camera_name = img.camera_image_container.camera_name
        height = img.camera_image_container.raw_image_np.shape[0]
        width = img.camera_image_container.raw_image_np.shape[1]

        video_writer = self._video_writers.get(camera_name)

        if video_writer is None:
            img_datetime = datetime.datetime.fromtimestamp(
                img.camera_image_container.created_at / 1000, datetime.timezone.utc
            )

            file_path = os.path.join(
                self._output_directory,
                f"{camera_name}_{img_datetime.strftime('%Y-%m-%dT%H%M%S')}.mp4",
            )
            video_writer = VideoOutputFile(file_path, width, height)
            print("Created video writer:", file_path, video_writer)
            self._video_writers[camera_name] = video_writer

        return video_writer

    @staticmethod
    def close_video_writers(
        video_writers: List[Tuple[str, VideoOutputFile]]
    ) -> List[Tuple[str, VideoOutputFile]]:
        closed_video_writers: List[Tuple[str, VideoOutputFile]] = []

        for name, video_writer in video_writers:
            video_writer.close()
            closed_video_writers.append((name, video_writer))

        return closed_video_writers

    def close_eligible_video_writers(self) -> List[str]:
        closed_video_writers = self.close_video_writers(
            [
                (name, video_worker)
                for name, video_worker in self._video_writers.items()
                if video_worker.is_max_time_running()
                or video_worker.is_max_time_no_detection_running()
            ]
        )

        for name, _ in closed_video_writers:
            del self._video_writers[name]

        return [video_writer.file_path for _, video_writer in closed_video_writers]

    def close_all_video_writers(self) -> List[str]:
        closed_video_writers = self.close_video_writers(
            list(self._video_writers.items())
        )

        for name, _ in closed_video_writers:
            del self._video_writers[name]

        return [video_writer.file_path for _, video_writer in closed_video_writers]
