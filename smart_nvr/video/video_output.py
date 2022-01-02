from datetime import datetime
import time
from fractions import Fraction
from typing import Optional

import av

from ..camera.image import DetectionCameraImageContainer

VIDEO_MAX_TIME_MILLIS = 15 * 1000  # 15 seconds
VIDEO_MAX_NO_DETECTION_TIME_MILLIS = 5 * 1000  # 5 seconds


class VideoOutput:
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

    @property
    def timestamp(self) -> datetime:
        if self._first_frame_time is None:
            raise ValueError("first_frame_time not defined")
        return datetime.fromtimestamp(self._first_frame_time / 1000)

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
