import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..app_config import CameraFeedConfig

from ..camera.image import CameraImageContainer, get_split_image_dimensions
from ..camera.motion_detection.hikvision import HikvisionMotionDetection
from .base_worker import BaseWorker


class CameraFeedWorker(BaseWorker):
    _image_queue: "queue.Queue[CameraImageContainer]"

    def __init__(self, camera_name: str, config: CameraFeedConfig):
        super().__init__(name=f"CameraFeedWorker[{camera_name}]")
        self._camera_name = camera_name
        # self._rtstp_url = rtstp_url
        self._config = config
        self._should_read = threading.Event()
        self._motion_detection = HikvisionMotionDetection(config.motion)
        self._motion_detection.set_callback(self._handle_motion_changed)

        self._image_queue = queue.Queue(1)

    @property
    def image_queue(self) -> "queue.Queue[CameraImageContainer]":
        return self._image_queue

    def _handle_motion_changed(self, motion: bool):
        print(f"{self._camera_name} motion: {motion}")
        if motion:
            self.enable_read()
        else:
            self.disable_read()

    def start(self):
        super().start()
        self._motion_detection.start()

    def run_processing(self):
        cap: Optional[cv2.VideoCapture] = None

        try:
            if not self._should_read.wait(timeout=1):
                return

            # cap = cv2.VideoCapture(self._rtstp_url)
            cap = cv2.VideoCapture(self._config.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            time.sleep(1)

            while self._should_read.is_set():
                ret = cap.grab()

                if ret is not True:
                    print("Failed to grab image from camera")
                    time.sleep(5)
                    break

                if not self._image_queue.empty():
                    continue

                ret, raw_image_np = cap.retrieve()
                if ret is not True:
                    print("Failed to retrieve image from camera")
                    time.sleep(5)
                    break

                raw_image_np = raw_image_np[..., ::-1]  #  Convert from BGR to RGB
                raw_image_np = raw_image_np.astype(np.uint8)

                image_container = CameraImageContainer.create(
                    self._camera_name,
                    raw_image_np,
                    get_split_image_dimensions(raw_image_np),
                )

                try:
                    self._image_queue.put_nowait(image_container)
                except queue.Full:
                    pass
        except Exception as error:
            print("Camera feed failed")
            print(error)
            if cap is not None:
                cap.release()
            time.sleep(5)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception as error:
                print("Failed to release camera feed")
                print(error)

    def enable_read(self):
        if self._should_exit.is_set():
            raise RuntimeError("Can't enable read on destroyed CameraFeed")
        self._should_read.set()

    def disable_read(self):
        self._should_read.clear()

    def stop(self):
        self._motion_detection.stop()
        super().stop()
        self.disable_read()
