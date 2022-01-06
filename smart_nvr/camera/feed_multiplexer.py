from typing import List, Optional
from threading import Condition, Lock

from .image import CameraImageContainer


class CameraFeedMultiplexer:
    class Full(BaseException):
        pass

    class Empty(BaseException):
        pass

    _camera_images: List[CameraImageContainer]
    _mutex: Lock
    _cv: Condition

    def __init__(self):
        self._camera_images = []
        self._cv = Condition(Lock())

    def _contains_image(self, camera_name: str) -> bool:
        return (
            next(
                (img for img in self._camera_images if img.camera_name == camera_name),
                None,
            )
            is not None
        )

    def put_nowait(self, img: CameraImageContainer):
        with self._cv:
            if self._contains_image(img.camera_name):
                raise self.Full()

            self._camera_images.append(img)

            self._cv.notify(n=1)

    def get(self, timeout: Optional[float] = None) -> CameraImageContainer:
        with self._cv:
            try:
                return self._camera_images.pop()
            except IndexError:
                pass

            if timeout is None:
                raise self.Empty()

            if self._cv.wait(timeout):
                try:
                    return self._camera_images.pop()
                except IndexError:
                    # This should never happen, but just in case...
                    raise self.Empty()
            else:
                raise self.Empty()

    def contains(self, camera_name: str) -> bool:
        with self._cv:
            return self._contains_image(camera_name)
