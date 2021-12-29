import itertools
import queue
from typing import List

from .image import CameraImageContainer


class CameraFeedMultiplexer:
    def __init__(self, camera_feed_queues: List["queue.Queue[CameraImageContainer]"]):
        self._camera_feed_queues = camera_feed_queues
        self._camera_feed_queues_iterator = itertools.cycle(self._camera_feed_queues)

    def get(self, block=True, timeout=None) -> CameraImageContainer:
        camera_feed_queue = next(self._camera_feed_queues_iterator)
        img = camera_feed_queue.get(block, timeout)
        return img
