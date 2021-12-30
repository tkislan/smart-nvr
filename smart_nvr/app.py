import threading
import time
from typing import List

import cv2
import torch
from smart_nvr.camera.feed_multiplexer import CameraFeedMultiplexer
from smart_nvr.camera.image import CameraImageContainer, get_split_image_dimensions
from smart_nvr.workers.base_worker import BaseWorker

from smart_nvr.workers.camera_feed_worker import CameraFeedWorker
from smart_nvr.workers.detection_worker import DetectionWorker, detect
from smart_nvr.workers.video_worker import VideoWorker
from smart_nvr.workers.visualize_worker import VisualizeWorker
from smart_nvr.app_config import ApplicationConfig


def warmup_model(model):
    img_raw = cv2.imread("data/cat.jpeg")
    img = CameraImageContainer.create(
        "warmup", img_raw, get_split_image_dimensions(img_raw)
    )
    detections = detect(model, img)
    assert "cat" in {d.name for d in detections}


def run():
    config = ApplicationConfig.load_from_file("config.yaml")
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5m"
    )  # or yolov5m, yolov5l, yolov5x, custom
    model = model.to(device)

    warmup_model(model)

    workers: List[BaseWorker] = []

    camera_workers = [
        CameraFeedWorker(camera_name, camera_config)
        for camera_name, camera_config in config.camera_feeds.items()
    ]
    workers.extend(camera_workers)

    detection_worker = DetectionWorker(
        model=model,
        image_queue=CameraFeedMultiplexer([c.image_queue for c in camera_workers]),
    )
    # Run detection worker in main thread

    visualizer_worker = VisualizeWorker(
        detection_queue=detection_worker.detection_queue,
    )
    workers.append(visualizer_worker)

    video_worker = VideoWorker(
        output_directory="output",
        detection_queue=visualizer_worker.output_queue,
    )
    workers.append(video_worker)

    for worker in workers:
        worker.start()

    def timeout_stop_detection_worker():
        time.sleep(60)
        detection_worker.stop()

    threading.Thread(target=timeout_stop_detection_worker).start()

    # RUN
    # time.sleep(30)
    detection_worker.run()

    for worker in workers:
        worker.stop()

    for worker in workers:
        worker.join()


if __name__ == "__main__":
    run()
    print("Done")
