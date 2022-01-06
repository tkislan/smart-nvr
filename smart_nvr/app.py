import logging
import os
import signal
from typing import List

import cv2

from smart_nvr.app_config import ApplicationConfig
from smart_nvr.camera.feed_multiplexer import CameraFeedMultiplexer
from smart_nvr.camera.image import CameraImageContainer, get_split_image_dimensions
from smart_nvr.detection.base_model import BaseDetectionModel
from smart_nvr.detection.model_map import MODEL_MAP
from smart_nvr.workers.base_worker import BaseWorker
from smart_nvr.workers.camera_feed_worker import CameraFeedWorker
from smart_nvr.workers.detection_worker import DetectionWorker
from smart_nvr.workers.minio_worker import MinioWorker
from smart_nvr.workers.video_worker import VideoWorker
from smart_nvr.workers.visualize_worker import VisualizeWorker

logger = logging.getLogger(__name__)


def warmup_model(model: BaseDetectionModel):
    img_raw = cv2.imread("data/images/cat.jpeg")
    img = CameraImageContainer.create(
        "warmup", img_raw, get_split_image_dimensions(img_raw)
    )
    detections = model.detect(img)
    assert "cat" in {d.name for d in detections}


def run():
    config = ApplicationConfig.load_from_file("/config/app.yaml")
    logger.info(config)

    model_cls = MODEL_MAP[config.model.name]
    model = model_cls()
    model.load()
    warmup_model(model)

    workers: List[BaseWorker] = []

    camera_feed_multiplexer = CameraFeedMultiplexer()

    camera_workers = [
        CameraFeedWorker(camera_name, camera_feed_multiplexer, camera_config)
        for camera_name, camera_config in config.camera_feeds.items()
    ]
    workers.extend(camera_workers)

    detection_worker = DetectionWorker(
        model=model,
        image_queue=camera_feed_multiplexer,
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

    minio_worker = MinioWorker(
        config=config.minio, file_queue=video_worker.file_path_queue
    )
    workers.append(minio_worker)

    for worker in workers:
        worker.start()

    def stop_detection_worker(signum, frame):
        detection_worker.stop()

    signal.signal(signal.SIGINT, stop_detection_worker)
    signal.signal(signal.SIGTERM, stop_detection_worker)

    # RUN
    detection_worker.run()

    for worker in workers:
        worker.stop()

    for worker in workers:
        worker.join()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname).1s] %(message)s",
        level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    )

    logger.info("Starting")
    run()
    logger.info("Done")
