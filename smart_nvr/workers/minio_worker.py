import logging
import os
import queue
from pathlib import PosixPath

from minio import Minio

from ..app_config import MinioConfig
from ..video.output_file import OutputFile
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


def get_object_name(output_file: OutputFile) -> str:
    file_name = os.path.basename(output_file.file_path)
    return str(
        PosixPath(output_file.file_type.value)
        / str(output_file.timestamp.year).zfill(2)
        / str(output_file.timestamp.month).zfill(2)
        / str(output_file.timestamp.day).zfill(2)
        / file_name
    )


class MinioWorker(BaseWorker):
    def __init__(
        self,
        config: MinioConfig,
        file_queue: "queue.Queue[OutputFile]",
    ):
        super().__init__(name="MinioWorker")
        self._file_queue = file_queue

        self._bucket_name = config.bucket

        try:
            self._minio_client = Minio(
                f"{config.host}:{config.port}",
                secure=config.secure,
                access_key=config.access_key,
                secret_key=config.secret_key,
            )
            if not self._minio_client.bucket_exists(self._bucket_name):
                self._minio_client.make_bucket(self._bucket_name)
        except Exception as error:
            logger.error(error)
            self._minio_client = None

    def run_processing(self):
        try:
            output_file = self._file_queue.get(block=True, timeout=1)

            self.upload_file(output_file)
        except queue.Empty:
            pass

    def upload_file(self, output_file: OutputFile):
        try:
            if self._minio_client is not None:
                object_name = get_object_name(output_file)

                logger.info(
                    f"Uploading file from {output_file.file_path} to {self._bucket_name}/{object_name}"
                )
                self._minio_client.fput_object(
                    bucket_name=self._bucket_name,
                    object_name=object_name,
                    file_path=output_file.file_path,
                )
        except Exception as error:
            logger.error(f"Failed to upload file to minio: {error}")
        finally:
            try:
                os.remove(output_file.file_path)
            except Exception as error:
                logger.error(f"Failed to remove local file: {error}")
