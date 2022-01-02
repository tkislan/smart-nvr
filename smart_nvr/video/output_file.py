from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class OutputFileType(str, Enum):
    image = "image"
    video = "video"


class OutputFile(BaseModel):
    file_type: OutputFileType
    file_path: str
    timestamp: datetime
