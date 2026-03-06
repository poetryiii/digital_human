from pydantic import BaseModel
from typing import Optional, Literal

class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: int = 0
    video_url: Optional[str] = None
    error_msg: Optional[str] = None

class GenerationRequest(BaseModel):
    text: str
    voice_name: str