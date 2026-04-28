import enum
import uuid
from pydantic import BaseModel, Field
from typing import Optional

class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobRegistry(BaseModel):
    # Using default_factory to generate a new UUID on instantiation
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    parameters: dict = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    error_log: Optional[str] = None
