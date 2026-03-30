import os
import sys
import json
import traceback
import redis
from redis.exceptions import WatchError
from rq import get_current_job
from .models import JobStatus

# Ensure the root directory is in the path so we can import engine.core.controller
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from engine.core.controller import ApplicationController
from engine.core.logger import daemon_logger

# 1MB limit in bytes
MAX_ARTIFACT_SIZE_BYTES = 1024 * 1024 
ARTIFACT_DIR = os.path.join(root_dir, "artifacts")

def safe_commit_results(redis_client, job_id: str, result_str: str) -> bool:
    """
    Safely commits the final job state and artifacts to Redis.
    
    Uses Optimistic Locking (WATCH) via a Pipeline to prevent race conditions.
    """
    job_key = f"job:{job_id}"
    
    # Modern redis-py requires WATCH to be called inside a pipeline object
    with redis_client.pipeline() as pipe:
        try:
            # Monitor the job key
            pipe.watch(job_key)
            
            # Read the current state using the pipeline
            current_status = pipe.hget(job_key, "status")
            
            # Evaluate if the user hit Cancel while we were running
            if current_status in [JobStatus.CANCELLED.value, "CANCEL_REQUESTED"]:
                daemon_logger.info(f"Worker finished, but Job {job_id} was cancelled. Dropping artifacts.")
                pipe.unwatch()
                return False
                
            # Prepare the artifact
            artifact_value = ""
            result_bytes = result_str.encode('utf-8')
            
            if len(result_bytes) > MAX_ARTIFACT_SIZE_BYTES:
                os.makedirs(ARTIFACT_DIR, exist_ok=True)
                file_path = os.path.join(ARTIFACT_DIR, f"{job_id}.json")
                
                with open(file_path, "w") as f:
                    f.write(result_str)
                artifact_value = f"FILE_PATH:{file_path}"
                daemon_logger.info(f"Artifact > 1MB. Saved to disk: {file_path}")
            else:
                artifact_value = result_str

            # Start the atomic transaction
            pipe.multi()
            
            pipe.hset(job_key, mapping={
                "status": JobStatus.COMPLETED.value,
                "progress": "100.0",
                "artifact_path": artifact_value
            })
            
            # Execute fires the pipeline atomically
            pipe.execute()
            
            daemon_logger.info(f"Job {job_id} committed successfully.")
            return True

        except WatchError:
            daemon_logger.warning(f"Race condition detected for {job_id}. Aborting commit.")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            return False
            
        except Exception as e:
            daemon_logger.error(f"Failed to commit results for {job_id}: {e}")
            pipe.unwatch()
            raise e


def process_job(job_id: str, payload: dict):
    """
    The background task executed by RQ. 
    """
    job = get_current_job()
    redis_client = job.connection
    job_key = f"job:{job_id}"
    
    try:
        daemon_logger.info(f"Worker claimed Job: {job_id}")
        
        # 1. EARLY EXIT CHECK: Must happen BEFORE overwriting status
        current_status = redis_client.hget(job_key, "status")
        if current_status == "CANCEL_REQUESTED":
            redis_client.hset(job_key, "status", JobStatus.CANCELLED.value)
            daemon_logger.info(f"Job {job_id} cancelled before execution.")
            return

        # 2. UPDATE STATE: We are cleared to compute
        redis_client.hset(job_key, mapping={
            "status": JobStatus.RUNNING.value,
            "progress": "10.0"
        })

        controller = ApplicationController()
        
        # Simulate mid-job progress update
        redis_client.hset(job_key, "progress", "50.0")
        
        result = controller.execute_job(payload)
        
        result_str = json.dumps(result) if result else ""

        safe_commit_results(redis_client, job_id, result_str)
        
    except Exception as e:
        error_log = traceback.format_exc()
        daemon_logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        
        current_status = redis_client.hget(job_key, "status")
        if current_status not in [JobStatus.CANCELLED.value, "CANCEL_REQUESTED"]:
            redis_client.hset(job_key, mapping={
                "status": JobStatus.FAILED.value,
                "error_log": error_log
            })