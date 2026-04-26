import json
import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from engine.core.controller import ExecutionMode
from .models import JobRegistry, JobStatus
from engine.core.logger import logger, daemon_logger
from engine.core.config import config

# Global variables for connection pools
redis_pool = None
redis_client = None
task_queue = None

STRATEGIES_DIR = config.STRATEGIES_FOLDER

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the API server."""
    global redis_pool, redis_client, task_queue
    
    logger.info(f"API Online at {config.api_url}")
    daemon_logger.info("FastAPI Server starting up... Connecting to Redis.")
    
    # Initialize Redis connection pool
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_pool = redis.ConnectionPool.from_url(redis_url, decode_responses=True)
    redis_client = redis.Redis(connection_pool=redis_pool)
    task_queue = Queue('default', connection=redis_client)
    
    # Ping Redis to ensure the connection is active before accepting requests
    redis_client.ping()
    daemon_logger.info("Connected to Redis successfully.")
    
    yield
    
    # Teardown logic
    redis_pool.disconnect()
    logger.info("API Offline.")
    daemon_logger.info("FastAPI Server shut down. Redis disconnected.")

app = FastAPI(title="Research Engine API", lifespan=lifespan)

class TimeframeRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

class JobPayloadRequest(BaseModel):
    strategy: str
    assets: List[str]
    interval: str
    mode: ExecutionMode
    timeframe: Optional[TimeframeRequest] = None
    multi_asset_mode: Optional[str] = "BATCH"

@app.get("/health")
def health_check():
    """Verify system status and Redis connectivity."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
    return {"status": "ok", "redis": redis_client.ping()}

@app.get("/api/v1/strategies")
def list_strategies():
    """
    Scans the local workspace for available strategies and returns their configurations.
    
    This endpoint iterates through the strategy directory, parses valid manifest.json 
    files, and attaches the folder name as a unique ID. It catches and logs missing 
    or malformed JSONs rather than crashing, ensuring the GUI always receives a 
    partial list if some files are corrupted.
    """
    strategies = []
    
    if not os.path.exists(STRATEGIES_DIR):
        daemon_logger.warning(f"Strategies directory not found at {STRATEGIES_DIR}")
        return []
        
    for entry in os.scandir(STRATEGIES_DIR):
        if entry.is_dir():
            manifest_path = os.path.join(entry.path, "manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest_data = json.load(f)
                        manifest_data["id"] = entry.name
                        strategies.append(manifest_data)
                except json.JSONDecodeError:
                    daemon_logger.warning(f"Malformed manifest.json in {entry.name}, skipping.")
                except Exception as e:
                    daemon_logger.error(f"Error reading manifest for {entry.name}: {e}")
                    
    return strategies

@app.post("/api/v1/jobs")
def submit_job(payload: JobPayloadRequest):
    """
    Validates job parameters, saves the state to Redis, and enqueues the background task.
    
    Implements a fail-fast architecture. It synchronously verifies that the requested 
    strategy folder exists and that assets are provided before ever touching the Redis 
    database. It utilizes Redis pipelines to update the multi-index architecture safely.
    """
    try:
        if task_queue is None or redis_client is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
            
        # Fail Fast Validation
        target_strat_path = os.path.join(STRATEGIES_DIR, payload.strategy)
        if not os.path.exists(target_strat_path) or not os.path.isdir(target_strat_path):
            raise HTTPException(status_code=400, detail=f"Strategy '{payload.strategy}' does not exist.")
            
        if not payload.assets:
            raise HTTPException(status_code=400, detail="Assets list cannot be empty.")
        
        job = JobRegistry(
            strategy_name=payload.strategy,
            parameters=payload.model_dump()
        )
        job_id = job.job_id
        timestamp = time.time()
        job_key = f"job:{job_id}"
        
        # Serialize nested dicts to strings for Redis Hashes
        mapping = {
            "job_id": job_id,
            "strategy_name": job.strategy_name,
            "status": job.status.value,
            "progress": str(job.progress),
            "parameters": json.dumps(job.parameters),
            "created_at": str(timestamp)
        }
        
        # Use a transaction pipeline to update the Hash and all ZSET indices simultaneously
        pipe = redis_client.pipeline()
        pipe.hset(job_key, mapping=mapping)
        pipe.zadd("jobs:all", {job_id: timestamp})
        pipe.zadd(f"jobs:status:{job.status.value}", {job_id: timestamp})
        pipe.execute()
        
        # Enqueue the background task
        from .tasks import process_job 
        task_queue.enqueue(
            process_job, 
            args=(job_id, job.parameters), 
            job_id=job_id, 
            job_timeout='1h'
        )
        
        daemon_logger.info(f"Job Queued: {job_id} for strategy {payload.strategy}")
        return {"job_id": job_id, "status": job.status}
        
    except HTTPException:
        raise
    except Exception as e:
        daemon_logger.error(f"Failed to submit job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    """Fetch the real-time state and metrics of a specific job."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
    
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Deserialize the parameters string back into a JSON object for the frontend
    if "parameters" in job_data:
        job_data["parameters"] = json.loads(job_data["parameters"])
        
    return job_data

@app.delete("/api/v1/jobs/{job_id}")
def cancel_job(job_id: str):
    """
    Attempts to gracefully or forcefully cancel a computational job.
    
    If the job is queued, it is purged from RQ immediately. If the job is currently 
    running, a cancellation flag is written to the Redis state machine, allowing the 
    worker to catch the signal during its next polling cycle and shut down safely.
    """
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
        
    job_key = f"job:{job_id}"
    current_status = redis_client.hget(job_key, "status")
    
    if not current_status:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Prevent operations on jobs that have already concluded
    if current_status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in terminal state: {current_status}")
        
    if current_status == JobStatus.QUEUED.value:
        # Purge the job from the RQ broker
        try:
            rq_job = Job.fetch(job_id, connection=redis_client)
            rq_job.cancel()
            rq_job.delete()
        except NoSuchJobError:
            daemon_logger.warning(f"RQ Job {job_id} not found during cancellation, proceeding to clean state.")
            
        # Migrate the job through the state machine ZSETs
        pipe = redis_client.pipeline()
        pipe.hset(job_key, "status", JobStatus.CANCELLED.value)
        pipe.zrem(f"jobs:status:{JobStatus.QUEUED.value}", job_id)
        pipe.zadd(f"jobs:status:{JobStatus.CANCELLED.value}", {job_id: time.time()})
        pipe.execute()
        
        return {"job_id": job_id, "status": "CANCELLED", "message": "Job removed from queue."}
        
    elif current_status == JobStatus.RUNNING.value:
        # Flag the job for cooperative cancellation by the worker
        redis_client.hset(job_key, "status", "CANCEL_REQUESTED")
        return {"job_id": job_id, "status": "CANCEL_REQUESTED", "message": "Kill signal sent to worker."}
        
    return {"job_id": job_id, "status": current_status}

@app.get("/api/v1/jobs")
def list_jobs(limit: int = 50, offset: int = 0, status: Optional[str] = None):
    """
    Return a paginated, reverse-chronological list of jobs.
    
    Leverages Redis ZSETs to fetch recent jobs efficiently. Allows the frontend 
    to filter by specific job statuses without iterating over the entire database.
    """
    try:
        if redis_client is None:
            raise HTTPException(status_code=503, detail="Redis client not initialized")
        
        # Route query to either the master list or a specific status index
        target_zset = f"jobs:status:{status}" if status else "jobs:all"
        
        # Fetch the chunk of job IDs based on pagination
        job_ids = redis_client.zrevrange(target_zset, offset, offset + limit - 1)
        
        if not job_ids:
            return []
            
        # Bulk fetch all corresponding Hashes in a single round-trip
        pipe = redis_client.pipeline()
        for j_id in job_ids:
            pipe.hgetall(f"job:{j_id}")
        jobs_data = pipe.execute()
        
        # Format payloads for the GUI
        formatted_jobs = []
        for job in jobs_data:
            if job:
                if "parameters" in job:
                    job["parameters"] = json.loads(job["parameters"])
                formatted_jobs.append(job)
                
        return formatted_jobs
    except Exception as e:
        daemon_logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")