import os
import json
import pytest
from unittest.mock import patch, MagicMock
from daemon.tasks import process_job

@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_worker_claim_state(mock_controller_class, mock_get_job, mock_redis_client):
    """Verify that the worker correctly updates the job status to RUNNING upon claiming it."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job

    state_during_execution = {}
    
    # Capture the Redis state exactly when the controller is executing
    def side_effect_execute(*args, **kwargs):
        state_during_execution.update(mock_redis_client.hgetall("job:test-1"))
        return {}

    mock_instance = MagicMock()
    mock_instance.execute_job.side_effect = side_effect_execute
    mock_controller_class.return_value = mock_instance

    process_job("test-1", {"some": "payload"})

    assert state_during_execution.get("status") == "RUNNING"
    assert state_during_execution.get("progress") == "50.0"


@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_worker_early_cancellation(mock_controller_class, mock_get_job, mock_redis_client):
    """Ensure the worker drops the job immediately if cancelled before execution begins."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    job_id = "test-cancel-early"
    job_key = f"job:{job_id}"
    
    # Simulate the API setting the status to CANCEL_REQUESTED while the job was queued
    mock_redis_client.hset(job_key, "status", "CANCEL_REQUESTED")
    
    process_job(job_id, {"some": "payload"})
    
    # Verify ApplicationController was never instantiated or executed
    mock_controller_class.return_value.execute_job.assert_not_called()
    
    # Verify the status finalized cleanly to CANCELLED
    assert mock_redis_client.hget(job_key, "status") == "CANCELLED"


@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_worker_late_cancellation_aborts_commit(mock_controller_class, mock_get_job, mock_redis_client):
    """Verify that if a user cancels a job while it is computing, the worker does not save artifacts."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    job_id = "test-cancel-late"
    job_key = f"job:{job_id}"
    
    # Simulate the user hitting cancel exactly while the strategy is executing
    def side_effect_execute(*args, **kwargs):
        mock_redis_client.hset(job_key, "status", "CANCEL_REQUESTED")
        return {"heavy": "data"}

    mock_instance = MagicMock()
    mock_instance.execute_job.side_effect = side_effect_execute
    mock_controller_class.return_value = mock_instance
    
    process_job(job_id, {"some": "payload"})
    
    job_data = mock_redis_client.hgetall(job_key)
    
    # The status should not be COMPLETED, and no artifact path should be written
    assert job_data.get("status") == "CANCEL_REQUESTED"
    assert "artifact_path" not in job_data


@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_artifact_under_1mb_redis_storage(mock_controller_class, mock_get_job, mock_redis_client):
    """Verify that small execution results are stored directly in the Redis hash."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    mock_instance = MagicMock()
    mock_instance.execute_job.return_value = {"test": "data"}
    mock_controller_class.return_value = mock_instance
    
    process_job("test-2", {})
    
    job_data = mock_redis_client.hgetall("job:test-2")
    assert job_data["status"] == "COMPLETED"
    assert job_data["progress"] == "100.0"
    assert job_data["artifact_path"] == json.dumps({"test": "data"})


@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_artifact_over_1mb_disk_spillover(mock_controller_class, mock_get_job, mock_redis_client, managed_artifact_dir):
    """Verify that execution results exceeding 1MB trigger a disk spillover response."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    # Generate massive payload to force disk spillover
    large_data = [{"key": "A" * 1000} for _ in range(2000)]
    
    mock_instance = MagicMock()
    mock_instance.execute_job.return_value = large_data
    mock_controller_class.return_value = mock_instance
    
    process_job("test-3", {})
    
    job_data = mock_redis_client.hgetall("job:test-3")
    assert job_data["status"] == "COMPLETED"
    assert job_data["artifact_path"].startswith("FILE_PATH:")


@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_artifact_over_1mb_file_creation(mock_controller_class, mock_get_job, mock_redis_client, managed_artifact_dir):
    """Verify that when a large payload spills to disk, the JSON file is fully valid and accessible."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    large_data = [{"key": "A" * 1000} for _ in range(2000)]
    
    mock_instance = MagicMock()
    mock_instance.execute_job.return_value = large_data
    mock_controller_class.return_value = mock_instance
    
    process_job("test-4", {})
    
    job_data = mock_redis_client.hgetall("job:test-4")
    file_path = job_data["artifact_path"].split("FILE_PATH:")[1]
    
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        loaded = json.load(f)
    assert loaded == large_data


@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_worker_exception_handling(mock_controller_class, mock_get_job, mock_redis_client):
    """Verify that internal strategy errors transition the job state to FAILED."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    mock_instance = MagicMock()
    mock_instance.execute_job.side_effect = RuntimeError("Data missing")
    mock_controller_class.return_value = mock_instance
    
    process_job("test-5", {})
    
    job_data = mock_redis_client.hgetall("job:test-5")
    assert job_data["status"] == "FAILED"


@patch("daemon.tasks.get_current_job")
@patch("daemon.tasks.ApplicationController")
def test_worker_traceback_logging(mock_controller_class, mock_get_job, mock_redis_client):
    """Verify that when a job fails, the full Python traceback is serialized to Redis for GUI rendering."""
    mock_job = MagicMock()
    mock_job.connection = mock_redis_client
    mock_get_job.return_value = mock_job
    
    mock_instance = MagicMock()
    mock_instance.execute_job.side_effect = RuntimeError("Data missing")
    mock_controller_class.return_value = mock_instance
    
    process_job("test-6", {})
    
    job_data = mock_redis_client.hgetall("job:test-6")
    assert "error_log" in job_data
    assert "RuntimeError: Data missing" in job_data["error_log"]
    assert "Traceback (most recent call last)" in job_data["error_log"]