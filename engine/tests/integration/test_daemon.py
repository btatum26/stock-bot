import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from daemon.main import app
from daemon.models import JobStatus

client = TestClient(app)

def test_health_check(mock_redis_client):
    """Verify the API health check and Redis broker connectivity."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": True}


@patch("daemon.main.task_queue")
def test_submit_job(mock_task_queue, mock_redis_client, dummy_strategy):
    """Verify that a valid job payload passes Fail Fast validation and enters the queue."""
    payload = {
        "strategy": "dummy_strat",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST",
        "timeframe": {"start": "2023-01-01", "end": "2023-12-31"}
    }
    response = client.post("/api/v1/jobs", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    
    status = data["status"]
    assert status == JobStatus.QUEUED.value
    
    mock_task_queue.enqueue.assert_called_once()


@patch("daemon.main.task_queue")
def test_list_jobs(mock_task_queue, mock_redis_client, dummy_strategy):
    """Verify that the job listing endpoint correctly retrieves recently submitted jobs."""
    payload = {
        "strategy": "dummy_strat",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST"
    }
    
    # Submit a job to populate the Redis state machine
    submit_response = client.post("/api/v1/jobs", json=payload)
    assert submit_response.status_code == 200
        
    # Retrieve the list of jobs
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    
    jobs = response.json()
    assert len(jobs) >= 1
    assert jobs[0]["strategy_name"] == "dummy_strat"
    
    status = jobs[0]["status"]
    assert status == JobStatus.QUEUED.value