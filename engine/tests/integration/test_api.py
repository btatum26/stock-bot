import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from daemon.main import app

client = TestClient(app)

@pytest.fixture
def mock_strategy_env(dummy_strategy):
    """
    Patches the global STRATEGIES_DIR in daemon/main.py to point to the temporary
    directory created by the dummy_strategy fixture. This ensures the Fail Fast 
    validation passes during testing.
    """
    base_dir = os.path.dirname(dummy_strategy)
    with patch("daemon.main.STRATEGIES_DIR", base_dir):
        yield base_dir


def test_health_endpoint(mock_redis_client):
    """Verify the health check endpoint properly pings Redis."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": True}


def test_submit_invalid_enum(mock_redis_client):
    """Ensure Pydantic catches invalid execution modes before processing."""
    payload = {
        "strategy": "dummy_strat",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "MAGIC"
    }
    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 422


def test_submit_invalid_multi_asset_mode(mock_redis_client):
    """Ensure Pydantic catches invalid multi-asset mode values."""
    payload = {
        "strategy": "dummy_strat",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST",
        "multi_asset_mode": "MAGIC",
    }
    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 422


@patch("daemon.main.task_queue")
def test_submit_rejects_strategy_traversal(mock_task_queue, mock_redis_client):
    """Ensure strategy names cannot escape STRATEGIES_DIR."""
    payload = {
        "strategy": "../outside",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST",
    }
    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid strategy name."
    mock_task_queue.enqueue.assert_not_called()


@patch("daemon.main.task_queue")
def test_submit_invalid_strategy_fails_fast(mock_task_queue, mock_redis_client, mock_strategy_env):
    """Test that a non-existent strategy directory gets blocked synchronously."""
    payload = {
        "strategy": "does_not_exist", 
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST"
    }
    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


@patch("daemon.main.task_queue")
def test_submit_success_redis_hash(mock_task_queue, mock_redis_client, mock_strategy_env):
    """Verify that a successful submission correctly formats the Redis Hash."""
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
    job_id = data["job_id"]
    
    redis_data = mock_redis_client.hgetall(f"job:{job_id}")
    assert redis_data["status"] == "QUEUED"
    
    parameters_str = redis_data["parameters"]
    parameters_dict = json.loads(parameters_str)
    assert parameters_dict["strategy"] == "dummy_strat"
    
    mock_task_queue.enqueue.assert_called_once()


@patch("daemon.main.task_queue")
def test_submit_success_redis_zset(mock_task_queue, mock_redis_client, mock_strategy_env):
    """Verify that a successful submission updates the multi-index ZSETs."""
    payload = {
        "strategy": "dummy_strat",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "BACKTEST"
    }
    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    master_score = mock_redis_client.zscore("jobs:all", job_id)
    assert master_score is not None
    assert isinstance(master_score, float)
    
    status_score = mock_redis_client.zscore("jobs:status:QUEUED", job_id)
    assert status_score is not None
    assert isinstance(status_score, float)


def test_get_job_404(mock_redis_client):
    """Ensure fetching a non-existent job returns a proper 404."""
    response = client.get("/api/v1/jobs/fake-uuid")
    assert response.status_code == 404


def test_get_job_deserialization(mock_redis_client):
    """Verify that the API correctly unpacks stringified JSON parameters from Redis."""
    job_id = "test-job-123"
    params = {"key": "value"}
    mapping = {
        "job_id": job_id,
        "parameters": json.dumps(params)
    }
    mock_redis_client.hset(f"job:{job_id}", mapping=mapping)
    
    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["job_id"] == job_id
    assert isinstance(data["parameters"], dict)
    assert data["parameters"]["key"] == "value"


def test_cancel_queued_job(mock_redis_client):
    """Verify the state machine transition when cancelling a queued job."""
    job_id = "cancel-test-123"
    
    mock_redis_client.hset(f"job:{job_id}", mapping={"status": "QUEUED"})
    mock_redis_client.zadd("jobs:status:QUEUED", {job_id: 100.0})
    
    response = client.delete(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLED"
    
    assert mock_redis_client.hget(f"job:{job_id}", "status") == "CANCELLED"
    assert mock_redis_client.zscore("jobs:status:QUEUED", job_id) is None
    assert mock_redis_client.zscore("jobs:status:CANCELLED", job_id) is not None


def test_cancel_running_job_updates_status_indexes(mock_redis_client):
    """Verify a running job moves from RUNNING to CANCEL_REQUESTED indexes."""
    job_id = "cancel-running-test-123"

    mock_redis_client.hset(f"job:{job_id}", mapping={"status": "RUNNING"})
    mock_redis_client.zadd("jobs:status:RUNNING", {job_id: 100.0})

    response = client.delete(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "CANCEL_REQUESTED"

    assert mock_redis_client.hget(f"job:{job_id}", "status") == "CANCEL_REQUESTED"
    assert mock_redis_client.zscore("jobs:status:RUNNING", job_id) is None
    assert mock_redis_client.zscore("jobs:status:CANCEL_REQUESTED", job_id) is not None


def test_get_strategies_endpoint(mock_strategy_env):
    """Verify the endpoint correctly scans and parses valid strategy manifests."""
    response = client.get("/api/v1/strategies")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 1
    assert data[0]["name"] == "Dummy Strategy"
    assert data[0]["id"] == "dummy_strat" 


def test_list_jobs_empty_state(mock_redis_client):
    """Ensure listing jobs returns an empty array when Redis is bare."""
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_jobs_pagination_limit(mock_redis_client):
    """Verify the limit parameter restricts the payload size correctly."""
    for i in range(10):
        job_id = f"job-{i}"
        mock_redis_client.hset(f"job:{job_id}", "job_id", job_id)
        mock_redis_client.zadd("jobs:all", {job_id: float(i)})
        
    response = client.get("/api/v1/jobs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5


def test_list_jobs_pagination_ordering(mock_redis_client):
    """Verify that jobs are returned in reverse-chronological order based on ZSET score."""
    mock_redis_client.hset("job:job1", mapping={"job_id": "job1"})
    mock_redis_client.zadd("jobs:all", {"job1": 100.0})
    
    mock_redis_client.hset("job:job2", mapping={"job_id": "job2"})
    mock_redis_client.zadd("jobs:all", {"job2": 200.0})
    
    mock_redis_client.hset("job:job3", mapping={"job_id": "job3"})
    mock_redis_client.zadd("jobs:all", {"job3": 300.0})
    
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data) == 3
    assert data[0]["job_id"] == "job3"
    assert data[1]["job_id"] == "job2"
    assert data[2]["job_id"] == "job1"
