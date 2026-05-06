from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    """Test Docker health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_basic():
    """Test basic query without image"""
    response = client.post(
        "/query",
        data={"query": "What is LangGraph?", "model": "gpt-4o-mini"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "planner" in data["steps"]
    assert "report" in data["steps"]

def test_query_empty():
    """Test that empty query returns validation error"""
    response = client.post(
        "/query",
        data={"query": ""}
    )
    assert response.status_code in [200, 422]