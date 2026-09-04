from fastapi.testclient import TestClient
from backend.main import app

def test_health():
    c=TestClient(app); r=c.get('/health'); assert r.status_code==200; assert r.json()['status']=='healthy'

def test_tools():
    c=TestClient(app); names=c.get('/tools').json()['tools']; assert 'vqa' in names; assert 'change_vqa' in names
