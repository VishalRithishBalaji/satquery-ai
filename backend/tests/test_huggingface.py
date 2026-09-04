from huggingface_hub import whoami

def test_hf_auth():
    info = whoami()
    assert info
