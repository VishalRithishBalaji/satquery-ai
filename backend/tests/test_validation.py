from pathlib import Path
from PIL import Image
from backend.preprocessing.validation import validate_image


def test_validate_image(tmp_path: Path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), "white").save(path)
    meta = validate_image(str(path))
    assert meta["width"] == 32
    assert meta["height"] == 32
