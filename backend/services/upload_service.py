from pathlib import Path
from uuid import uuid4
from ..config.settings import settings

def save_upload(filename: str, data: bytes) -> str:
    upload_dir=settings.output_path/'uploads'; upload_dir.mkdir(parents=True,exist_ok=True)
    safe=Path(filename).name; out=upload_dir/f'{uuid4().hex}_{safe}'; out.write_bytes(data); return str(out.resolve())
