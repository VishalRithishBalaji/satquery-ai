from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]

def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT_DIR / p

class Settings(BaseSettings):
    grounding_model: str = (
    "IDEA-Research/grounding-dino-base"
    )

    grounding_box_threshold: float = 0.35
    grounding_text_threshold: float = 0.25
    object_detection_model: str = "yolo26n-obb.pt"
    object_detection_confidence: float = 0.25
    object_detection_image_size: int = 1024
    app_name: str = 'SatQuery AI'
    app_env: str = 'development'
    app_version: str = '1.0.0'
    debug: bool = True
    host: str = '127.0.0.1'
    port: int = 8000
    hf_token: str | None = None
    model_id: str = 'tugrulkaya/GeoQwen-VL-2B-EuroSAT'
    base_model_id: str = 'Qwen/Qwen2-VL-2B-Instruct'
    agent_framework: str = 'langgraph'
    frontend_url: str = 'http://localhost:3000'
    max_upload_size_mb: int = 200
    allowed_extensions: str = '.tif,.tiff,.png,.jpg,.jpeg'
    max_image_side: int = 1536
    model_max_pixels: int = 401408
    max_new_tokens: int = 128
    temperature: float = 0.15
    confidence_threshold: float = 0.55
    output_dir: str = './datasets/outputs'
    report_dir: str = './reports'
    log_dir: str = './backend/logs'
    model_dir: str = './models'
    model_max_memory_gb: float = 5.0
    model_config = SettingsConfigDict(env_file=ROOT_DIR / '.env', env_file_encoding='utf-8', extra='ignore')
    @property
    def output_path(self): return resolve_path(self.output_dir)
    @property
    def report_path(self): return resolve_path(self.report_dir)
    @property
    def log_path(self): return resolve_path(self.log_dir)
    @property
    def model_path(self): return resolve_path(self.model_dir)
settings = Settings()
