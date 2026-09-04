import json
from ..config.settings import settings

def save_json_report(result):
    settings.report_path.mkdir(parents=True,exist_ok=True)
    out=settings.report_path/f"{result['run_id']}.json"; out.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8'); return str(out.resolve())
