from __future__ import annotations
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
from .config.settings import settings
from .agent.registry import list_tools
from .models.geochat import GeoChatModel
from .services.upload_service import save_upload
from .services.analysis_service import analyze
from .services.report_service import save_json_report
from . import tools  # noqa: F401

settings.output_path.mkdir(parents=True,exist_ok=True); settings.report_path.mkdir(parents=True,exist_ok=True)
app=FastAPI(title=settings.app_name,version=settings.app_version,description='Agentic remote-sensing vision-language assistant')
class AnalyzeRequest(BaseModel):
    query:str=Field(min_length=1)
    image_paths:list[str]=Field(min_length=1,max_length=3)
@app.get('/')
def root(): return {'name':settings.app_name,'version':settings.app_version,'status':'running'}
@app.get('/health')
def health(): return {'status':'healthy','cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'tools':list_tools()}
@app.get('/tools')
def tools_endpoint(): return {'tools':list_tools()}
@app.get('/model-status')
def model_status(): return GeoChatModel.singleton(settings.model_id,settings.base_model_id,settings.model_max_pixels).info()
@app.post('/model-load')
def model_load():
    try: return GeoChatModel.singleton(settings.model_id,settings.base_model_id,settings.model_max_pixels).load().info()
    except Exception as e: raise HTTPException(500,detail=f'Model load failed: {type(e).__name__}: {e}')
@app.post('/model-unload')
def model_unload():
    model=GeoChatModel.singleton(settings.model_id,settings.base_model_id,settings.model_max_pixels); model.unload(); return model.info()
@app.post('/upload')
async def upload(files: Annotated[list[UploadFile],File(...)]):
    allowed={x.strip().lower() for x in settings.allowed_extensions.split(',')}; max_bytes=settings.max_upload_size_mb*1024*1024; paths=[]
    for f in files:
        suffix=Path(f.filename or '').suffix.lower()
        if suffix not in allowed: raise HTTPException(400,detail=f'Unsupported file type: {suffix}')
        data=await f.read()
        if not data: raise HTTPException(400,detail=f'Empty file: {f.filename}')
        if len(data)>max_bytes: raise HTTPException(413,detail=f'File too large: {f.filename}')
        paths.append(save_upload(f.filename or 'upload.bin',data))
    return {'files':paths}
@app.post('/analyze')
def analyze_endpoint(request:AnalyzeRequest):
    try:
        root=Path.cwd().resolve(); paths=[]
        for raw in request.image_paths:
            p=Path(raw).resolve();
            if not p.exists(): raise FileNotFoundError(raw)
            try: p.relative_to(root)
            except ValueError: raise PermissionError(f'Path outside project workspace: {raw}')
            paths.append(str(p))
        result=analyze(request.query,paths); result['report']=save_json_report(result); return result
    except Exception as e: raise HTTPException(400,detail=f'Analysis failed: {type(e).__name__}: {e}') from e
@app.get('/artifact/{path:path}')
def artifact(path:str):
    candidate=(settings.output_path/path).resolve()
    if settings.output_path.resolve() not in candidate.parents: raise HTTPException(403,detail='Invalid artifact path')
    if not candidate.exists() or not candidate.is_file(): raise HTTPException(404,detail='Artifact not found')
    return FileResponse(candidate)
