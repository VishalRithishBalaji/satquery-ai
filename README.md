# SatQuery AI

Agentic remote-sensing vision-language assistant prototype.

## Runtime
- Python 3.12
- FastAPI
- LangGraph
- PyTorch / Transformers
- Rasterio / GeoPandas / OpenCV
- Next.js / React

## Current status
The scaffold implements the application plumbing: upload, validation, task routing, tool registry, execution trace, evidence contract, and reports. Specialist model wrappers are intentionally adapters/placeholders until their exact model checkpoints and inference APIs are selected.

## Start backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000/docs
# SatQuery AI Backend Runbook

## 1. Activate the existing virtual environment

```powershell
cd D:\satquery-ai\satquery-ai-starter
..\.venv\Scripts\Activate.ps1
```

If your `.venv` is inside this folder instead, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Install the remaining VLM runtime helper

```powershell
python -m pip install qwen-vl-utils
```

Refresh the requirements file after installation:

```powershell
python -m pip freeze > requirements.txt
```

## 3. Verify GPU and Hugging Face

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

```powershell
hf auth whoami
```

## 4. Start the backend

```powershell
python -m uvicorn backend.main:app --reload
```

Open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/model-status

## 5. Test the model

Upload an image with `/upload`, then send the returned path to `/analyze`.

Example PowerShell flow:

```powershell
$upload = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/upload -Form @{files=Get-Item .\datasets\sample\sample.png}
$upload | ConvertTo-Json
```

Then call `/analyze` using the returned path.

The first VQA/caption/change-VQA request downloads the base Qwen2-VL-2B-Instruct model and the GeoQwen-VL-2B-EuroSAT PEFT adapter, then loads them into the GPU. The prototype is designed to keep visual-token resolution bounded for a 6 GB RTX 4050.

## Runtime model

- Adapter: `tugrulkaya/GeoQwen-VL-2B-EuroSAT`
- Base model: `Qwen/Qwen2-VL-2B-Instruct`
- Runtime: PyTorch + Transformers + PEFT + qwen-vl-utils

The adapter is a remote-sensing model fine-tuned on EuroSAT; it supports satellite scene interpretation and geospatial VQA/caption-style use. For the SIH submission, document the remote-sensing adaptation dataset you actually use and later add the prescribed BigEarthNet-based adaptation run if your final training pipeline uses it.

## Mandatory prototype paths

- Single image VQA: enabled
- Single image captioning: enabled
- Bi-temporal change map: enabled (OpenCV pixel-difference prototype)
- Bi-temporal change VQA: enabled
- Optical-SAR prototype fusion: enabled
- Tool registry + agent routing: enabled
- Confidence + audit trace: enabled
- Dedicated grounding model: registered but not enabled in the lightweight runtime
- Dedicated object detector: registered but not enabled in the lightweight runtime



# Runbook

# SatQuery AI Backend Runbook

## 1. Activate the existing virtual environment

```powershell
cd D:\satquery-ai\satquery-ai-starter
..\.venv\Scripts\Activate.ps1
```

If your `.venv` is inside this folder instead, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Install the remaining VLM runtime helper

```powershell
python -m pip install qwen-vl-utils
```

Refresh the requirements file after installation:

```powershell
python -m pip freeze > requirements.txt
```

## 3. Verify GPU and Hugging Face

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

```powershell
hf auth whoami
```

## 4. Start the backend

```powershell
python -m uvicorn backend.main:app --reload
```

Open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/model-status

## 5. Test the model

Upload an image with `/upload`, then send the returned path to `/analyze`.

Example PowerShell flow:

```powershell
$upload = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/upload -Form @{files=Get-Item .\datasets\sample\sample.png}
$upload | ConvertTo-Json
```

Then call `/analyze` using the returned path.

The first VQA/caption/change-VQA request downloads the base Qwen2-VL-2B-Instruct model and the GeoQwen-VL-2B-EuroSAT PEFT adapter, then loads them into the GPU. The prototype is designed to keep visual-token resolution bounded for a 6 GB RTX 4050.

## Runtime model

- Adapter: `tugrulkaya/GeoQwen-VL-2B-EuroSAT`
- Base model: `Qwen/Qwen2-VL-2B-Instruct`
- Runtime: PyTorch + Transformers + PEFT + qwen-vl-utils

The adapter is a remote-sensing model fine-tuned on EuroSAT; it supports satellite scene interpretation and geospatial VQA/caption-style use. For the SIH submission, document the remote-sensing adaptation dataset you actually use and later add the prescribed BigEarthNet-based adaptation run if your final training pipeline uses it.

## Mandatory prototype paths

- Single image VQA: enabled
- Single image captioning: enabled
- Bi-temporal change map: enabled (OpenCV pixel-difference prototype)
- Bi-temporal change VQA: enabled
- Optical-SAR prototype fusion: enabled
- Tool registry + agent routing: enabled
- Confidence + audit trace: enabled
- Dedicated grounding model: registered but not enabled in the lightweight runtime
- Dedicated object detector: registered but not enabled in the lightweight runtime

