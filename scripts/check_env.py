import sys
import torch
from huggingface_hub import whoami

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
try:
    print("Hugging Face:", whoami())
except Exception as exc:
    print("Hugging Face auth not available:", exc)
