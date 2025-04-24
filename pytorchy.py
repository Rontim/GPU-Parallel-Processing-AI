import torch

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version built with: {torch.version.cuda}")  # Crucial!
print(f"PyTorch cuDNN version built with: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")
    print(
        f"Detected PyTorch Device Name (GPU 0): {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch cannot access CUDA devices.")
