import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("""
    GPU NOT DETECTED! Follow these steps:
    1. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
    2. Install CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
    3. Reinstall PyTorch with the exact command above
    """)