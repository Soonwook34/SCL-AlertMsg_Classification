import torch
x = torch.rand(5, 3)
print(x)
print(f"CUDA 사용가능 여부: {torch.cuda.is_available()}")
print(f"CUDA 현재 GPU 번호: {torch.cuda.current_device()}")
print(f"CUDA에서 사용가능한 GPU 수: {torch.cuda.device_count()}")
print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

