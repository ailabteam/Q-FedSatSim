import torch
import time

device = torch.device("cuda:0")  # GPU 0 đang rảnh
print(f"Using device: {torch.cuda.get_device_name(device)}")

x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)

torch.cuda.synchronize()
start = time.time()

for _ in range(10):
    z = torch.mm(x, y)

torch.cuda.synchronize()
end = time.time()

print(f"Elapsed time: {end - start:.2f} seconds")

