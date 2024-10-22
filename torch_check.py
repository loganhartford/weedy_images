import torch
import torchvision

print(torch.cuda.is_available())    # Should return True
print(torch.cuda.device_count())    # Should show the number of GPUs available

print(torch.__version__)            # Should return cu124 version
print(torchvision.__version__)      # Should return cu124 version
