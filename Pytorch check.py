import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Get the number of GPUs available
num_gpus = torch.cuda.device_count()
print("Number of GPUs Available:", num_gpus)

# Print the name of the GPU(s)
if cuda_available:
    for i in range(num_gpus):
        print("GPU", i, ":", torch.cuda.get_device_name(i))
else:
    print("No GPU is available.")
