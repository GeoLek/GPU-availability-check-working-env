import pycuda.driver as cuda
def get_gpu_info():
    cuda.init()
    device_count = cuda.Device.count()

    for i in range(device_count):
        gpu = cuda.Device(i)
        print(f"Device #{i}: {gpu.name()}")
        print(f"  Compute Capability: {gpu.compute_capability()}")
        print(f"  Total Memory: {gpu.total_memory() // (1024 ** 2)} MB")
        attributes = gpu.get_attributes()
        print(f"  Multiprocessors: {attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
        cuda_cores_per_multiprocessor = {
            (5, 0): 128,  # Maxwells
            (6, 1): 128,  # Pascals
            (7, 0): 64,  # Volta
            (7, 5): 64,  # Turing
            (8, 0): 64,  # Ampere
            (8, 6): 128  # Ampere (RTX 3080)
        }.get(gpu.compute_capability(), 128)  # Default to 128 if not listed

        print(f"  CUDA Cores per Multiprocessor: {cuda_cores_per_multiprocessor}")
        print(
            f"  Total CUDA Cores: {attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT] * cuda_cores_per_multiprocessor}")
        # Note: Tensor Cores are not directly exposed via pycuda attributes; this information might not be accurately retrievable via pycuda.

get_gpu_info()
