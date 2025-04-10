import pycuda.driver as cuda

# Initialize CUDA driver
cuda.init()

# Get the number of GPUs available
gpu_count = cuda.Device.count()
print(f"Number of GPUs: {gpu_count}")

# Print details about each GPU
for i in range(gpu_count):
    gpu = cuda.Device(i)
    print(f"GPU {i}: {gpu.name()}")
    print(f"  Total Memory: {gpu.total_memory() // (1024**2)} MB")
    print(f"  Compute Capability: {gpu.compute_capability()}")
    print(f"  Memory Bus: {gpu.memory_bus_width()} bits")
    print(f"  Clock Rate: {gpu.clock_rate() / 1000} MHz")
    print("-" * 30)
