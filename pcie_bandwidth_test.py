import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

def measure_bandwidth(host_data, device_data, size, direction, iterations=10):
    # Record the start time
    start_time = time.time()

    for _ in range(iterations):
        if direction == 'HtoD':  # Host to Device (Upload/Upstream)
            cuda.memcpy_htod(device_data, host_data)
        elif direction == 'DtoH':  # Device to Host (Download/Downstream)
            cuda.memcpy_dtoh(host_data, device_data)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time and bandwidth
    elapsed_time = end_time - start_time
    bandwidth = (size * iterations) / elapsed_time / 1e9  # Bandwidth in GB/s

    return bandwidth

def main():
    # Define the size of the data in MB
    size_in_mb = 100
    size = size_in_mb * 1024 * 1024  # Convert MB to bytes

    # Create a host buffer (CPU)
    host_data = np.random.rand(size // 8).astype(np.float64)

    # Allocate device memory (GPU)
    device_data = cuda.mem_alloc(host_data.nbytes)

    # Measure HtoD (Host to Device - Upload) bandwidth
    h2d_bandwidth = measure_bandwidth(host_data, device_data, host_data.nbytes, direction='HtoD')
    print(f"Host to Device (Upstream) Bandwidth: {h2d_bandwidth:.2f} GB/s")

    # Measure DtoH (Device to Host - Download) bandwidth
    d2h_bandwidth = measure_bandwidth(host_data, device_data, host_data.nbytes, direction='DtoH')
    print(f"Device to Host (Downstream) Bandwidth: {d2h_bandwidth:.2f} GB/s")

if __name__ == "__main__":
    main()
