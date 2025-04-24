import streamlit as st
import torch
import numpy as np
import cupy as cp
import time
import platform
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="GPU Profiling Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if CUDA is available (both PyTorch and CuPy)
try:
    pytorch_cuda_available = torch.cuda.is_available()
except Exception as e:
    st.error(f"Error checking PyTorch CUDA availability: {e}")
    pytorch_cuda_available = False

try:
    cupy_available = cp.is_available()
except Exception as e:
    st.error(f"Error checking CuPy availability: {e}")
    cupy_available = False

# App title and description
st.title("🚀 GPU Detection & Profiling Dashboard")
st.markdown("""
This dashboard presents a comprehensive analysis of your system's GPU capabilities,
memory performance, and processing power. Use this information to optimize your
deep learning and data science workflows.
""")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a section:",
        ["System Information", "GPU Detection", "Memory Profile",
            "Performance Benchmark", "Custom Operations", "Comparison", "Benchmark Summary"]
    )

    st.header("About")
    st.info("""
    This application provides diagnostics about your GPU setup and performance metrics.
    Perfect for understanding your hardware capabilities for AI and ML workloads.
    """)

# System Information Section
if page == "System Information":
    st.header("💻 System Information")

    # Function to get system info
    def get_system_info():
        system_info = {
            "Operating System": f"{platform.system()} {platform.release()} ({platform.version()})",
            "Architecture": platform.architecture()[0],
            "Processor": platform.processor(),
            "CPU Cores (Physical)": psutil.cpu_count(logical=False),
            "CPU Cores (Logical)": psutil.cpu_count(logical=True),
            "Total RAM": f"{round(psutil.virtual_memory().total / (1024.0 ** 3), 2)} GB",
            "Python Version": platform.python_version(),
            "NumPy Version": np.__version__,
            "PyTorch Version": torch.__version__,
            "CuPy Version": cp.__version__,
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return system_info

    # Get system info
    system_info = get_system_info()

    # Display system info in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hardware")
        st.metric("Operating System", system_info["Operating System"])
        st.metric("Architecture", system_info["Architecture"])
        st.metric("Processor", system_info["Processor"])
        st.metric("CPU Cores (Physical)", system_info["CPU Cores (Physical)"])
        st.metric("CPU Cores (Logical)", system_info["CPU Cores (Logical)"])
        st.metric("Total RAM", system_info["Total RAM"])

    with col2:
        st.subheader("Software")
        st.metric("Python Version", system_info["Python Version"])
        st.metric("NumPy Version", system_info["NumPy Version"])
        st.metric("PyTorch Version", system_info["PyTorch Version"])
        st.metric("CuPy Version", system_info["CuPy Version"])
        st.metric("Date & Time", system_info["Date & Time"])

    # RAM Usage
    st.subheader("RAM Usage")
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total RAM", f"{ram_total_gb:.2f} GB")
    col2.metric("Used RAM", f"{ram_used_gb:.2f} GB")
    col3.metric("RAM Usage", f"{ram.percent}%")

    # RAM usage progress bar
    st.progress(ram.percent / 100)

# GPU Detection Section
elif page == "GPU Detection":
    st.header("🖥️ GPU Detection")

    # Function to detect GPUs
    def detect_gpu():
        gpu_info = []

        # Check PyTorch CUDA
        if pytorch_cuda_available:
            st.success(f"PyTorch CUDA available: {pytorch_cuda_available}")
            st.success(f"PyTorch GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                st.write(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                st.write(
                    f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                st.write(
                    f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        else:
            st.warning("PyTorch CUDA not available")

        # Check CuPy CUDA
        if cupy_available:
            st.success(
                f"CuPy detected {cp.cuda.runtime.getDeviceCount()} CUDA-enabled GPU(s)")

            for i in range(cp.cuda.runtime.getDeviceCount()):
                props = cp.cuda.runtime.getDeviceProperties(i)
                gpu_name = props['name'].decode()

                # Basic info for table
                gpu_info.append({
                    "GPU ID": i,
                    "Name": gpu_name,
                    "Compute Capability": f"{props['major']}.{props['minor']}",
                    "Memory (GB)": round(props['totalGlobalMem'] / (1024**3), 2),
                    # Approximate for most GPUs
                    "CUDA Cores": props['multiProcessorCount'] * 64,
                    "Clock Rate (MHz)": props['clockRate'] / 1000
                })

                # Create expandable section for detailed info
                with st.expander(f"Detailed information for GPU {i}: {gpu_name}"):
                    # Select important properties
                    important_props = [
                        "totalGlobalMem", "sharedMemPerBlock", "maxThreadsPerBlock",
                        "multiProcessorCount", "clockRate", "memoryClockRate",
                        "memoryBusWidth", "l2CacheSize", "computeMode"
                    ]

                    for key in important_props:
                        if key in props:
                            val = props[key]
                            if key == "totalGlobalMem":
                                val = f"{val / (1024**3):.2f} GB"
                            elif key == "clockRate" or key == "memoryClockRate":
                                val = f"{val / 1000:.2f} MHz"
                            elif key == "l2CacheSize":
                                val = f"{val / (1024**2):.2f} MB"

                            st.write(f"{key}: {val}")

            return gpu_info
        else:
            st.warning("CuPy not available for GPU detection")
            return None

    # Detect GPUs
    gpu_info = detect_gpu()

    # Display GPU info in a table if available
    if gpu_info:
        st.subheader("GPU Summary")
        df = pd.DataFrame(gpu_info)
        st.dataframe(df)

        # Visualize GPU Memory
        st.subheader("GPU Memory Comparison")
        fig = px.bar(
            df,
            x="GPU ID",
            y="Memory (GB)",
            color="Name",
            labels={"Memory (GB)": "Memory (GB)"},
            title="GPU Memory Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Visualize CUDA Cores
        st.subheader("CUDA Cores Comparison")
        fig = px.bar(
            df,
            x="GPU ID",
            y="CUDA Cores",
            color="Name",
            labels={"CUDA Cores": "CUDA Cores"},
            title="GPU CUDA Cores Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No GPU information available")

# Memory Profile Section
elif page == "Memory Profile":
    st.header("📊 GPU Memory Profile")

    if not pytorch_cuda_available:
        st.warning("CUDA not available for memory profiling")
    else:
        # Memory allocation test
        st.subheader("Memory Allocation Test")

        # Run test button
        if st.button("Run Memory Test"):
            with st.spinner("Testing memory allocation..."):
                # Test with different tensor sizes
                sizes = [10, 100, 1000, 2000, 5000]
                memory_usage = []

                # Reserve GPU for accurate measurement
                torch.cuda.empty_cache()
                baseline = torch.cuda.memory_allocated()

                for size in sizes:
                    # Create a square tensor
                    torch.cuda.empty_cache()
                    before = torch.cuda.memory_allocated()

                    # Create tensor and record memory
                    x = torch.rand(size, size, device='cuda')
                    after = torch.cuda.memory_allocated()

                    # Calculate actual tensor size
                    tensor_bytes = size * size * 4  # 4 bytes per float32
                    measured_bytes = after - before

                    memory_usage.append({
                        "Matrix Size": f"{size}x{size}",
                        "Expected Memory (MB)": tensor_bytes / (1024**2),
                        "Actual Memory (MB)": measured_bytes / (1024**2),
                        "Overhead (%)": ((measured_bytes - tensor_bytes) / tensor_bytes) * 100 if tensor_bytes > 0 else 0
                    })

                    # Clean up
                    del x
                    torch.cuda.empty_cache()

                # Display results in table
                st.subheader("Memory Usage Results")
                mem_df = pd.DataFrame(memory_usage)
                st.dataframe(mem_df)

                # Plot memory usage
                st.subheader("Memory Usage Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[x["Matrix Size"] for x in memory_usage],
                    y=[x["Expected Memory (MB)"] for x in memory_usage],
                    mode='lines+markers',
                    name='Expected'
                ))
                fig.add_trace(go.Scatter(
                    x=[x["Matrix Size"] for x in memory_usage],
                    y=[x["Actual Memory (MB)"] for x in memory_usage],
                    mode='lines+markers',
                    name='Actual'
                ))
                fig.update_layout(
                    title='GPU Memory Usage by Matrix Size',
                    xaxis_title='Matrix Size',
                    yaxis_title='Memory Usage (MB)',
                    legend_title="Memory Type"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Overhead visualization
                st.subheader("Memory Overhead")
                fig = px.bar(
                    mem_df,
                    x="Matrix Size",
                    y="Overhead (%)",
                    title="Memory Allocation Overhead"
                )
                st.plotly_chart(fig, use_container_width=True)

# Performance Benchmark Section
elif page == "Performance Benchmark":
    st.header("⚡ Performance Benchmark")

    if not pytorch_cuda_available:
        st.warning("CUDA not available for performance benchmarking")
    else:
        # Add tabs for different benchmarks
        benchmark_tab = st.tabs(
            ["Transfer Speed", "Matrix Multiplication", "Element-wise Ops", "Reduction Ops"])

        # CPU to GPU Transfer Speeds
        with benchmark_tab[0]:
            st.subheader("CPU-GPU Transfer Speeds")

            # Run benchmark button
            if st.button("Run Transfer Benchmark"):
                with st.spinner("Testing data transfer speeds..."):
                    # Test with different tensor sizes
                    sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
                    transfer_times = []

                    for size in sizes:
                        # Create tensor on CPU
                        cpu_tensor = torch.rand(size)

                        # Time transfer to GPU
                        start = time.time()
                        gpu_tensor = cpu_tensor.cuda()
                        torch.cuda.synchronize()  # Ensure operation is complete
                        cpu_to_gpu = time.time() - start

                        # Time transfer back to CPU
                        start = time.time()
                        cpu_tensor_2 = gpu_tensor.cpu()
                        gpu_to_cpu = time.time() - start

                        # Calculate bandwidth
                        bytes_transferred = size * 4  # 4 bytes per float32

                        # Fix division by zero errors
                        cpu_to_gpu_bandwidth = bytes_transferred / cpu_to_gpu / \
                            (1024**3) if cpu_to_gpu > 0 else 0  # GB/s
                        gpu_to_cpu_bandwidth = bytes_transferred / gpu_to_cpu / \
                            (1024**3) if gpu_to_cpu > 0 else 0  # GB/s

                        transfer_times.append({
                            "Data Size (MB)": bytes_transferred / (1024**2),
                            "CPU→GPU Time (ms)": cpu_to_gpu * 1000,
                            "GPU→CPU Time (ms)": gpu_to_cpu * 1000,
                            "CPU→GPU Bandwidth (GB/s)": cpu_to_gpu_bandwidth,
                            "GPU→CPU Bandwidth (GB/s)": gpu_to_cpu_bandwidth
                        })

                        # Clean up
                        del cpu_tensor, gpu_tensor, cpu_tensor_2
                        torch.cuda.empty_cache()

                    # Display results in table
                    st.subheader("Transfer Speed Results")
                    transfer_df = pd.DataFrame(transfer_times)
                    st.dataframe(transfer_df)

                    # Plot transfer times
                    st.subheader("Transfer Times")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[x["Data Size (MB)"] for x in transfer_times],
                        y=[x["CPU→GPU Time (ms)"] for x in transfer_times],
                        mode='lines+markers',
                        name='CPU→GPU'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[x["Data Size (MB)"] for x in transfer_times],
                        y=[x["GPU→CPU Time (ms)"] for x in transfer_times],
                        mode='lines+markers',
                        name='GPU→CPU'
                    ))
                    fig.update_layout(
                        title='Data Transfer Times',
                        xaxis_title='Data Size (MB)',
                        yaxis_title='Transfer Time (ms)',
                        xaxis_type="log",
                        yaxis_type="log"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Plot bandwidth
                    st.subheader("Transfer Bandwidth")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[x["Data Size (MB)"] for x in transfer_times],
                        y=[x["CPU→GPU Bandwidth (GB/s)"]
                           for x in transfer_times],
                        mode='lines+markers',
                        name='CPU→GPU'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[x["Data Size (MB)"] for x in transfer_times],
                        y=[x["GPU→CPU Bandwidth (GB/s)"]
                           for x in transfer_times],
                        mode='lines+markers',
                        name='GPU→CPU'
                    ))
                    fig.update_layout(
                        title='Data Transfer Bandwidth',
                        xaxis_title='Data Size (MB)',
                        yaxis_title='Bandwidth (GB/s)',
                        xaxis_type="log"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Matrix Multiplication Benchmark
        with benchmark_tab[1]:
            st.subheader("Matrix Multiplication Benchmark")

            # Run benchmark button
            if st.button("Run Matrix Multiplication Benchmark"):
                with st.spinner("Testing matrix multiplication..."):
                    sizes = [128, 256, 512, 1024, 2048]
                    results = []

                    for size in sizes:
                        # Create matrices
                        A_cpu = torch.rand(size, size)
                        B_cpu = torch.rand(size, size)

                        # CPU computation
                        start = time.time()
                        C_cpu = torch.matmul(A_cpu, B_cpu)
                        cpu_time = time.time() - start

                        # GPU computation
                        A_gpu = A_cpu.cuda()
                        B_gpu = B_cpu.cuda()
                        torch.cuda.synchronize()

                        start = time.time()
                        C_gpu = torch.matmul(A_gpu, B_gpu)
                        torch.cuda.synchronize()
                        gpu_time = time.time() - start

                        # GFLOPS calculation (2 * N^3 operations for matmul)
                        ops = 2 * size**3

                        # Fix division by zero errors
                        cpu_gflops = ops / \
                            (cpu_time * 1e9) if cpu_time > 0 else 0
                        gpu_gflops = ops / \
                            (gpu_time * 1e9) if gpu_time > 0 else 0
                        speedup = cpu_time / gpu_time if gpu_time > 0 and cpu_time > 0 else 0

                        results.append({
                            "Matrix Size": f"{size}x{size}",
                            "CPU Time (ms)": cpu_time * 1000,
                            "GPU Time (ms)": gpu_time * 1000,
                            "CPU GFLOPS": cpu_gflops,
                            "GPU GFLOPS": gpu_gflops,
                            "Speedup": speedup
                        })

                        # Clean up
                        del A_cpu, B_cpu, C_cpu, A_gpu, B_gpu, C_gpu
                        torch.cuda.empty_cache()

                    # Display results
                    st.subheader("Matrix Multiplication Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)

                    # Plot execution time
                    st.subheader("Execution Time Comparison")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[x["Matrix Size"] for x in results],
                        y=[x["CPU Time (ms)"] for x in results],
                        name='CPU'
                    ))
                    fig.add_trace(go.Bar(
                        x=[x["Matrix Size"] for x in results],
                        y=[x["GPU Time (ms)"] for x in results],
                        name='GPU'
                    ))
                    fig.update_layout(
                        title='Matrix Multiplication Time',
                        xaxis_title='Matrix Size',
                        yaxis_title='Time (ms)',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Plot GFLOPS
                    st.subheader("Computing Performance (GFLOPS)")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[x["Matrix Size"] for x in results],
                        y=[x["CPU GFLOPS"] for x in results],
                        name='CPU'
                    ))
                    fig.add_trace(go.Bar(
                        x=[x["Matrix Size"] for x in results],
                        y=[x["GPU GFLOPS"] for x in results],
                        name='GPU'
                    ))
                    fig.update_layout(
                        title='Computing Performance',
                        xaxis_title='Matrix Size',
                        yaxis_title='GFLOPS',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Plot speedup
                    st.subheader("GPU Speedup")
                    fig = px.bar(
                        results_df,
                        x="Matrix Size",
                        y="Speedup",
                        title="GPU Speedup over CPU"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Element-wise Operations Benchmark
        with benchmark_tab[2]:
            st.subheader("Element-wise Operations Benchmark")

            # Run benchmark button
            if st.button("Run Element-wise Operations Benchmark"):
                with st.spinner("Testing element-wise operations..."):
                    # Element-wise operations to test
                    operations = ["addition", "multiplication",
                                  "exponential", "trigonometric"]
                    sizes = [1000, 10000, 100000, 1000000, 10000000]
                    results = []

                    for size in sizes:
                        # Create tensors
                        a_cpu = torch.rand(size)
                        b_cpu = torch.rand(size)

                        # Move to GPU
                        if pytorch_cuda_available:
                            a_gpu = a_cpu.cuda()
                            b_gpu = b_cpu.cuda()

                        # Test operations
                        for op in operations:
                            # CPU operation
                            start = time.time()
                            if op == "addition":
                                c_cpu = a_cpu + b_cpu
                            elif op == "multiplication":
                                c_cpu = a_cpu * b_cpu
                            elif op == "exponential":
                                c_cpu = torch.exp(a_cpu)
                            elif op == "trigonometric":
                                c_cpu = torch.sin(a_cpu)
                            cpu_time = time.time() - start

                            # GPU operation
                            if pytorch_cuda_available:
                                torch.cuda.synchronize()
                                start = time.time()
                                if op == "addition":
                                    c_gpu = a_gpu + b_gpu
                                elif op == "multiplication":
                                    c_gpu = a_gpu * b_gpu
                                elif op == "exponential":
                                    c_gpu = torch.exp(a_gpu)
                                elif op == "trigonometric":
                                    c_gpu = torch.sin(a_gpu)
                                torch.cuda.synchronize()
                                gpu_time = time.time() - start
                            else:
                                gpu_time = 0

                            # Calculate speedup
                            speedup = cpu_time / gpu_time if gpu_time > 0 and cpu_time > 0 else 0

                            # Calculate throughput in elements per second
                            cpu_throughput = size / cpu_time if cpu_time > 0 else 0
                            gpu_throughput = size / gpu_time if gpu_time > 0 else 0

                            results.append({
                                "Operation": op,
                                "Vector Size": size,
                                "CPU Time (ms)": cpu_time * 1000,
                                "GPU Time (ms)": gpu_time * 1000,
                                "Speedup": speedup,
                                "CPU Throughput (M elements/s)": cpu_throughput / 1e6,
                                "GPU Throughput (M elements/s)": gpu_throughput / 1e6
                            })

                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("Element-wise Operation Results")
                    st.dataframe(results_df)

                    # Plot speedup by operation type and size
                    st.subheader("Element-wise Operations Speedup")
                    fig = px.bar(
                        results_df,
                        x="Vector Size",
                        y="Speedup",
                        color="Operation",
                        barmode="group",
                        title="GPU Speedup for Element-wise Operations"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Plot throughput comparison
                    st.subheader("Throughput Comparison")
                    # Reshape dataframe for plotting
                    plot_df = pd.melt(
                        results_df,
                        id_vars=["Operation", "Vector Size"],
                        value_vars=[
                            "CPU Throughput (M elements/s)", "GPU Throughput (M elements/s)"],
                        var_name="Device",
                        value_name="Throughput (M elements/s)"
                    )

                    fig = px.line(
                        plot_df,
                        x="Vector Size",
                        y="Throughput (M elements/s)",
                        color="Device",
                        facet_col="Operation",
                        log_x=True,
                        title="Element-wise Operations Throughput"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Reduction Operations Benchmark
        with benchmark_tab[3]:
            st.subheader("Reduction Operations Benchmark")

            # Run benchmark button
            if st.button("Run Reduction Operations Benchmark"):
                with st.spinner("Testing reduction operations..."):
                    # Reduction operations to test
                    operations = ["sum", "mean", "max", "min"]
                    sizes = [1000, 10000, 100000, 1000000, 10000000]
                    results = []

                    for size in sizes:
                        # Create tensor
                        a_cpu = torch.rand(size)

                        # Move to GPU
                        if pytorch_cuda_available:
                            a_gpu = a_cpu.cuda()

                        # Test operations
                        for op in operations:
                            # CPU operation
                            start = time.time()
                            if op == "sum":
                                res_cpu = torch.sum(a_cpu)
                            elif op == "mean":
                                res_cpu = torch.mean(a_cpu)
                            elif op == "max":
                                res_cpu = torch.max(a_cpu)
                            elif op == "min":
                                res_cpu = torch.min(a_cpu)
                            cpu_time = time.time() - start

                            # GPU operation
                            if pytorch_cuda_available:
                                torch.cuda.synchronize()
                                start = time.time()
                                if op == "sum":
                                    res_gpu = torch.sum(a_gpu)
                                elif op == "mean":
                                    res_gpu = torch.mean(a_gpu)
                                elif op == "max":
                                    res_gpu = torch.max(a_gpu)
                                elif op == "min":
                                    res_gpu = torch.min(a_gpu)
                                torch.cuda.synchronize()
                                gpu_time = time.time() - start
                            else:
                                gpu_time = 0

                            # Calculate speedup
                            speedup = cpu_time / gpu_time if gpu_time > 0 and cpu_time > 0 else 0

                            results.append({
                                "Operation": op,
                                "Vector Size": size,
                                "CPU Time (ms)": cpu_time * 1000,
                                "GPU Time (ms)": gpu_time * 1000,
                                "Speedup": speedup
                            })

                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("Reduction Operation Results")
                    st.dataframe(results_df)

                    # Plot speedup by operation type and size
                    st.subheader("Reduction Operations Speedup")
                    fig = px.line(
                        results_df,
                        x="Vector Size",
                        y="Speedup",
                        color="Operation",
                        markers=True,
                        log_x=True,
                        title="GPU Speedup for Reduction Operations"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Execution time comparison
                    st.subheader("Execution Time Comparison")
                    # Reshape dataframe for plotting
                    plot_df = pd.melt(
                        results_df,
                        id_vars=["Operation", "Vector Size"],
                        value_vars=["CPU Time (ms)", "GPU Time (ms)"],
                        var_name="Device",
                        value_name="Time (ms)"
                    )

                    fig = px.line(
                        plot_df,
                        x="Vector Size",
                        y="Time (ms)",
                        color="Device",
                        facet_col="Operation",
                        log_x=True,
                        log_y=True,
                        title="Reduction Operations Execution Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Custom Operations Section
elif page == "Custom Operations":
    st.header("🔧 Custom Kernel Performance")

    if not pytorch_cuda_available:
        st.warning("CUDA not available for custom kernel performance testing")
    else:
        st.subheader("Custom CUDA Kernel Execution")

        # Example CUDA kernel code display
        st.code("""
# Example CUDA kernel for vector addition
@cp.cuda.memoize(for_each_device=True)
def vector_add_kernel(grid_size, block_size):
    return cp.cuda.compile_with_cache(r'''
    extern "C" __global__ void vector_add(const float* x, const float* y, float* z, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n) {
            z[tid] = x[tid] + y[tid];
        }
    }
    ''')
        """, language="python")

        # Run custom kernel button
        if st.button("Run Custom Kernel Benchmark"):
            with st.spinner("Testing custom kernel performance..."):
                try:
                    # Define vector sizes
                    sizes = [10000, 100000, 1000000, 10000000, 100000000]
                    results = []

                    # Try to use CuPy for custom kernel if available
                    if cupy_available:
                        # Define vector add kernel
                        vector_add_kernel = cp.RawKernel(r'''
                        extern "C" __global__ void vector_add(const float* x, const float* y, float* z, int n) {
                            int tid = blockDim.x * blockIdx.x + threadIdx.x;
                            if (tid < n) {
                                z[tid] = x[tid] + y[tid];
                            }
                        }
                        ''', 'vector_add')

                        for size in sizes:
                            # Create input arrays
                            x_gpu = cp.random.random(size, dtype=cp.float32)
                            y_gpu = cp.random.random(size, dtype=cp.float32)
                            z_gpu = cp.zeros_like(x_gpu)

                            # PyTorch implementation for comparison
                            x_torch = torch.from_numpy(
                                cp.asnumpy(x_gpu)).cuda()
                            y_torch = torch.from_numpy(
                                cp.asnumpy(y_gpu)).cuda()

                            # Determine grid and block sizes
                            block_size = 256
                            grid_size = (size + block_size - 1) // block_size

                            # Time PyTorch implementation
                            torch.cuda.synchronize()
                            start = time.time()
                            z_torch = x_torch + y_torch
                            torch.cuda.synchronize()
                            torch_time = time.time() - start

                            # Time CuPy kernel
                            cp.cuda.runtime.deviceSynchronize()
                            start = time.time()
                            # Launch kernel
                            vector_add_kernel(
                                (grid_size,),
                                (block_size,),
                                (x_gpu, y_gpu, z_gpu, size)
                            )
                            cp.cuda.runtime.deviceSynchronize()
                            kernel_time = time.time() - start

                            # Speedup relative to PyTorch
                            speedup = torch_time / kernel_time if kernel_time > 0 else 0

                            # Calculate throughput
                            kernel_throughput = size / kernel_time if kernel_time > 0 else 0
                            torch_throughput = size / torch_time if torch_time > 0 else 0

                            results.append({
                                "Vector Size": size,
                                "PyTorch Time (ms)": torch_time * 1000,
                                "Custom Kernel Time (ms)": kernel_time * 1000,
                                "Speedup over PyTorch": speedup,
                                "PyTorch Throughput (GElements/s)": torch_throughput / 1e9,
                                "Kernel Throughput (GElements/s)": kernel_throughput / 1e9
                            })

                        # Display results
                        results_df = pd.DataFrame(results)
                        st.subheader("Custom Kernel Results")
                        st.dataframe(results_df)

                        # Plot execution time comparison
                        st.subheader("Execution Time Comparison")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[x["Vector Size"] for x in results],
                            y=[x["PyTorch Time (ms)"] for x in results],
                            mode='lines+markers',
                            name='PyTorch'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[x["Vector Size"] for x in results],
                            y=[x["Custom Kernel Time (ms)"] for x in results],
                            mode='lines+markers',
                            name='Custom CUDA Kernel'
                        ))
                        fig.update_layout(
                            title='Execution Time: PyTorch vs Custom CUDA Kernel',
                            xaxis_title='Vector Size',
                            yaxis_title='Time (ms)',
                            xaxis_type="log",
                            yaxis_type="log"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Plot throughput comparison
                        st.subheader("Throughput Comparison")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[x["Vector Size"] for x in results],
                            y=[x["PyTorch Throughput (GElements/s)"]
                               for x in results],
                            mode='lines+markers',
                            name='PyTorch'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[x["Vector Size"] for x in results],
                            y=[x["Kernel Throughput (GElements/s)"]
                               for x in results],
                            mode='lines+markers',
                            name='Custom CUDA Kernel'
                        ))
                        fig.update_layout(
                            title='Throughput: PyTorch vs Custom CUDA Kernel',
                            xaxis_title='Vector Size',
                            yaxis_title='Throughput (GElements/s)',
                            xaxis_type="log"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Optimization insights
                        st.subheader("Optimization Insights")
                        st.info("""
                        Custom CUDA kernels can sometimes outperform PyTorch operations because:
                        1. They have less overhead by skipping the Python API layer
                        2. They can be specifically optimized for a particular task
                        3. They allow for fine-tuned control over memory access patterns
                        4. They can utilize CUDA-specific optimizations and shared memory
                        """)

                    else:
                        st.error(
                            "CuPy is not available for custom kernel benchmarking")
                except Exception as e:
                    st.error(f"Error running custom kernel benchmark: {e}")

# Benchmark Summary Section
elif page == "Benchmark Summary":
    st.header("📊 Performance Analysis & Benchmark Summary")

    # Overall performance metrics
    st.subheader("Overall GPU Performance Metrics")

    col1, col2, col3 = st.columns(3)

    # Example values - in a real app these would come from the benchmarks
    with col1:
        st.metric("Avg. Matrix Mult. Speedup", "120x", delta="↑")
        st.metric("Avg. Element-wise Speedup", "50x", delta="↑")
    with col2:
        st.metric("Avg. Transfer Bandwidth", "12 GB/s", delta="↑")
        st.metric("Peak FLOPS", "15 TFLOPS", delta="↑")
    with col3:
        st.metric("Avg. Reduction Speedup", "35x", delta="↑")
        st.metric("Custom Kernel Efficiency", "95%", delta="↑")

    # Performance comparison across operations
    st.subheader("Performance Comparison Across Operations")

    # Generate example data for visualization
    operations = ["Matrix Multiplication", "Vector Addition", "Reduction (Sum)",
                  "Element-wise Exp", "Custom Kernel", "Sort"]
    speedups = [120, 50, 35, 45, 55, 80]

    op_comparison = pd.DataFrame({
        "Operation": operations,
        "Speedup (GPU vs CPU)": speedups
    })

    fig = px.bar(
        op_comparison,
        x="Operation",
        y="Speedup (GPU vs CPU)",
        color="Operation",
        title="GPU Speedup by Operation Type"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Performance scaling
    st.subheader("Performance Scaling with Problem Size")

    # Generate example data for scaling visualization
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    matrix_mult = [5, 25, 80, 150, 200]
    vector_add = [2, 10, 30, 60, 100]
    reduction = [1, 5, 20, 40, 70]

    scaling_data = pd.DataFrame({
        "Problem Size": sizes * 3,
        "Operation": ["Matrix Multiplication"] * 5 + ["Vector Addition"] * 5 + ["Reduction"] * 5,
        "Speedup": matrix_mult + vector_add + reduction
    })

    fig = px.line(
        scaling_data,
        x="Problem Size",
        y="Speedup",
        color="Operation",
        markers=True,
        log_x=True,
        title="GPU Speedup vs Problem Size"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Performance bottlenecks
    st.subheader("Performance Bottlenecks Analysis")

    bottlenecks = pd.DataFrame({
        "Bottleneck": ["Memory Transfer", "Kernel Launch Overhead", "Memory Bandwidth", "Compute Bound", "Synchronization"],
        "Impact (%)": [40, 10, 25, 15, 10]
    })

    fig = px.pie(
        bottlenecks,
        values="Impact (%)",
        names="Bottleneck",
        title="Performance Bottlenecks Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.subheader("Performance Optimization Recommendations")

    st.info("""
    Based on the benchmark results, here are key recommendations to optimize GPU performance:
    
    1. **Minimize Host-Device Transfers**: Batch operations to reduce overhead
    2. **Use Asynchronous Operations**: Overlap computation with data transfers
    3. **Optimize Memory Access Patterns**: Ensure coalesced memory access
    4. **Utilize Shared Memory**: For operations with data reuse
    5. **Consider Custom Kernels**: For critical performance paths
    6. **Batch Small Operations**: Reduce kernel launch overhead
    7. **Use Mixed Precision**: Consider FP16 for appropriate workloads
    """)

    # Technical explanations
    with st.expander("Technical Explanation of GPU Performance Factors"):
        st.markdown("""
        ### Memory Hierarchy
        GPUs have a complex memory hierarchy including global memory, shared memory, 
        and registers. Understanding how to effectively use each level is crucial for performance.
        
        ### Thread Divergence
        When threads in a warp take different execution paths, performance suffers due to serialization.
        Design algorithms to minimize thread divergence.
        
        ### Occupancy
        The ratio of active warps to the maximum possible warps on a multiprocessor. 
        Higher occupancy generally leads to better performance by hiding memory latency.
        
        ### Memory Coalescing
        When threads in a warp access contiguous memory locations, the GPU can coalesce 
        these into a single memory transaction, significantly improving bandwidth utilization.
        
        ### Kernel Fusion
        Combining multiple operations into a single kernel to reduce memory traffic and 
        kernel launch overhead.
        """)

# Comparison Section
elif page == "Comparison":
    st.header("📈 Hardware Comparison")
    st.subheader("Your Hardware vs. Common GPUs")

    # Example data for comparison
    common_gpus = pd.DataFrame({
        "GPU": ["Your GPU", "RTX 3080", "RTX 3090", "RTX 4090", "A100", "V100"],
        "CUDA Cores": [0, 8704, 10496, 16384, 6912, 5120],
        "Memory (GB)": [0, 10, 24, 24, 40, 32],
        "Memory Bandwidth (GB/s)": [0, 760, 936, 1008, 1555, 900],
        "FP32 Performance (TFLOPS)": [0, 29.8, 35.6, 82.6, 19.5, 15.7]
    })

    # Update with actual values if available
    if pytorch_cuda_available:
        try:
            for i in range(torch.cuda.device_count()):
                props = cp.cuda.runtime.getDeviceProperties(i)
                common_gpus.at[0, "GPU"] = props['name'].decode()
                common_gpus.at[0,
                               "CUDA Cores"] = props['multiProcessorCount'] * 64
                common_gpus.at[0, "Memory (GB)"] = round(
                    props['totalGlobalMem'] / (1024**3), 2)
                common_gpus.at[0, "Memory Bandwidth (GB/s)"] = props['memoryClockRate'] / \
                    1000 * props['memoryBusWidth'] / 8
                # FP32 Performance is an approximation
                common_gpus.at[0, "FP32 Performance (TFLOPS)"] = common_gpus.at[0,
                                                                                "CUDA Cores"] * props['clockRate'] / 1e6 * 2 / 1000
        except:
            st.warning("Could not update GPU information automatically")

    # Display comparison table
    st.dataframe(common_gpus)

    # Visualize comparisons
    metrics = ["CUDA Cores",
               "Memory (GB)", "Memory Bandwidth (GB/s)", "FP32 Performance (TFLOPS)"]
    selected_metric = st.selectbox("Select metric for comparison", metrics)

    fig = px.bar(
        common_gpus,
        x="GPU",
        y=selected_metric,
        color="GPU",
        title=f"GPU Comparison: {selected_metric}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart for overall comparison
    st.subheader("Overall Comparison (Normalized)")

    # Normalize data for radar chart
    radar_data = common_gpus.copy()
    for metric in metrics:
        max_val = radar_data[metric].max()
        if max_val > 0:
            radar_data[metric] = radar_data[metric] / max_val

    # Create radar chart
    fig = go.Figure()

    for i, gpu in enumerate(radar_data["GPU"]):
        fig.add_trace(go.Scatterpolar(
            r=[radar_data.loc[i, metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=gpu
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "👨‍💻 Created for technical presentations on GPU capabilities and performance analysis")
