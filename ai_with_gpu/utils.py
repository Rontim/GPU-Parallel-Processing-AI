import numpy as np
import cupy as cp
import torch
import platform
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import os
import plotly.express as px
import plotly.graph_objects as go


def get_system_info():
    """Collect and display system information"""
    system_info = {
        "Operating System": f"{platform.system()} {platform.release()} ({platform.version()})",
        "Architecture": platform.architecture()[0],
        "Processor": platform.processor(),
        "CPU Cores (Physical)": psutil.cpu_count(logical=False),
        "CPU Cores (Logical)": psutil.cpu_count(logical=True),
        "Total RAM": f"{round(psutil.virtual_memory().total / (1024.0 ** 3), 2)} GB",
        "Python Version": platform.python_version(),
        "NumPy Version": np.__version__,
    }

    # Add GPU-related info if available
    if has_pytorch_gpu():
        system_info["PyTorch Version"] = torch.__version__
        system_info["CUDA Version"] = torch.version.cuda if torch.version.cuda else "N/A"
        system_info["Number of GPUs"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            system_info[f"GPU {i}"] = torch.cuda.get_device_name(i)

    if has_cupy():
        system_info["CuPy Version"] = cp.__version__

    system_info["Date & Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return system_info


def has_pytorch_gpu():
    """Check if PyTorch can detect a GPU"""
    try:
        return torch.cuda.is_available()
    except:
        return False


def has_cupy():
    """Check if CuPy is available and can access GPU"""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except:
        return False


def sigmoid(x, use_gpu=False):
    """Sigmoid activation function with GPU option"""
    if use_gpu and has_cupy():
        return 1 / (1 + cp.exp(-x))
    else:
        return 1 / (1 + np.exp(-x))


def tanh(x, use_gpu=False):
    """Tanh activation function with GPU option"""
    if use_gpu and has_cupy():
        return cp.tanh(x)
    else:
        return np.tanh(x)


def binary_cross_entropy(y_pred, y_true, use_gpu=False):
    """Binary cross entropy loss with GPU option"""
    eps = 1e-8
    if use_gpu and has_cupy():
        return -cp.mean(y_true * cp.log(y_pred + eps) + (1 - y_true) * cp.log(1 - y_pred + eps))
    else:
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def benchmark_operation(func, *args, **kwargs):
    """Benchmark the execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def to_device(data, use_gpu=False):
    """Move data to appropriate device (GPU or CPU)"""
    if use_gpu and has_cupy():
        return cp.asarray(data)
    else:
        return np.asarray(data)


def to_cpu(data, use_gpu=False):
    """Move data back to CPU if needed"""
    if use_gpu and has_cupy() and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data


def plot_comparison(cpu_times, gpu_times, operation_names, title="CPU vs GPU Performance"):
    """Create a bar chart comparing CPU and GPU performance"""
    fig = px.bar(
        x=operation_names,
        y=[cpu_times, gpu_times],
        title=title,
        labels={"x": "Operations", "y": "Time (seconds)"},
        color_discrete_sequence=["#1E88E5", "#FF0D57"],
        barmode="group"
    )

    fig.update_layout(
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add data labels
    fig.update_traces(
        texttemplate="%{y:.4f}s",
        textposition="outside",
        selector=dict(type='bar')
    )

    # Update x-axis appearance
    fig.update_xaxes(
        tickangle=-45,
        title_font=dict(size=14),
        tickfont=dict(size=12),
    )

    # Update y-axis appearance
    fig.update_yaxes(
        title_font=dict(size=14),
        tickfont=dict(size=12),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)'
    )

    return fig
