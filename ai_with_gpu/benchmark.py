import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
import pandas as pd
from .utils import has_cupy, to_device, to_cpu, benchmark_operation
from .neural_network import NeuralNetwork, generate_data


def run_matrix_operations_benchmark(sizes=None, runs=3):
    """Run benchmark comparing CPU and GPU for basic matrix operations"""
    if sizes is None:
        sizes = [1000, 2000, 5000, 10000]

    results = {
        'operation': [],
        'size': [],
        'cpu_time': [],
        'gpu_time': [],
        'speedup': []
    }

    operations = [
        ('Matrix Multiplication', lambda x, y: x @ y),
        ('Element-wise Operations', lambda x, y: x * y + x),
        ('Matrix Transpose', lambda x, _: x.T @ x)
    ]

    for size in sizes:
        for op_name, op_func in operations:
            cpu_times = []
            gpu_times = []

            for _ in range(runs):
                # CPU benchmark
                A_cpu = np.random.random((size, size)).astype(np.float32)
                B_cpu = np.random.random((size, size)).astype(np.float32)

                start = time.time()
                _ = op_func(A_cpu, B_cpu)
                cpu_time = time.time() - start
                cpu_times.append(cpu_time)

                # GPU benchmark (if available)
                if has_cupy():
                    A_gpu = cp.random.random((size, size)).astype(cp.float32)
                    B_gpu = cp.random.random((size, size)).astype(cp.float32)

                    # Warmup
                    _ = op_func(A_gpu, B_gpu)
                    cp.cuda.stream.get_current_stream().synchronize()

                    start = time.time()
                    _ = op_func(A_gpu, B_gpu)
                    cp.cuda.stream.get_current_stream().synchronize()
                    gpu_time = time.time() - start
                    gpu_times.append(gpu_time)
                else:
                    gpu_times.append(float('nan'))

            # Average the results
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            avg_gpu_time = sum([t for t in gpu_times if not np.isnan(
                t)]) / len([t for t in gpu_times if not np.isnan(t)]) if has_cupy() else float('nan')

            # Calculate speedup
            speedup = avg_cpu_time / avg_gpu_time if has_cupy() and avg_gpu_time > 0 else float('nan')

            # Add to results
            results['operation'].append(f"{op_name} ({size}×{size})")
            results['size'].append(size)
            results['cpu_time'].append(avg_cpu_time)
            results['gpu_time'].append(avg_gpu_time)
            results['speedup'].append(speedup)

    return pd.DataFrame(results)


def benchmark_neural_network(epochs=1000, lr=0.1, hidden_size=32):
    """Benchmark neural network training on CPU vs GPU"""
    X, y = generate_data(n_samples=2000, noise=0.2)

    # CPU training
    start_time = time.time()
    model_cpu = NeuralNetwork(
        n_input=2, n_hidden=hidden_size, n_output=1, use_gpu=False)
    cpu_loss_history = model_cpu.train(
        X, y, epochs=epochs, lr=lr, verbose=False)
    cpu_time = time.time() - start_time

    # GPU training (if available)
    if has_cupy():
        start_time = time.time()
        model_gpu = NeuralNetwork(
            n_input=2, n_hidden=hidden_size, n_output=1, use_gpu=True)
        gpu_loss_history = model_gpu.train(
            X, y, epochs=epochs, lr=lr, verbose=False)
        gpu_time = time.time() - start_time

        speedup = cpu_time / gpu_time
    else:
        gpu_time = float('nan')
        gpu_loss_history = None
        speedup = float('nan')

    results = {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'cpu_loss_history': cpu_loss_history,
        'gpu_loss_history': gpu_loss_history,
        'model_cpu': model_cpu,
        'model_gpu': model_gpu if has_cupy() else None,
        'X': X,
        'y': y
    }

    return results


def benchmark_neural_network_components(X=None, y=None, hidden_size=32, samples=10):
    """Benchmark individual components of neural network (forward pass, backward pass, etc.)"""
    if X is None or y is None:
        X, y = generate_data(n_samples=2000, noise=0.2)

    results = {
        'operation': [],
        'cpu_time': [],
        'gpu_time': [],
        'speedup': []
    }

    # Initialize models
    model_cpu = NeuralNetwork(
        n_input=2, n_hidden=hidden_size, n_output=1, use_gpu=False)

    if has_cupy():
        model_gpu = NeuralNetwork(
            n_input=2, n_hidden=hidden_size, n_output=1, use_gpu=True)
        X_gpu = to_device(X, True)
        y_gpu = to_device(y, True)

    # Benchmark forward pass
    cpu_times = []
    gpu_times = []

    for _ in range(samples):
        # CPU forward pass
        start = time.time()
        y_pred_cpu, a1_cpu = model_cpu.forward(X)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)

        # GPU forward pass
        if has_cupy():
            cp.cuda.stream.get_current_stream().synchronize()
            start = time.time()
            y_pred_gpu, a1_gpu = model_gpu.forward(X_gpu)
            cp.cuda.stream.get_current_stream().synchronize()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)

    avg_cpu_forward = sum(cpu_times) / len(cpu_times)
    if has_cupy():
        avg_gpu_forward = sum(gpu_times) / len(gpu_times)
        speedup_forward = avg_cpu_forward / avg_gpu_forward
    else:
        avg_gpu_forward = float('nan')
        speedup_forward = float('nan')

    results['operation'].append('Forward Pass')
    results['cpu_time'].append(avg_cpu_forward)
    results['gpu_time'].append(avg_gpu_forward)
    results['speedup'].append(speedup_forward)

    # Benchmark backward pass
    cpu_times = []
    gpu_times = []

    for _ in range(samples):
        # CPU backward pass
        start = time.time()
        dW1_cpu, db1_cpu, dW2_cpu, db2_cpu = model_cpu.backward(
            X, y, y_pred_cpu, a1_cpu)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)

        # GPU backward pass
        if has_cupy():
            cp.cuda.stream.get_current_stream().synchronize()
            start = time.time()
            dW1_gpu, db1_gpu, dW2_gpu, db2_gpu = model_gpu.backward(
                X_gpu, y_gpu, y_pred_gpu, a1_gpu)
            cp.cuda.stream.get_current_stream().synchronize()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)

    avg_cpu_backward = sum(cpu_times) / len(cpu_times)
    if has_cupy():
        avg_gpu_backward = sum(gpu_times) / len(gpu_times)
        speedup_backward = avg_cpu_backward / avg_gpu_backward
    else:
        avg_gpu_backward = float('nan')
        speedup_backward = float('nan')

    results['operation'].append('Backward Pass')
    results['cpu_time'].append(avg_cpu_backward)
    results['gpu_time'].append(avg_gpu_backward)
    results['speedup'].append(speedup_backward)

    # Benchmark parameter update
    cpu_times = []
    gpu_times = []

    for _ in range(samples):
        # CPU parameter update
        start = time.time()
        model_cpu.update_params(dW1_cpu, db1_cpu, dW2_cpu, db2_cpu, lr=0.1)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)

        # GPU parameter update
        if has_cupy():
            cp.cuda.stream.get_current_stream().synchronize()
            start = time.time()
            model_gpu.update_params(dW1_gpu, db1_gpu, dW2_gpu, db2_gpu, lr=0.1)
            cp.cuda.stream.get_current_stream().synchronize()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)

    avg_cpu_update = sum(cpu_times) / len(cpu_times)
    if has_cupy():
        avg_gpu_update = sum(gpu_times) / len(gpu_times)
        speedup_update = avg_cpu_update / avg_gpu_update
    else:
        avg_gpu_update = float('nan')
        speedup_update = float('nan')

    results['operation'].append('Parameter Update')
    results['cpu_time'].append(avg_cpu_update)
    results['gpu_time'].append(avg_gpu_update)
    results['speedup'].append(speedup_update)

    return pd.DataFrame(results)


def plot_matrix_benchmark_results(benchmark_df):
    """Create charts visualizing matrix operation benchmark results"""
    # Group by size
    sizes = benchmark_df['size'].unique()
    operations = [op for op in benchmark_df['operation'].unique() if '×' in op]

    fig, axes = plt.subplots(1, len(sizes), figsize=(20, 6))
    if len(sizes) == 1:
        axes = [axes]

    for i, size in enumerate(sizes):
        size_data = benchmark_df[benchmark_df['size'] == size]

        # Get CPU and GPU times
        cpu_times = size_data['cpu_time'].values
        gpu_times = size_data['gpu_time'].values

        # Create labels
        labels = [op.split('(')[0].strip() for op in size_data['operation']]

        # Sort by CPU time
        sort_indices = np.argsort(cpu_times)[::-1]
        cpu_times = cpu_times[sort_indices]
        gpu_times = gpu_times[sort_indices]
        labels = [labels[j] for j in sort_indices]

        # Plot
        x = np.arange(len(labels))
        width = 0.35

        axes[i].bar(x - width/2, cpu_times, width,
                    label='CPU', color='#1E88E5')
        axes[i].bar(x + width/2, gpu_times, width,
                    label='GPU', color='#FF0D57')

        axes[i].set_title(f'Matrix Size: {size}×{size}')
        axes[i].set_xlabel('Operation')
        axes[i].set_ylabel('Time (seconds)')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].legend()

        # Add speedup text
        for j, (cpu_t, gpu_t) in enumerate(zip(cpu_times, gpu_times)):
            if not np.isnan(gpu_t) and gpu_t > 0:
                speedup = cpu_t / gpu_t
                axes[i].text(j, max(cpu_t, gpu_t) * 1.05, f'{speedup:.1f}x',
                             ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_nn_component_benchmark(component_df):
    """Create bar chart comparing neural network component performance"""
    operations = component_df['operation'].values
    cpu_times = component_df['cpu_time'].values
    gpu_times = component_df['gpu_time'].values

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(operations))
    width = 0.35

    ax.bar(x - width/2, cpu_times, width, label='CPU', color='#1E88E5')
    ax.bar(x + width/2, gpu_times, width, label='GPU', color='#FF0D57')

    ax.set_title('Neural Network Component Performance')
    ax.set_xlabel('Component')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend()

    # Add speedup text
    for i, (cpu_t, gpu_t) in enumerate(zip(cpu_times, gpu_times)):
        if not np.isnan(gpu_t) and gpu_t > 0:
            speedup = cpu_t / gpu_t
            ax.text(i, max(cpu_t, gpu_t) * 1.05, f'{speedup:.1f}x',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig
