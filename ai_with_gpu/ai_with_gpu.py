from ai_with_gpu.benchmark import (
    run_matrix_operations_benchmark,
    benchmark_neural_network,
    benchmark_neural_network_components,
    plot_matrix_benchmark_results,
    plot_nn_component_benchmark
)
from ai_with_gpu.neural_network import NeuralNetwork, generate_data, plot_data, plot_decision_boundary, plot_training_loss
from ai_with_gpu.utils import get_system_info, has_pytorch_gpu, has_cupy, plot_comparison
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Add the parent directory to sys.path to enable imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local imports

# Page configuration
st.set_page_config(
    page_title="AI Acceleration with GPU",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Material Design Colors */
    :root {
        --primary-color: #2962FF;
        --secondary-color: #7B1FA2;
        --background-color: #F5F5F5;
        --surface-color: #FFFFFF;
        --text-color: #212121;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --error-color: #F44336;
    }
    
    /* Card styles */
    .card {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: var(--surface-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    .card-title {
        color: var(--primary-color);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.3rem;
    }
    .card-content {
        color: var(--text-color);
    }
    
    /* Header styles */
    h1, h2, h3 {
        color: var(--primary-color);
    }
    h1 {
        font-weight: 700;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    h2 {
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h3 {
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        color: var(--secondary-color);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🚀 AI Acceleration with GPU")
st.markdown("""
This application demonstrates the power of GPU acceleration for AI and machine learning tasks. 
We'll explore how GPUs can significantly speed up neural network training and other operations
through parallel processing.
""")

# Sidebar with navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Section",
    [
        "System Information",
        "Matrix Operations Benchmark",
        "Neural Network from Scratch",
        "Neural Network Components",
        "About GPU Acceleration"
    ]
)

# System information page
if page == "System Information":
    st.header("🖥️ System Information")

    system_info = get_system_info()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">System</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**OS:** {system_info['Operating System']}")
        st.markdown(f"**Architecture:** {system_info['Architecture']}")
        st.markdown(f"**Processor:** {system_info['Processor']}")
        st.markdown(
            f"**Physical CPU Cores:** {system_info['CPU Cores (Physical)']}")
        st.markdown(
            f"**Logical CPU Cores:** {system_info['CPU Cores (Logical)']}")
        st.markdown(f"**Total RAM:** {system_info['Total RAM']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">GPU & Libraries</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Python Version:** {system_info['Python Version']}")
        st.markdown(f"**NumPy Version:** {system_info['NumPy Version']}")

        if "PyTorch Version" in system_info:
            st.markdown(
                f"**PyTorch Version:** {system_info['PyTorch Version']}")
            st.markdown(f"**CUDA Version:** {system_info['CUDA Version']}")

        if "CuPy Version" in system_info:
            st.markdown(f"**CuPy Version:** {system_info['CuPy Version']}")

        # Display GPU information if available
        if "Number of GPUs" in system_info and system_info["Number of GPUs"] > 0:
            st.markdown("**GPUs:**")
            for i in range(system_info["Number of GPUs"]):
                st.markdown(f"- {system_info[f'GPU {i}']}")

        st.markdown('</div>', unsafe_allow_html=True)

    # GPU availability status
    gpu_status_col1, gpu_status_col2 = st.columns(2)

    with gpu_status_col1:
        pytorch_gpu = has_pytorch_gpu()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">PyTorch GPU Status</div>',
                    unsafe_allow_html=True)
        if pytorch_gpu:
            st.success("✅ PyTorch can access GPU")
        else:
            st.error("❌ PyTorch cannot access GPU")
        st.markdown('</div>', unsafe_allow_html=True)

    with gpu_status_col2:
        cupy_gpu = has_cupy()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">CuPy GPU Status</div>',
                    unsafe_allow_html=True)
        if cupy_gpu:
            st.success("✅ CuPy can access GPU")
        else:
            st.error("❌ CuPy cannot access GPU")
        st.markdown('</div>', unsafe_allow_html=True)

# Matrix Operations Benchmark
elif page == "Matrix Operations Benchmark":
    st.header("📊 Matrix Operations Benchmark")

    st.markdown("""
    This section benchmarks common matrix operations on CPU vs GPU.
    Matrix operations are fundamental building blocks in machine learning and deep learning,
    and GPUs can significantly accelerate these operations through parallelization.
    """)

    # Benchmark options
    st.subheader("Benchmark Options")

    col1, col2 = st.columns(2)

    with col1:
        sizes = st.multiselect(
            "Matrix Sizes",
            options=[1000, 2000, 5000, 10000],
            default=[1000, 5000],
            help="Select the sizes of matrices to benchmark"
        )

    with col2:
        runs = st.slider(
            "Number of Runs",
            min_value=1,
            max_value=10,
            value=3,
            help="More runs provide more accurate average times but take longer"
        )

    # Run benchmark button
    run_matrix_benchmark = st.button(
        "Run Matrix Benchmark", key="run_matrix_benchmark")

    if run_matrix_benchmark or "matrix_benchmark_results" in st.session_state:
        # Show spinner while computing
        with st.spinner("Running benchmark..."):
            if run_matrix_benchmark:
                # Run the benchmark
                matrix_benchmark_results = run_matrix_operations_benchmark(
                    sizes=sizes, runs=runs)
                st.session_state.matrix_benchmark_results = matrix_benchmark_results
            else:
                # Get results from session state
                matrix_benchmark_results = st.session_state.matrix_benchmark_results

        # Display results
        st.subheader("Benchmark Results")

        # Show result table
        st.dataframe(matrix_benchmark_results)

        # Plot results
        fig = plot_matrix_benchmark_results(matrix_benchmark_results)
        st.pyplot(fig)

        # Show a summary
        st.subheader("Summary")

        # Calculate average speedup for valid operations
        valid_speedups = matrix_benchmark_results["speedup"][~np.isnan(
            matrix_benchmark_results["speedup"])]
        if len(valid_speedups) > 0:
            avg_speedup = valid_speedups.mean()
            max_speedup = valid_speedups.max()
            st.markdown(f"**Average GPU Speedup:** {avg_speedup:.2f}x")
            st.markdown(f"**Maximum GPU Speedup:** {max_speedup:.2f}x")

            st.markdown("""
            **Observations:**
            - Matrix multiplication shows the highest speedup on GPU, especially for larger matrices
            - The speedup increases with the size of the matrices
            - For very small operations, the overhead of transferring data to the GPU can outweigh the benefits
            """)
        else:
            st.warning(
                "No valid GPU speedup data available. GPU may not be accessible.")

# Neural Network from Scratch
elif page == "Neural Network from Scratch":
    st.header("🧠 Neural Network from Scratch")

    st.markdown("""
    This section demonstrates a neural network implemented from scratch,
    with options to run on either CPU or GPU. You can see firsthand the 
    performance difference when training the same model architecture.
    """)

    # Neural network training options
    st.subheader("Training Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        hidden_size = st.slider(
            "Hidden Layer Size",
            min_value=8,
            max_value=128,
            value=32,
            step=8,
            help="Number of neurons in the hidden layer"
        )

    with col2:
        epochs = st.slider(
            "Number of Epochs",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of training epochs"
        )

    with col3:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
            value=0.1,
            help="Step size for gradient updates"
        )

    # Run neural network training button
    run_nn_training = st.button("Train Neural Network", key="run_nn_training")

    if run_nn_training or "nn_benchmark_results" in st.session_state:
        # Show spinner while training
        with st.spinner("Training neural network on CPU and GPU..."):
            if run_nn_training:
                # Run the benchmark
                nn_benchmark_results = benchmark_neural_network(
                    epochs=epochs,
                    lr=learning_rate,
                    hidden_size=hidden_size
                )
                st.session_state.nn_benchmark_results = nn_benchmark_results
            else:
                # Get results from session state
                nn_benchmark_results = st.session_state.nn_benchmark_results

        # Display results
        st.subheader("Training Results")

        # Show training times
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="CPU Training Time",
                value=f"{nn_benchmark_results['cpu_time']:.2f} sec"
            )

        with col2:
            if not np.isnan(nn_benchmark_results['gpu_time']):
                st.metric(
                    label="GPU Training Time",
                    value=f"{nn_benchmark_results['gpu_time']:.2f} sec"
                )
            else:
                st.metric(
                    label="GPU Training Time",
                    value="N/A"
                )

        with col3:
            if not np.isnan(nn_benchmark_results['speedup']):
                st.metric(
                    label="Speedup",
                    value=f"{nn_benchmark_results['speedup']:.2f}x"
                )
            else:
                st.metric(
                    label="Speedup",
                    value="N/A"
                )

        # Show training loss curves
        st.subheader("Training Loss")
        loss_fig = plot_training_loss(
            nn_benchmark_results['cpu_loss_history'],
            nn_benchmark_results['gpu_loss_history'] if 'gpu_loss_history' in nn_benchmark_results else None
        )
        st.pyplot(loss_fig)

        # Show decision boundaries
        st.subheader("Model Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### CPU Model Decision Boundary")
            decision_boundary_cpu = plot_decision_boundary(
                nn_benchmark_results['model_cpu'],
                nn_benchmark_results['X'],
                nn_benchmark_results['y']
            )
            st.pyplot(decision_boundary_cpu)

        with col2:
            if nn_benchmark_results['model_gpu'] is not None:
                st.markdown("### GPU Model Decision Boundary")
                decision_boundary_gpu = plot_decision_boundary(
                    nn_benchmark_results['model_gpu'],
                    nn_benchmark_results['X'],
                    nn_benchmark_results['y']
                )
                st.pyplot(decision_boundary_gpu)
            else:
                st.warning(
                    "GPU model not available. GPU may not be accessible.")

elif page == "Neural Network Components":
    st.header("🔬 Neural Network Components")

    st.markdown("""
    This section breaks down the neural network into its components to analyze
    where GPU acceleration provides the most benefit.
    """)

    # Component benchmark options
    st.subheader("Benchmark Options")

    col1, col2 = st.columns(2)

    with col1:
        hidden_size = st.slider(
            "Hidden Layer Size",
            min_value=8,
            max_value=256,
            value=64,
            step=8,
            help="Number of neurons in the hidden layer"
        )

    with col2:
        samples = st.slider(
            "Number of Samples",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of repetitions for accurate timing"
        )

    # Run component benchmark button
    run_component_benchmark = st.button(
        "Run Component Benchmark", key="run_component_benchmark")

    if run_component_benchmark or "component_benchmark_results" in st.session_state:
        # Show spinner while computing
        with st.spinner("Benchmarking neural network components..."):
            if run_component_benchmark:
                # Generate data
                X, y = generate_data(n_samples=2000, noise=0.2)

                # Run the benchmark
                component_benchmark_results = benchmark_neural_network_components(
                    X=X, y=y, hidden_size=hidden_size, samples=samples
                )
                st.session_state.component_benchmark_results = component_benchmark_results
                st.session_state.benchmark_data = (X, y)
            else:
                # Get results from session state
                component_benchmark_results = st.session_state.component_benchmark_results
                X, y = st.session_state.benchmark_data

        # Display results
        st.subheader("Component Benchmark Results")

        # Show result table
        st.dataframe(component_benchmark_results)

        # Plot results
        fig = plot_nn_component_benchmark(component_benchmark_results)
        st.pyplot(fig)

        # Show the dataset
        st.subheader("Dataset")
        data_fig = plot_data(X, y)
        st.pyplot(data_fig)

        # Show a summary
        st.subheader("Analysis")

        # Calculate average speedup for valid operations
        valid_speedups = component_benchmark_results["speedup"][~np.isnan(
            component_benchmark_results["speedup"])]
        if len(valid_speedups) > 0:
            avg_speedup = valid_speedups.mean()
            max_speedup = valid_speedups.max()

            max_speedup_component = component_benchmark_results.loc[
                component_benchmark_results["speedup"].idxmax(), "operation"
            ]

            st.markdown(f"**Average Component Speedup:** {avg_speedup:.2f}x")
            st.markdown(
                f"**Maximum Component Speedup:** {max_speedup:.2f}x (for {max_speedup_component})")

            st.markdown("""
            **What makes neural networks well-suited for GPU acceleration?**
            
            1. **Parallelizable Operations**: Most neural network operations can be expressed as matrix and vector operations,
               which are highly parallelizable and perfect for GPU execution.
            
            2. **Data Independence**: Many calculations in neural networks can be performed independently,
               allowing efficient utilization of thousands of GPU cores.
            
            3. **Memory Locality**: Modern GPU architectures are designed to efficiently access and process
               data that exhibits spatial and temporal locality, which neural networks do.
            """)
        else:
            st.warning(
                "No valid GPU speedup data available. GPU may not be accessible.")

# About GPU Acceleration
elif page == "About GPU Acceleration":
    st.header("📚 About GPU Acceleration")

    st.markdown("""
    ## Why GPUs Matter for AI
    
    GPUs (Graphics Processing Units) have revolutionized the field of artificial intelligence and deep learning.
    Here's why they're so important:
    
    ### Architecture Differences: CPU vs GPU
    
    **CPU (Central Processing Unit):**
    - Few cores (typically 4-32)
    - Optimized for sequential serial processing
    - Complex instruction sets and branch prediction
    - Large cache memory
    
    **GPU (Graphics Processing Unit):**
    - Thousands of smaller, more efficient cores (can be 2,000+ on modern GPUs)
    - Designed for highly parallel operations
    - Simpler instruction sets focused on throughput
    - High memory bandwidth
    
    ### Perfect Match for Deep Learning
    
    Modern deep learning models require billions of calculations, most of which are matrix multiplications
    and other operations that can be parallelized. GPUs can perform these operations orders of magnitude 
    faster than CPUs.
    
    ### Libraries for GPU Acceleration
    
    Several libraries make it easy to leverage GPU computing:
    
    - **CUDA**: NVIDIA's platform for general-purpose computing on GPUs
    - **cuDNN**: NVIDIA's GPU-accelerated library for deep neural networks
    - **CuPy**: NumPy-compatible array library accelerated with CUDA
    - **PyTorch & TensorFlow**: Deep learning frameworks with GPU support
    
    ### When to Use GPU Acceleration
    
    GPU acceleration is particularly beneficial for:
    
    - Training large neural networks
    - Processing high-dimensional data
    - Running complex simulations
    - Real-time inference with deep models
    - Processing large batches of data in parallel
    """)

    # Display a comparison chart
    st.subheader("Typical Performance Comparison")

    # Sample data for illustration
    operations = ["Matrix Multiplication",
                  "Neural Network Training", "Data Preprocessing", "Inference"]
    cpu_times = [10.0, 300.0, 5.0, 2.0]
    gpu_times = [0.5, 30.0, 2.0, 0.2]

    # Create the comparison chart
    comp_fig = plot_comparison(
        cpu_times, gpu_times, operations, "CPU vs GPU Performance (Lower is Better)")
    st.plotly_chart(comp_fig)

    st.markdown("""
    ### Future of GPU Computing in AI
    
    As AI models continue to grow in size and complexity, GPU acceleration becomes increasingly essential.
    The latest developments include:
    
    - **Specialized AI chips**: Hardware designed specifically for machine learning workloads
    - **Multi-GPU systems**: Distributing workloads across multiple GPUs for even more parallelism
    - **GPU memory innovations**: New technologies to overcome memory limitations for larger models
    - **Optimized algorithms**: Software innovations to better utilize GPU architecture
    """)

# Add a footer
st.markdown("""
---
*This application is part of a technical presentation on AI Acceleration with GPU*
""")

# Execute the Streamlit app
if __name__ == "__main__":
    pass
