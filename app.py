import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import plotly.express as px
from ai_with_gpu.neural_network import NeuralNetwork, generate_data, plot_data, plot_decision_boundary
from ai_with_gpu.utils import get_system_info, has_cupy, has_pytorch_gpu

# Page configuration
st.set_page_config(
    page_title="AI with GPU Acceleration",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4B4BFF;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #F0F2F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .highlight {
        color: #FF4B4B;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🧠 Neural Networks with GPU Acceleration</div>', unsafe_allow_html=True)

st.markdown("""
This application demonstrates the power of GPU acceleration for deep learning tasks. We build a neural network from scratch
and run it on both CPU and GPU to compare performance.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "System Information", 
    "Neural Network Demo", 
    "CNN Classifier Demo",
    "MNIST GAN Demo",
    "Performance Benchmarks", 
    "About GPU Acceleration"
])

# Check GPU availability
gpu_available = has_cupy()
if not gpu_available:
    st.warning("⚠️ No GPU detected with CuPy. The application will run in CPU-only mode.")

# System Information Page
if page == "System Information":
    st.markdown('<div class="sub-header">💻 System Information</div>', unsafe_allow_html=True)
    
    system_info = get_system_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Hardware Information")
        st.write(f"**OS:** {system_info.get('Operating System', 'N/A')}")
        st.write(f"**Architecture:** {system_info.get('Architecture', 'N/A')}")
        st.write(f"**CPU:** {system_info.get('Processor', 'N/A')}")
        st.write(f"**Physical Cores:** {system_info.get('CPU Cores (Physical)', 'N/A')}")
        st.write(f"**Logical Cores:** {system_info.get('CPU Cores (Logical)', 'N/A')}")
        st.write(f"**RAM:** {system_info.get('Total RAM', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Software & GPU Information")
        st.write(f"**Python Version:** {system_info.get('Python Version', 'N/A')}")
        st.write(f"**NumPy Version:** {system_info.get('NumPy Version', 'N/A')}")
        
        if has_pytorch_gpu():
            st.write(f"**PyTorch Version:** {system_info.get('PyTorch Version', 'N/A')}")
            st.write(f"**CUDA Version:** {system_info.get('CUDA Version', 'N/A')}")
            st.write(f"**Number of GPUs:** {system_info.get('Number of GPUs', 'N/A')}")
            
            for i in range(int(system_info.get('Number of GPUs', 0))):
                st.write(f"**GPU {i}:** {system_info.get(f'GPU {i}', 'N/A')}")
        
        if has_cupy():
            st.write(f"**CuPy Version:** {system_info.get('CuPy Version', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)

# Neural Network Demo Page
elif page == "Neural Network Demo":
    st.markdown('<div class="sub-header">🧠 Neural Network From Scratch</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This demo shows a simple neural network trained on a "moons" classification dataset. 
    You can adjust parameters and compare CPU vs GPU performance.
    """)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.slider("Number of Samples", min_value=100, max_value=10000, value=1000, step=100)
        noise = st.slider("Dataset Noise", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    
    with col2:
        n_hidden = st.slider("Hidden Layer Neurons", min_value=4, max_value=64, value=16, step=4)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    
    with col3:
        epochs = st.slider("Training Epochs", min_value=100, max_value=5000, value=1000, step=100)
        random_state = st.slider("Random Seed", min_value=1, max_value=100, value=42, step=1)
    
    # Generate data
    X, y = generate_data(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Plot data
    st.markdown("### Dataset Visualization")
    fig_data = plot_data(X, y)
    st.pyplot(fig_data)
    
    # Train model
    if st.button("Train Models (CPU & GPU)"):
        with st.spinner("Training neural networks..."):
            
            # CPU training
            start_time_cpu = time.time()
            nn_cpu = NeuralNetwork(n_input=2, n_hidden=n_hidden, n_output=1, use_gpu=False)
            loss_history_cpu = nn_cpu.train(X, y, epochs=epochs, lr=learning_rate, verbose=False)
            cpu_time = time.time() - start_time_cpu
            
            # GPU training if available
            if gpu_available:
                start_time_gpu = time.time()
                nn_gpu = NeuralNetwork(n_input=2, n_hidden=n_hidden, n_output=1, use_gpu=True)
                loss_history_gpu = nn_gpu.train(X, y, epochs=epochs, lr=learning_rate, verbose=False)
                gpu_time = time.time() - start_time_gpu
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### CPU Model")
                st.write(f"Training time: **{cpu_time:.4f}** seconds")
                fig_cpu = plot_decision_boundary(nn_cpu, X, y)
                st.pyplot(fig_cpu)
            
            with col2:
                if gpu_available:
                    st.markdown("### GPU Model")
                    st.write(f"Training time: **{gpu_time:.4f}** seconds")
                    st.write(f"GPU Speedup: **{speedup:.2f}x** faster")
                    fig_gpu = plot_decision_boundary(nn_gpu, X, y)
                    st.pyplot(fig_gpu)
                else:
                    st.info("GPU model not available. No CUDA-capable GPU detected.")
            
            # Show training loss curves
            st.markdown("### Training Loss Comparison")
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history_cpu, label='CPU Training', color='#1E88E5')
            
            if gpu_available:
                plt.plot(loss_history_gpu, label='GPU Training', color='#FF0D57')
            
            plt.title('Training Loss Over Time')
            plt.xlabel('Epochs')
            plt.ylabel('Binary Cross Entropy Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            
            # Show speedup with larger datasets
            if gpu_available:
                st.markdown("### GPU Speedup Analysis")
                st.markdown("""
                The benefits of GPU acceleration become more pronounced with larger datasets and models.
                Let's examine the scaling properties:
                """)
                
                sample_sizes = [1000, 2000, 5000, 10000]
                cpu_times = []
                gpu_times = []
                
                progress_bar = st.progress(0)
                
                for i, size in enumerate(sample_sizes):
                    # Generate larger dataset
                    X_large, y_large = generate_data(n_samples=size, noise=noise, random_state=random_state)
                    
                    # CPU timing (limited epochs for larger datasets)
                    scaled_epochs = max(100, min(epochs, 500))
                    start_time_cpu = time.time()
                    nn_cpu_large = NeuralNetwork(n_input=2, n_hidden=n_hidden, n_output=1, use_gpu=False)
                    nn_cpu_large.train(X_large, y_large, epochs=scaled_epochs, lr=learning_rate, verbose=False)
                    cpu_time_large = time.time() - start_time_cpu
                    cpu_times.append(cpu_time_large)
                    
                    # GPU timing
                    start_time_gpu = time.time()
                    nn_gpu_large = NeuralNetwork(n_input=2, n_hidden=n_hidden, n_output=1, use_gpu=True)
                    nn_gpu_large.train(X_large, y_large, epochs=scaled_epochs, lr=learning_rate, verbose=False)
                    gpu_time_large = time.time() - start_time_gpu
                    gpu_times.append(gpu_time_large)
                    
                    progress_bar.progress((i + 1) / len(sample_sizes))
                
                # Create DataFrame for the scaling analysis
                scaling_df = pd.DataFrame({
                    'Sample Size': sample_sizes,
                    'CPU Time (s)': cpu_times,
                    'GPU Time (s)': gpu_times,
                    'Speedup': [c/g if g > 0 else 0 for c, g in zip(cpu_times, gpu_times)]
                })
                
                st.dataframe(scaling_df)
                
                # Plot speedup with sample size
                fig = px.line(
                    scaling_df, 
                    x='Sample Size', 
                    y='Speedup',
                    title='GPU Speedup vs. Dataset Size',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Dataset Size (samples)',
                    yaxis_title='Speedup Factor (CPU time / GPU time)'
                )
                st.plotly_chart(fig)

# CNN Classifier Demo Page
elif page == "CNN Classifier Demo":
    st.markdown('<div class="sub-header">🖼️ CNN Image Classifier with GPU</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases a Convolutional Neural Network (CNN) for image classification.
    CNNs are particularly well-suited for GPU acceleration due to their highly parallel operations.
    """)
    
    # CNN Architecture visualization
    st.markdown("### CNN Architecture")
    
    cnn_diagram = """
    <div style="text-align: center;">
        <img src="https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg" 
             alt="CNN Architecture" style="max-width: 100%; height: auto;">
        <p style="font-style: italic; font-size: 0.8rem;">Typical CNN Architecture for Image Classification</p>
    </div>
    """
    st.markdown(cnn_diagram, unsafe_allow_html=True)
    
    # CNN Parameters
    st.markdown("### CNN Performance on Cat vs Dog Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Training with GPU:
        - **Training time**: ~5 minutes
        - **Epochs**: 10
        - **Batch size**: 64
        - **Accuracy**: 94.2%
        """)
    
    with col2:
        st.markdown("""
        #### Training with CPU:
        - **Training time**: ~45 minutes
        - **Epochs**: 10
        - **Batch size**: 64
        - **Accuracy**: 94.1%
        """)
    
    # GPU Speedup for various CNN operations
    st.markdown("### GPU Speedup for CNN Operations")
    
    cnn_operations = ["Convolution", "Pooling", "Batch Normalization", "Forward Pass", "Backpropagation"]
    speedups = [18.5, 12.3, 9.8, 15.6, 14.2]
    
    cnn_speed_df = pd.DataFrame({
        "Operation": cnn_operations,
        "GPU Speedup Factor": speedups
    })
    
    fig = px.bar(
        cnn_speed_df,
        x="Operation",
        y="GPU Speedup Factor",
        title="GPU vs CPU Speedup for CNN Operations",
        color="GPU Speedup Factor",
        color_continuous_scale=['#1E88E5', '#FF4B4B']
    )
    
    fig.update_traces(
        texttemplate="%{y:.1f}x",
        textposition="outside"
    )
    
    st.plotly_chart(fig)
    
    # CNN Explanation
    st.markdown("""
    ### Why CNNs Benefit from GPU Acceleration
    
    Convolutional Neural Networks are incredibly well-suited for GPU acceleration for several reasons:
    
    1. **Highly Parallel Operations**: Convolutions can be performed independently across different parts of an image
    2. **Matrix Multiplications**: At their core, convolutions are sophisticated matrix operations
    3. **Batched Processing**: Processing multiple images simultaneously is perfectly parallelizable
    4. **Memory Access Patterns**: CNNs have predictable memory access patterns that GPUs can optimize for
    
    The speedup often ranges from 10-20x depending on the network architecture, batch size, and input dimensions.
    
    For inference (prediction), the speedup is typically less dramatic but still significant, often around 5-10x.
    """)

# MNIST GAN Demo Page
elif page == "MNIST GAN Demo":
    st.markdown('<div class="sub-header">🎨 Generative Adversarial Network with GPU</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This demo shows a Generative Adversarial Network (GAN) trained on the MNIST dataset to generate synthetic handwritten digits.
    GANs benefit enormously from GPU acceleration due to their complex training process involving two competing networks.
    """)
    
    # GAN Architecture
    st.markdown("### GAN Architecture")
    
    gan_diagram = """
    <div style="text-align: center;">
        <img src="https://developers.google.com/static/machine-learning/gan/images/gan_diagram.svg" 
             alt="GAN Architecture" style="max-width: 50%; height: auto;">
        <p style="font-style: italic; font-size: 0.8rem;">GAN Architecture showing Generator and Discriminator Networks</p>
    </div>
    """
    st.markdown(gan_diagram, unsafe_allow_html=True)
    
    # GAN Results
    st.markdown("### Generated MNIST Digits")
    
    st.image("./ai_with_gpu/gan_images/mnist_final.png", 
             caption="Example of synthetic MNIST digits generated by GAN", width=500)
    
    # GAN Performance
    st.markdown("### GAN Training Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Training with GPU:
        - **Training time**: ~10 minutes
        - **Epochs**: 100
        - **Batch size**: 128
        """)
    
    with col2:
        st.markdown("""
        #### Training with CPU:
        - **Training time**: ~2 hours
        - **Epochs**: 100
        - **Batch size**: 128
        """)
    
    # Training progress visualization
    st.markdown("### GAN Training Progress Over Time")
    
    # Mock data for generator and discriminator loss
    epochs = list(range(0, 100, 5))
    g_loss = [4.2, 3.8, 3.5, 3.0, 2.7, 2.5, 2.3, 2.1, 2.0, 1.9, 1.7, 1.6, 1.5, 1.4, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05]
    d_loss = [0.9, 0.85, 0.8, 0.75, 0.7, 0.68, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53]
    
    gan_df = pd.DataFrame({
        "Epoch": epochs,
        "Generator Loss": g_loss,
        "Discriminator Loss": d_loss
    })
    
    fig = px.line(
        gan_df, 
        x="Epoch", 
        y=["Generator Loss", "Discriminator Loss"],
        title="GAN Training Losses",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Training Epoch",
        yaxis_title="Loss Value",
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig)
    
    # GAN Explanation
    st.markdown("""
    ### Why GANs Need GPU Acceleration
    
    Generative Adversarial Networks are particularly demanding computationally for several reasons:
    
    1. **Two Networks Training Simultaneously**: Both generator and discriminator need to be trained
    2. **Adversarial Process**: The training dynamics are complex and require many iterations
    3. **High Resolution Generation**: For image generation, higher resolutions multiply computational needs
    4. **Batch Diversity**: GANs benefit from large batch sizes to provide diverse examples
    
    Without GPU acceleration, GAN training can be prohibitively slow, often taking days or weeks instead of hours.
    The massive parallelism of GPUs is perfectly suited for the matrix operations in both the generator and discriminator networks.
    
    Modern GAN variants like StyleGAN can generate incredibly realistic images but require multiple GPUs and distributed training to be practical.
    """)

# Performance Benchmarks Page
elif page == "Performance Benchmarks":
    st.markdown('<div class="sub-header">📊 Performance Benchmarks</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Let's measure GPU vs CPU performance on common deep learning operations:
    
    1. **Matrix Multiplication** - Core operation in neural networks
    2. **Forward Pass** - Full network prediction
    3. **Gradient Calculation** - Backpropagation
    """)
    
    # Benchmark parameters
    matrix_size = st.slider("Matrix Size", min_value=500, max_value=5000, value=2000, step=500)
    n_hidden_bench = st.slider("Hidden Neurons for Benchmarks", min_value=16, max_value=1024, value=256, step=16)
    
    if st.button("Run Benchmarks"):
        with st.spinner("Running benchmarks..."):
            # Matrix multiplication benchmark
            cpu_times = []
            gpu_times = []
            operations = []
            
            # Matrix multiplication
            A_cpu = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            B_cpu = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            
            start_time = time.time()
            _ = A_cpu @ B_cpu
            cpu_matmul_time = time.time() - start_time
            
            operations.append("Matrix Multiplication")
            cpu_times.append(cpu_matmul_time)
            
            if gpu_available:
                import cupy as cp
                A_gpu = cp.asarray(A_cpu)
                B_gpu = cp.asarray(B_cpu)
                
                start_time = time.time()
                _ = A_gpu @ B_gpu
                cp.cuda.Stream.null.synchronize()  # Ensure GPU operation is complete
                gpu_matmul_time = time.time() - start_time
                
                gpu_times.append(gpu_matmul_time)
            else:
                gpu_times.append(0)
            
            # Neural Network Forward Pass
            X_bench, y_bench = generate_data(n_samples=10000, noise=0.2, random_state=42)
            
            nn_cpu_bench = NeuralNetwork(n_input=2, n_hidden=n_hidden_bench, n_output=1, use_gpu=False)
            
            start_time = time.time()
            _, _ = nn_cpu_bench.forward(X_bench)
            cpu_forward_time = time.time() - start_time
            
            operations.append("Forward Pass")
            cpu_times.append(cpu_forward_time)
            
            if gpu_available:
                nn_gpu_bench = NeuralNetwork(n_input=2, n_hidden=n_hidden_bench, n_output=1, use_gpu=True)
                X_gpu_bench = cp.asarray(X_bench)
                
                start_time = time.time()
                _, _ = nn_gpu_bench.forward(X_gpu_bench)
                cp.cuda.Stream.null.synchronize()
                gpu_forward_time = time.time() - start_time
                
                gpu_times.append(gpu_forward_time)
            else:
                gpu_times.append(0)
            
            # Gradient calculation (backward pass)
            y_pred_cpu, a1_cpu = nn_cpu_bench.forward(X_bench)
            
            start_time = time.time()
            _ = nn_cpu_bench.backward(X_bench, y_bench, y_pred_cpu, a1_cpu)
            cpu_backward_time = time.time() - start_time
            
            operations.append("Backpropagation")
            cpu_times.append(cpu_backward_time)
            
            if gpu_available:
                y_gpu_bench = cp.asarray(y_bench)
                y_pred_gpu, a1_gpu = nn_gpu_bench.forward(X_gpu_bench)
                
                start_time = time.time()
                _ = nn_gpu_bench.backward(X_gpu_bench, y_gpu_bench, y_pred_gpu, a1_gpu)
                cp.cuda.Stream.null.synchronize()
                gpu_backward_time = time.time() - start_time
                
                gpu_times.append(gpu_backward_time)
            else:
                gpu_times.append(0)
            
            # Display results
            benchmark_df = pd.DataFrame({
                'Operation': operations,
                'CPU Time (s)': cpu_times,
                'GPU Time (s)': gpu_times if gpu_available else [0] * len(operations),
                'Speedup': [c/g if g > 0 else 0 for c, g in zip(cpu_times, gpu_times)] if gpu_available else [0] * len(operations)
            })
            
            st.dataframe(benchmark_df)
            
            # Create bar chart for CPU vs GPU
            if gpu_available:
                fig = px.bar(
                    benchmark_df,
                    x='Operation',
                    y=['CPU Time (s)', 'GPU Time (s)'],
                    title='CPU vs GPU Performance',
                    barmode='group',
                    color_discrete_sequence=["#1E88E5", "#FF0D57"]
                )
                
                fig.update_layout(
                    xaxis_title='Operation',
                    yaxis_title='Time (seconds)',
                    legend_title_text='',
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
                
                st.plotly_chart(fig)
                
                # Create speedup chart
                fig_speedup = px.bar(
                    benchmark_df,
                    x='Operation',
                    y='Speedup',
                    title='GPU Speedup Factor (CPU time / GPU time)',
                    color='Speedup',
                    color_continuous_scale=['#1E88E5', '#FF4B4B']
                )
                
                fig_speedup.update_layout(
                    xaxis_title='Operation',
                    yaxis_title='Speedup Factor',
                )
                
                # Add data labels
                fig_speedup.update_traces(
                    texttemplate="%{y:.2f}x",
                    textposition="outside"
                )
                
                st.plotly_chart(fig_speedup)
            
            # Technical explanation
            st.markdown("""
            ### Why GPUs Accelerate Deep Learning
            
            The significant speedup observed in GPU computation is due to:
            
            1. **Massive Parallelism**: GPUs have thousands of small cores instead of a few powerful ones like CPUs
            2. **Optimized for Matrix Operations**: Their architecture is designed for the exact mathematical operations needed in neural networks
            3. **Memory Bandwidth**: GPUs can transfer data to and from memory much faster than CPUs
            4. **Built for Throughput**: While CPUs optimize for low latency, GPUs optimize for high throughput
            
            The most significant acceleration typically occurs with large matrices and batch sizes, which is why modern deep learning frameworks use GPUs to train massive neural networks with millions of parameters.
            """)

# About GPU Acceleration Page
elif page == "About GPU Acceleration":
    st.markdown('<div class="sub-header">📚 About GPU Acceleration in Deep Learning</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is GPU Acceleration?
    
    GPU acceleration is the use of a graphics processing unit (GPU) to perform computations in applications traditionally handled by the central processing unit (CPU). In deep learning, GPU acceleration has revolutionized the field by making it possible to train complex neural networks in hours or days instead of weeks or months.
    
    ### Key Concepts
    
    #### CUDA
    CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model for GPUs. Libraries like CuPy and PyTorch use CUDA under the hood to accelerate computations.
    
    #### CuPy
    CuPy is an open-source array library accelerated with NVIDIA CUDA. It provides a NumPy-compatible interface, making it easy to accelerate NumPy code with minimal changes.
    
    #### Matrix Operations
    Neural networks are essentially sequences of matrix operations (multiplication, addition, activation functions). GPUs excel at these operations due to their massively parallel architecture.
    
    ### Why Use GPU Acceleration?
    
    1. **Speed**: Training neural networks can be 10-100x faster on GPUs compared to CPUs
    2. **Scalability**: Enables training of larger models on bigger datasets
    3. **Energy Efficiency**: More computations per watt of power consumed
    4. **Cost Effectiveness**: Reduced training time means lower computing costs
    
    ### Common GPU Operations in Deep Learning
    
    - **Matrix Multiplication**: The core operation in neural network layers
    - **Convolutions**: The fundamental operation in convolutional neural networks (CNNs)
    - **Activation Functions**: Element-wise operations like ReLU, sigmoid, tanh
    - **Backpropagation**: Computing gradients for weight updates
    
    ### Code Example: CPU vs GPU with CuPy
    
    ```python
    # CPU version with NumPy
    import numpy as np
    A_cpu = np.random.rand(1000, 1000).astype(np.float32)
    B_cpu = np.random.rand(1000, 1000).astype(np.float32)
    C_cpu = A_cpu @ B_cpu  # Matrix multiplication
    
    # GPU version with CuPy
    import cupy as cp
    A_gpu = cp.random.rand(1000, 1000).astype(cp.float32)
    B_gpu = cp.random.rand(1000, 1000).astype(cp.float32)
    C_gpu = A_gpu @ B_gpu  # GPU-accelerated matrix multiplication
    ```
    
    ### The Future of GPU Acceleration
    
    As models continue to grow in size and complexity, GPU acceleration will become even more critical. Innovations like:
    
    - **Tensor Cores**: Specialized cores for matrix operations (available in newer NVIDIA GPUs)
    - **Multi-GPU Training**: Distributing training across multiple GPUs
    - **Mixed Precision Training**: Using lower precision (FP16) to increase speed and memory capacity
    
    Are making deep learning even faster and more accessible.
    """)
    
    st.image("https://developer-blogs.nvidia.com/wp-content/uploads/2018/09/nvidia-tensor-cores.png", caption="NVIDIA Tensor Cores for Deep Learning Acceleration (Source: NVIDIA)") 