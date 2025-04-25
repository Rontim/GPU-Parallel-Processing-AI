# AI with GPU Acceleration - Streamlit Demo

## Overview

This Streamlit application demonstrates the power of GPU acceleration for deep learning tasks. It provides an interactive platform to visualize and understand the performance benefits of using GPUs for neural network training and inference.

## Features

- **System Information**: View detailed hardware and software information, including GPU specifications
- **Neural Network Demo**: Train a simple neural network on both CPU and GPU, and visualize the performance difference
- **CNN Classifier Demo**: Explore how convolutional neural networks benefit from GPU acceleration for image classification
- **MNIST GAN Demo**: Visualize how Generative Adversarial Networks leverage GPUs to generate synthetic images
- **Performance Benchmarks**: Run benchmarks on common deep learning operations to see the GPU speedup
- **Educational Resources**: Learn about GPU acceleration in deep learning, CUDA, and CuPy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Rontim/GPU-Parallel-Processing-AI.git
   cd GPU-Parallel-Processing-AI
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: The `cupy-cuda11x` package requires CUDA 11.x to be installed. If you have a different CUDA version, replace with the appropriate package (e.g., `cupy-cuda12x`).

## Running the Application

Start the Streamlit app:
```
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## GPU Requirements

For GPU acceleration, you need:
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 11.x recommended)
- cuDNN library (for advanced deep learning operations)

The application will work without a GPU, but will only demonstrate CPU performance.

## Technical Details

The application uses:
- **CuPy**: NumPy-compatible array library accelerated with NVIDIA CUDA
- **Streamlit**: For the interactive web interface
- **Matplotlib/Plotly**: For data visualization
- **NumPy**: For CPU-based computations
- **Scikit-learn**: For dataset generation
- **TensorFlow/Keras**: For CNN and GAN implementations

The demos include:
1. A simple binary classifier neural network built from scratch
2. A convolutional neural network for image classification 
3. A generative adversarial network for creating synthetic MNIST digits

## For Technical Presentations

This application is designed to be used in technical presentations about GPU acceleration in AI, demonstrating:
1. The performance gap between CPU and GPU computation
2. How the speedup scales with dataset size and model complexity
3. Which neural network architectures (MLP, CNN, GAN) benefit most from GPU acceleration
4. Real-time comparison of CPU vs GPU training performance 