# GPU Parallel Processing in AI

This repository explores the use of **GPU parallel processing** in the context of **Artificial Intelligence (AI)**, specifically leveraging GPUs for accelerating computations in deep learning tasks. This project is designed for hands-on experimentation using both **Google Colab** and **school lab GPUs**. The goal is to gain a practical understanding of how GPUs can be used for tasks like training AI models faster, performing large-scale matrix operations, and improving the efficiency of parallel algorithms.

## Topics Covered

- **Introduction to GPU Parallel Programming**
  - Basics of GPU architecture
  - Introduction to **CUDA** and **OpenCL**
  - Writing simple parallel programs (e.g., matrix multiplication, parallel reductions)

- **AI with GPU Acceleration**
  - Accelerating AI tasks with **PyTorch** and **TensorFlow**
  - Training models like Convolutional Neural Networks (CNNs) and Transformers using GPUs
  - Fine-tuning models for better performance

- **Benchmarking GPU Performance**
  - Comparing **CPU vs GPU** performance in deep learning tasks
  - Profiling and optimizing GPU-based AI applications
  - Using tools like `nvprof`, `nvidia-smi`, and `tensorboard` for performance monitoring

- **Practical Applications and Experiments**
  - Hands-on code examples for using GPUs in AI applications
  - Training deep learning models on **Google Colab** (with GPU) and **school lab GPUs**
  - Benchmarks and experiments to measure speedup from GPU acceleration

## Repository Structure

```markdown
/GPU-Parallel-Processing-AI
  ├── README.md          # This file
  ├── gpu_programming/    # Basic GPU programming exercises
  ├── ai_with_gpu/        # AI models accelerated with GPU
  ├── benchmarking/       # Performance benchmarking and profiling
  ├── docs/               # Detailed documentation and setup guides
  └── LICENSE             # Open-source license
```

## Getting Started

### **Google Colab:**

You can start experimenting with GPUs using Google Colab. Simply open the respective notebook in the `colab_notebooks/` folder to begin training models or running experiments on the cloud-based GPU.

### **School Lab GPUs:**

For local experimentation, make sure your school lab GPU has the necessary software installed:

- **CUDA Toolkit**
- **cuDNN** (for AI tasks)
- **PyTorch or TensorFlow** (for deep learning models)
- **nvidia-smi** (to monitor GPU usage)

Follow the setup instructions in the `/docs/setup.md` file to configure your local GPU environment.

## How to Contribute

1. Fork the repository and clone it to your local machine.
2. Add new experiments, benchmarks, or improvements.
3. Create a pull request with your changes.

Feel free to submit issues or suggestions for further improvements and new experiments.

## 🚀 AI Acceleration with Streamlit

A Streamlit application that demonstrates the power of GPU acceleration for AI and machine learning tasks. This interactive app compares CPU vs GPU performance for various operations and provides visual insights into how GPUs accelerate AI workloads.

## 📋 Features

- **System Information**: Displays detailed information about your system, including CPU, RAM, and GPU detection.
- **Matrix Operations Benchmark**: Compares CPU vs GPU performance for common matrix operations.
- **Neural Network from Scratch**: Builds and trains a neural network on both CPU and GPU, showing real-time performance comparisons.
- **Neural Network Components**: Breaks down neural network training into components to analyze where GPU acceleration provides the most benefit.
- **About GPU Acceleration**: Educational content explaining GPU architecture and its benefits for AI.

## 🛠️ Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for comparisons)
- Dependencies listed in `requirements.txt`

## 🔧 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Rontim/GPU-Parallel-Processing-AI.git
   cd GPU-Parallel-Processing-AI
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate
   
   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Note: If you have a GPU, make sure to install the correct version of CuPy that matches your CUDA version:

   ```bash
   # For CUDA 11.x
   pip install cupy-cuda11x
   
   # For CUDA 12.x
   pip install cupy-cuda12x
   ```

## 🚀 Running the Application

Run the Streamlit app:

```bash
streamlit run ai_with_gpu/ai_with_gpu.py
```

The application will open in your default web browser. If it doesn't open automatically, you can access it at `http://localhost:8501`.

## 🧠 Neural Network Implementation

The application includes a neural network implemented from scratch with both CPU and GPU support:

- Fully connected network architecture
- Forward and backward propagation
- Gradient descent optimization
- Binary classification for the "moons" dataset
- Interactive visualization of decision boundaries

## 📊 Benchmarks

The benchmarking sections provide real-time comparisons between CPU and GPU performance for:

- Matrix operations (multiplication, element-wise operations, transpose)
- Neural network training and inference
- Individual components of neural network (forward pass, backward pass, parameter updates)

## 📝 Notes

- GPU acceleration requires a CUDA-compatible GPU and appropriate drivers.
- The speedup from GPU acceleration varies depending on the operation and data size.
- For very small operations, the overhead of transferring data to the GPU may outweigh the benefits.

## 📚 References

- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [GPU-Accelerated Computing with Python](https://developer.nvidia.com/how-to-cuda-python)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Rontim/GPU-Parallel-Processing-AI/issues).

## 📜 License

This project is [MIT](LICENSE) licensed.
