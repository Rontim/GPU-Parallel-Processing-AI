import tensorflow as tf
import sys  # To check Python version

print("TensorFlow Version:", tf.__version__)
print("Python Version:", sys.version)

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))

if gpus:
    print("GPU(s) detected by TensorFlow:", gpus)
    # Check if TensorFlow was built with CUDA support (redundant if GPU is listed, but informative)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
else:
    print("TensorFlow did NOT detect any GPU.")
    print("Ensure prerequisites are met: NVIDIA driver, CUDA Toolkit, cuDNN installed correctly and versions match TensorFlow requirements.")
    print("Check PATH environment variable includes CUDA bin/lib paths.")
