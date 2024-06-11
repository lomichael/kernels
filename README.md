# kernels
Benchmarked CUDA and Triton kernels.

![chiprace.png](/chiprace.png)

## Directory Structure
- `src/cuda`: CUDA kernel implementations
- `src/triton`: Triton kernel implementations
- `include/cuda`: CUDA header files
- `include/triton`: Triton header files
- `tests/cuda`: Unit tests for CUDA kernels
- `tests/triton`: Unit tests for Triton kernels
- `benchmarks/cuda`: Benchmarking scripts for CUDA kernels
- `benchmarks/triton`: Benchmarking scripts for Triton kernels
- `docs`: Documentation

## Implemented Kernels
- Matrix multiplication (GEMM)
- Convolution
	- 2D convolution
	- 3D convolution
- Reduction algorithms
	- sum
	- min
	- max
	- mean
- Prefix sum (scan)
- Sparse matrix operations
	- Sparse matrix-vector multiplication (SpMV)
	- Sparse matrix-matrix multiplication (SpMM)
- Batch normalization
- Activation functions
	- ReLU
	- Sigmoid
	- tanh
- Pooling layers
	- Max pooling
	- Average pooling
- Softmax
- Attention mechanisms
	- Scaled dot-product attention
- Optimizer algorithms
	- SGD
	- Adam
	- RMSProp
- Data augmentation algorithms
	- Random cropping
	- Flipping
	- Rotation
	- Normalization
	- Color jittering
