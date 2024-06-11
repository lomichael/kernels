# kernels
Benchmarked CUDA and Triton kernels.

![chiprace.jpeg](/chiprace.jpeg)

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

## Visualizing Benchmark Results
The benchmark results are visualized using Matplotlib and Seaborn. The following plots are available:
- Execution Time
- FLOPS
- Memory Bandwidth

## How to Run
To reproduce the benchmarks and visualizations, follow these steps:

1. Run the benchmarks:
```sh
./run_benchmarks.sh
```

2. Visualize the results:
```sh
python visualize_benchmarks.py
```
