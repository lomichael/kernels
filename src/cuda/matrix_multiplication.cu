#include <stdio.h>

__global__
void matrixMultiply(float *A, float *B, float *C, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;

	if (row < N && col < N) {
		for (int i = 0; i < N; i++) {
			sum += A[row * N + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	}
}

double benchmarkMatrixMultiplyCUDA(int N) {
	std::vector<float> h_A(N * N, 1.0f);
	std::vector<float> h_B(N * N, 1.0f);
	std::vector<float> h_C(N * N, 0.0f);
	float *d_A, *d_B, *d_C;

	cudaMalloc(&d_A, N * N * sizeof(float));
	cudaMalloc(&d_B, N * N * sizeof(float));
	cudaMalloc(&d_C, N * N * sizeof(float));

	cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

	auto start = std::chrono::high_resolution_clock::now();
	matrixMultiplyCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;

	cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return duation.count();
}	
