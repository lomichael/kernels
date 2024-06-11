#include <iostream>

extern double benchmarkMatrixMultiplyCPU(int N);
extern double benchmarkMatrixMultiplyCUDA(int N);
extern double benchmarkMatrixMultiplyTriton(int N);

double calculateFLOPS(double time, int N) {
	return (2.0 * N * N * N) / time;
}

double calculateBandwidth(double time, int N) {
	double dataSize = 3 * N * N * sizeof(float);
	return dataSize / time / (1024 * 1024 * 1024); // GB/s
}

int main() {
	int N = 1024;
	std::cout << "Benchmarking matrix multiplication with N = " << N << std::endl;

	double cpuTime = benchmarkMatrixMultiplyCPU(N);
	double cpuFLOPS = calculateFLOPS(cpuTime, N);
	double cpuBandwidth = calculateBandwidth(cpuTime, N);
	std::cout << "CPU Time: " << cpuTime << " seconds, FLOPS: " << cpuFLOPS << ", Bandwidth: " << cpuBandwidth << " GB/s" << std::endl;

	double cudaTime = benchmarkMatrixMultiplyCUDA(N);
	double cudaFLOPS = calculateFLOPS(cudaTime, N);
	double cudaBandwidth = calculateBandwidth(cudaTime, N);
	std::cout << "CUDA Time: " << cudaTime << " seconds, FLOPS: " << cudaFLOPS << ", Bandwidth: " << cudaBandwidth << " GB/s" << std::endl;

	double tritonTime = benchmarkMatrixMultiplyTriton(N);
	double tritonFLOPS = calculateFLOPS(tritonTime, N);
	double tritonBandwidth = calculateBandwidth(tritonTime, N);
	std::cout << "Triton Time: " << tritonTime << " seconds, FLOPS: " << tritonFLOPS << ", Bandwidth: " << tritonBandwidth << " GB/s" << std::endl;

	return 0;
}
