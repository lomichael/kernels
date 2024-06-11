#include <vector>
#include <chrono>

void matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			float sum = 0.0;
			for (int i = 0; i < N; ++i) {
				sum += A[row * N + i] * B[i * N + col];
			}
			C[row * N + col] = sum;
		}
	}
}

double benchmarkMatrixMutliplyCPU(int N) {
	std::vector<float> A(N * N, 1.0f);
	std::vector<float> B(N * N, 1.0f);
	std::vector<float> C(N * N, 0.0f);

	auto start = std::chrono::high_resolution_clock::now();
	matrixMultiplyCPU(A, B, C, N);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;

	return duration.count();
}
