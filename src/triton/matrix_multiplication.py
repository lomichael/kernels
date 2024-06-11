import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, M, N, K):
	pid = tl.program_id(0)
	row = pid // M
	col = pid % N

	acc = tl.zeros((16,), dtype=tl.float32)
	for i in range(K):
		a = A[row * K + i]
		b = B[i * N + col]
		acc += a * b

	C[row * N + col] = acc

def run_matmul(A, B, C, M, N, K):
	grid = (M, N)
	matmul_kernel[grid](A, B, C, M, N, K)

def benchmarkMatrixMultiplyTriton(N):
	A = torch.ones((N, N), device='cuda', dtype=torch.float32)
	B = torch.ones((N, N), device='cuda', dtype=torch.float32)
	C = torch.zeros((N, N), device='cuda', dtype=torch.float32)

	start = time.time()
	run_matmul(A, B, C, N, N, N)
	end = time.time()

	return end - start
