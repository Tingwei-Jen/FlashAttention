#include "attention.h"

#define BR 64
#define BC 32

namespace ATTENTION {

__global__ void flashAttention_kernel(float* Output,  float* m, float* l, 
                                      const float* Q, const float* K, const float* V, 
                                      int N, const int d, int Tr, int Tc, float softmaxScale) {
    int tid = threadIdx.x;                  
    int numOfThreads = blockDim.x;
    
    // shared memory
    extern __shared__ float sram[];
    float* sQ = sram;
    float* sK = sQ + BR * d;
    float* sV = sK + BC * d;
    float* S = sV + BC * d;

    // number of rows for loading K one time
    int s_stride_size = numOfThreads / d;
    // index of stride in shared memory
    int rowIdx_s = tid / d;
    int colIdx_s = tid % d;

    for (int j = 0; j < Tc; j++) {

        // load K and V in shared memory
        for (int k = 0; k < BC; k += s_stride_size) {
            sK[colIdx_s + (k + rowIdx_s) * d] = K[colIdx_s + (k + j * BC + rowIdx_s) * d];
            sV[colIdx_s + (k + rowIdx_s) * d] = V[colIdx_s + (k + j * BC + rowIdx_s) * d];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {

            // load Q in shared memory thread by thread
            for (int x = 0; x < d; x++) {
                sQ[tid * d + x] = Q[(i * BR + tid) * d + x];
            }

            // find local max
            float local_max = -INFINITY;

            // S = Q * K^T thread by thread
            for (int y = 0; y < BC; y++) {
                float s = 0.0f;
                for (int x = 0; x < d; x++) {
                    s += sQ[tid * d + x] * sK[y * d + x];
                }
                s *= softmaxScale;
                S[tid * BC + y] = s;

                if (s > local_max) {
                    local_max = s;
                }
            }

            // P = exp(S - max(S));
            // find local sum
            float local_sum = 0.0f;
            for (int y = 0; y < BC; y++) {
                S[tid * BC + y] = expf(S[tid * BC + y] - local_max);
                local_sum += S[tid * BC + y];
            }

            float global_max_prev = m[i * BR + tid];
            float global_sum_prev = l[i * BR + tid];

            // compute new max and sum
            float global_max_new = fmaxf(global_max_prev, local_max);
            float local_max_diff = local_max - global_max_new;
            float global_max_diff = global_max_prev - global_max_new;
            float exp_local_max_diff = expf(local_max_diff);
            float exp_global_max_diff = expf(global_max_diff);
            float global_sum_new = exp_local_max_diff * local_sum + exp_global_max_diff * global_sum_prev;

            // write output l and m. each thread generates 1 * d output
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < BC; y++) {
                    pv += S[tid * BC + y] * sV[y * d + x];
                }

                float output = Output[(i * BR + tid) * d + x];
                float result = 1.0f / global_sum_new * (global_sum_prev * exp_global_max_diff * output + exp_local_max_diff * pv);
                Output[(i * BR + tid) * d + x] = result;
            }

            m[i * BR + tid] = global_max_new;
            l[i * BR + tid] = global_sum_new;
        }
        __syncthreads(); 
    }
}

void flashAttention(float* Output, float* m, float* l, const float* Q, const float* K, const float* V, 
                    int N, int d) {

    // number of tile in row and column
    const int Tr = (N + BR - 1) / BR;
    const int Tc = (N + BC -1) / BC;
    const float softmaxScale = 1.0f / sqrtf(d);
    const int sramSize = (BR * d + BC * d * 2 + BR * BC) * sizeof(float);  // Q, K, V, S

    dim3 dimGrid(1);
    dim3 dimBlock(BR);
    flashAttention_kernel<<<dimGrid, dimBlock, sramSize>>>(Output, m, l, Q, K, V, N, d, Tr, Tc, softmaxScale);
}

}
