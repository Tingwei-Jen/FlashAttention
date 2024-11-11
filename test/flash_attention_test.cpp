#include <iostream>
#include <vector>
#include "attention.h"

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

void randomInit(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

int main() {

    int numberOfTokens = 4096;
    int dim = 32;

    float *hQ, *hK, *hV, *hOutput;
    float *hm, *hl;
    float *dQ, *dK, *dV, *dOutput;
    float *dm, *dl;

    // Allocate host memory
    hQ = (float *)malloc(numberOfTokens * dim * sizeof(float));
    hK = (float *)malloc(numberOfTokens * dim * sizeof(float));
    hV = (float *)malloc(numberOfTokens * dim * sizeof(float));
    hOutput = (float *)malloc(numberOfTokens * dim * sizeof(float));
    hm = (float *)malloc(numberOfTokens * sizeof(float));
    hl = (float *)malloc(numberOfTokens * sizeof(float));

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void **)&dQ, numberOfTokens * dim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&dK, numberOfTokens * dim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&dV, numberOfTokens * dim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&dOutput, numberOfTokens * dim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&dm, numberOfTokens * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&dl, numberOfTokens * sizeof(float)));

    // random initialization
    randomInit(hQ, numberOfTokens * dim);
    randomInit(hK, numberOfTokens * dim);
    randomInit(hV, numberOfTokens * dim);

    // set output and l as zero
    for (int i = 0; i < numberOfTokens * dim; i++) {
        hOutput[i] = 0;
    }

    for (int i = 0; i < numberOfTokens; i++) {
        hl[i] = 0;
    }

    // set m as minimum value
    for (int i = 0; i < numberOfTokens; i++) {
        hm[i] = -1.0e+10;
    }

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(dQ, hQ, numberOfTokens * dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dK, hK, numberOfTokens * dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dV, hV, numberOfTokens * dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dOutput, hOutput, numberOfTokens * dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dm, hm, numberOfTokens * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dl, hl, numberOfTokens * sizeof(float), cudaMemcpyHostToDevice));

    // call flashAttention
    ATTENTION::flashAttention(dOutput, dm, dl, dQ, dK, dV, numberOfTokens, dim);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(hOutput, dOutput, numberOfTokens * dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    // compare results with Eigen
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Q(hQ, numberOfTokens, dim);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> K(hK, numberOfTokens, dim);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> V(hV, numberOfTokens, dim);
    Eigen::MatrixXf attentionScores = Q * K.transpose() / std::sqrt(dim);

    // calculate softmax each row
    Eigen::MatrixXf expScores = attentionScores.array().exp();
    for (int i = 0; i < expScores.rows(); i++) {
        expScores.row(i) /= expScores.row(i).sum();
    }
    Eigen::MatrixXf Output = expScores * V;

    // compare results
    float epsilon = 1.0e-6;

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < dim; j++) {
            if (std::abs(hOutput[i * dim + j] - Output(i, j)) > epsilon) {
                std::cout << "Results do not match!" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Results match!" << std::endl;

    // free memory
    free(hQ);
    free(hK);
    free(hV);
    free(hOutput);
    free(hm);
    free(hl);
    checkCudaErrors(cudaFree(dQ));
    checkCudaErrors(cudaFree(dK));
    checkCudaErrors(cudaFree(dV));
    checkCudaErrors(cudaFree(dOutput));
    checkCudaErrors(cudaFree(dm));
    checkCudaErrors(cudaFree(dl));

    return 0;
}
