#ifndef ATTENTION_H
#define ATTENTION_H
#include <Eigen/Dense>
#include <vector>
#include <cuda_runtime.h>

namespace ATTENTION {

// // query: [numberOfQueries, dimOfQueriesAndKeys]
// // key: [numberOfKeysAndValues, dimOfQueriesAndKeys]
// // value: [numberOfKeysAndValues, dimOfValues]
// // output: [numberOfQueries, dimOfValues]
// void computeScaledDotProductAttention(float* output, const float* query, const float* key, const float* value, 
//                                       const int& numberOfQueries, const int& numberOfKeysAndValues,
//                                       const int& dimOfQueriesAndKeys, const int& dimOfValues);

// // The number of queries, keys, and values is the same as the number of tokens (sequence length).
// // input: [number of tokens * dimension of tokens]
// // Wq: [dimension of tokens, dimension of queries and keys]
// // Wk: [dimension of tokens, dimension of queries and keys]
// // Wv: [dimension of tokens, dimension of values]
// // output: [number of tokens * dimension of values]
// void computeSelfAttention(const Eigen::MatrixXf& input, 
//                           const Eigen::MatrixXf& Wq, 
//                           const Eigen::MatrixXf& Wk, 
//                           const Eigen::MatrixXf& Wv, 
//                           Eigen::MatrixXf& output);

// // The number of queries, keys, and values is the same as the number of tokens (sequence length).
// // input: [number of tokens * dimension of tokens]
// // Wqs: [number of heads, dimension of tokens, dimension of queries and keys]
// // Wks: [number of heads, dimension of tokens, dimension of queries and keys]
// // Wvs: [number of heads, dimension of tokens, dimension of values]
// // Wo: [number of heads * dimension of values, dimension of output]
// // output: [number of tokens, dimension of output]
// void computeMultiHeadAttention(const Eigen::MatrixXf& input,
//                                const std::vector<Eigen::MatrixXf>& Wqs,
//                                const std::vector<Eigen::MatrixXf>& Wks,
//                                const std::vector<Eigen::MatrixXf>& Wvs,
//                                const Eigen::MatrixXf& Wo,
//                                Eigen::MatrixXf& output);

void flashAttention(float* Output, float* m, float* l, const float* Q, const float* K, const float* V, 
                    int N, int d);

}

#endif // ATTENTION_H
