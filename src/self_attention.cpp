#include "attention.h"

namespace ATTENTION {

// The number of queries, keys, and values is the same as the number of tokens (sequence length).
// input: [number of tokens * dimension of tokens]
// Wq: [dimension of tokens, dimension of queries and keys]
// Wk: [dimension of tokens, dimension of queries and keys]
// Wv: [dimension of tokens, dimension of values]
// output: [number of tokens * dimension of values]
void computeSelfAttention(const Eigen::MatrixXf& input, 
                          const Eigen::MatrixXf& Wq, 
                          const Eigen::MatrixXf& Wk, 
                          const Eigen::MatrixXf& Wv, 
                          Eigen::MatrixXf& output) {

    Eigen::MatrixXf query = input * Wq;  // [number of tokens, dimension of queries and keys]
    Eigen::MatrixXf key = input * Wk;    // [number of tokens, dimension of queries and keys]
    Eigen::MatrixXf value = input * Wv;  // [number of tokens, dimension of values]

    int dimOfQueriesAndKeys = query.cols();

    // calculate scores
    Eigen::MatrixXf attentionScores = query * key.transpose() / std::sqrt(dimOfQueriesAndKeys);  // [number of tokens, number of tokens]

    // calculate softmax each row
    Eigen::MatrixXf expScores = attentionScores.array().exp();
    for (int i = 0; i < expScores.rows(); i++) {
        expScores.row(i) /= expScores.row(i).sum();
    }

    // output
    output = expScores * value;  // [number of tokens, dimension of values]
}

}