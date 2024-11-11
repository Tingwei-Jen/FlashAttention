#include "attention.h"

namespace ATTENTION {

// The number of queries, keys, and values is the same as the number of tokens (sequence length).
// input: [number of tokens * dimension of tokens]
// Wqs: [number of heads, dimension of tokens, dimension of queries and keys]
// Wks: [number of heads, dimension of tokens, dimension of queries and keys]
// Wvs: [number of heads, dimension of tokens, dimension of values]
// Wo: [number of heads * dimension of values, dimension of output]
// output: [number of tokens, dimension of output]
void computeMultiHeadAttention(const Eigen::MatrixXf& input,
                               const std::vector<Eigen::MatrixXf>& Wqs,
                               const std::vector<Eigen::MatrixXf>& Wks,
                               const std::vector<Eigen::MatrixXf>& Wvs,
                               const Eigen::MatrixXf& Wo,
                               Eigen::MatrixXf& output) {
    
    int numberOfTokens = input.rows();
    int numberOfHeads = Wqs.size();
    int dimOfValues = Wvs[0].cols();
    Eigen::MatrixXf concat = Eigen::MatrixXf::Random(numberOfTokens, numberOfHeads * dimOfValues);

    for (int i = 0; i < numberOfHeads; i++) {

        Eigen::MatrixXf query = input * Wqs[i];  // [number of tokens, dimension of queries and keys]
        Eigen::MatrixXf key = input * Wks[i];    // [number of tokens, dimension of queries and keys]
        Eigen::MatrixXf value = input * Wvs[i];  // [number of tokens, dimension of values]

        int dimOfQueriesAndKeys = query.cols();
        
        // calculate scores
        Eigen::MatrixXf attentionScores = query * key.transpose() / std::sqrt(dimOfQueriesAndKeys);  // [number of tokens, number of tokens]

        // calculate softmax each row
        Eigen::MatrixXf expScores = attentionScores.array().exp();
        for (int i = 0; i < expScores.rows(); i++) {
            expScores.row(i) /= expScores.row(i).sum();
        }

        Eigen::MatrixXf head = expScores * value;  // [number of tokens, dimension of values]
        concat.block(0, i * dimOfValues, numberOfTokens, dimOfValues) = head;
    }

    output = concat * Wo;  // [number of tokens, dimension of output]
}

}