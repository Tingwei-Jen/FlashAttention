#include <iostream>
#include <vector>
#include "attention.h"

int main() {

    int numberOfTokens = 6;
    int dimOfTokens = 3;
    int dimOfQueriesAndKeys = 5;
    int dimOfValues = 4;

    // init input, Wq, Wk, Wv
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(numberOfTokens, dimOfTokens);
    Eigen::MatrixXf Wq = Eigen::MatrixXf::Random(dimOfTokens, dimOfQueriesAndKeys);
    Eigen::MatrixXf Wk = Eigen::MatrixXf::Random(dimOfTokens, dimOfQueriesAndKeys);
    Eigen::MatrixXf Wv = Eigen::MatrixXf::Random(dimOfTokens, dimOfValues);
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(numberOfTokens, dimOfValues);

    // call computeSelfAttention
    ATTENTION::computeSelfAttention(input, Wq, Wk, Wv, output);

    // print input, weights, and outputs
    std::cout << "input: " << std::endl;
    std::cout << input << std::endl;
    std::cout << "Wq: " << std::endl;
    std::cout << Wq << std::endl;
    std::cout << "Wk: " << std::endl;
    std::cout << Wk << std::endl;
    std::cout << "Wv: " << std::endl;
    std::cout << Wv << std::endl;
    std::cout << "output: " << std::endl;
    std::cout << output << std::endl;

    return 0;
}