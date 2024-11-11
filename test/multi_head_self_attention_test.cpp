#include <iostream>
#include <vector>
#include "attention.h"

int main() {

    int numberOfTokens = 6;
    int numberOfHeads = 2;
    int dimOfTokens = 3;
    int dimOfQueriesAndKeys = 5;
    int dimOfValues = 4;
    int dimOfOutput = 3;
    
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(numberOfTokens, dimOfTokens);
    std::vector<Eigen::MatrixXf> Wqs(numberOfHeads, Eigen::MatrixXf::Random(dimOfTokens, dimOfQueriesAndKeys));
    std::vector<Eigen::MatrixXf> Wks(numberOfHeads, Eigen::MatrixXf::Random(dimOfTokens, dimOfQueriesAndKeys));
    std::vector<Eigen::MatrixXf> Wvs(numberOfHeads, Eigen::MatrixXf::Random(dimOfTokens, dimOfValues));
    Eigen::MatrixXf Wo = Eigen::MatrixXf::Random(numberOfHeads * dimOfValues, dimOfOutput);
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(numberOfTokens, dimOfOutput);

    ATTENTION::computeMultiHeadAttention(input, Wqs, Wks, Wvs, Wo, output);

    std::cout << "input: " << std::endl;
    std::cout << input << std::endl;
    std::cout << "Wqs: " << std::endl;
    for (int i = 0; i < numberOfHeads; i++) {
        std::cout << Wqs[i] << std::endl;
    }
    std::cout << "Wks: " << std::endl;
    for (int i = 0; i < numberOfHeads; i++) {
        std::cout << Wks[i] << std::endl;
    }
    std::cout << "Wvs: " << std::endl;
    for (int i = 0; i < numberOfHeads; i++) {
        std::cout << Wvs[i] << std::endl;
    }
    std::cout << "Wo: " << std::endl;
    std::cout << Wo << std::endl;
    std::cout << "output: " << std::endl;
    std::cout << output << std::endl;

    return 0;
}