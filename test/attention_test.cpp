#include <iostream>
#include <vector>
#include "attention.h"

void randomInit(std::vector<float>& arr) {
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

int main() {

    int numberOfQueries = 1;
    int numberOfKeysAndValues = 3;
    int dimOfQueriesAndKeys = 5;
    int dimOfValues = 10;
    
    std::vector<std::vector<float>> v_query(numberOfQueries, std::vector<float>(dimOfQueriesAndKeys, 0));
    std::vector<std::vector<float>> v_key(numberOfKeysAndValues, std::vector<float>(dimOfQueriesAndKeys, 0));
    std::vector<std::vector<float>> v_value(numberOfKeysAndValues, std::vector<float>(dimOfValues, 0));
    std::vector<std::vector<float>> v_output(numberOfQueries, std::vector<float>(dimOfValues, 0));

    // random initialization
    for (int i = 0; i < numberOfQueries; i++) {
        randomInit(v_query[i]);
    }

    for (int i = 0; i < numberOfKeysAndValues; i++) {
        randomInit(v_key[i]);
        randomInit(v_value[i]);
    }

    // set output as zero
    for (int i = 0; i < numberOfQueries; i++) {
        for (int j = 0; j < dimOfValues; j++) {
            v_output[i][j] = 0;
        }
    }

    // call scaledDotProductAttention
    ATTENTION::computeScaledDotProductAttention(v_output[0].data(), v_query[0].data(), v_key[0].data(), v_value[0].data(), 
                                         numberOfQueries, numberOfKeysAndValues, dimOfQueriesAndKeys, dimOfValues);

    // print input and outputs
    std::cout << "query: " << std::endl;
    for (int i = 0; i < numberOfQueries; i++) {
        for (int j = 0; j < dimOfQueriesAndKeys; j++) {
            std::cout << v_query[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "key: " << std::endl;
    for (int i = 0; i < numberOfKeysAndValues; i++) {
        for (int j = 0; j < dimOfQueriesAndKeys; j++) {
            std::cout << v_key[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "value: " << std::endl;
    for (int i = 0; i < numberOfKeysAndValues; i++) {
        for (int j = 0; j < dimOfValues; j++) {
            std::cout << v_value[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "output: " << std::endl;
    for (int i = 0; i < numberOfQueries; i++) {
        for (int j = 0; j < dimOfValues; j++) {
            std::cout << v_output[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}