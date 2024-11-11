#include "attention.h"
#include <cmath>

namespace ATTENTION {

float dotProduct(const float* a, const float* b, int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void softmax(float* expScores, const float* scores, const int& size) {
    // calculate sum of expScores
    float sumExpScores = 0;
    for (int i = 0; i < size; i++) {
        expScores[i] = std::exp(scores[i]);
        sumExpScores += expScores[i];
    }
    // calculate softmax
    for (int i = 0; i < size; i++) {
        expScores[i] /= sumExpScores;
    }
}

// query: [numberOfQueries, dimOfQueriesAndKeys]
// key: [numberOfKeysAndValues, dimOfQueriesAndKeys]
// value: [numberOfKeysAndValues, dimOfValues]
// output: [numberOfQueries, dimOfValues]
void computeScaledDotProductAttention(float* output, const float* query, const float* key, const float* value, 
                                      const int& numberOfQueries, const int& numberOfKeysAndValues,
                                      const int& dimOfQueriesAndKeys, const int& dimOfValues) {
    
    float* attentionScores = new float[numberOfKeysAndValues];
    float* attentionWeights = new float[numberOfKeysAndValues];

    for (int i = 0; i < numberOfQueries; i++) {
        // get query
        const float* queryPtr = query + i * dimOfQueriesAndKeys;

        // calculate scores
        for (int j = 0; j < numberOfKeysAndValues; j++) {
            const float* keyPtr = key + j * dimOfQueriesAndKeys;
            float score = dotProduct(queryPtr, keyPtr, dimOfQueriesAndKeys) / std::sqrt(dimOfQueriesAndKeys);
            attentionScores[j] = score;
        }

        // calculate softmax
        softmax(attentionWeights, attentionScores, numberOfKeysAndValues);

        // calculate output
        for (int j = 0; j < numberOfKeysAndValues; j++) {
            const float* valuePtr = value + j * dimOfValues;
            float weight = attentionWeights[j];

            for (int k = 0; k < dimOfValues; k++) {
                output[i * dimOfValues + k] += weight * valuePtr[k];
            }
        }
    }

    // free
    delete[] attentionScores;
    delete[] attentionWeights;
}

}