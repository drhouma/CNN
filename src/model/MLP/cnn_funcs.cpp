

#include "MLP_network.h"

namespace s21 {

/// не будет использоваться, но нужно что бы компилилось
auto GraphNetwork::CNNBackPropagation(std::vector<double> &ExpectedValues)
    -> std::vector<double> {}

auto MatrixNetwork::CNNBackPropagation(std::vector<double> &ExpectedValues)
    -> std::vector<double> {
  std::vector<double> localGrads;
  for (int i = m_weights.size() - 1; i >= 0; i--) {
    GetLocalGrads(localGrads, ExpectedValues, i + 1);

    for (size_t k = 0; k < m_weights[i].col(); k++) {
      for (size_t j = 0; j < m_weights[i].row(); j++) {
        m_weights[i][j][k] += step * localGrads[j] * m_neurons[i][k][0];
      }
    }
  }
  GetLocalGrads(localGrads, ExpectedValues, 0);
  return localGrads;
}
}  // namespace s21
