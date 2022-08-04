#include <omp.h>

#include "CNN_network.h"

auto CNN_Network::SetTopology(const std::vector<topology_member>& topology)
    -> void {
  m_topology = topology;
  for (auto elem : topology) {
    if (elem.what_layer == KERNEL) {
      m_kernels.push_back(std::vector<S21Matrix>(elem.count));
    }
  }
}

auto CNN_Network::FeedInput(const S21Matrix& input) -> void {
  m_current_input.clear();
  m_current_input.push_back(input);
}

auto CNN_Network::EvalCard(S21Matrix& input, S21Matrix& filter) -> S21Matrix {
  S21Matrix card(input);
  // #pragma omp parallel for
  for (int i = 0; i < card.GetRow(); i++) {
    for (int j = 0; j < card.GetColumn(); j++) {
      double val = 0;
      for (int k1 = 0; k1 < filter.GetRow(); k1++) {
        for (int k2 = 0; k2 < filter.GetColumn(); k2++) {
          if (i + k1 - 1 >= 0 && j + k2 - 1 >= 0) {
            val += input[i + k1 - 1][j + k2 - 1] * filter[k1][k2];
          }
        }
      }
    }
  }
}

auto CNN_Network::Conv(size_t kernel_layer) -> void {
  std::vector<S21Matrix> newLayer;
  for (auto& elem : m_current_input) {
    for (auto& kernel : m_kernels[kernel_layer]) {
      S21Matrix card = EvalCard(elem, kernel);
      newLayer.push_back(card);
    }
  }
  m_current_input = newLayer;
}

auto FindMax(S21Matrix& input, size_t i, size_t j, size_t dimension) -> double {
  double res = input[i][j];
  for (int i1 = 0; i < dimension; i++) {
    for (int j1 = 0; j < dimension; j++) {
      if (res < input[i + i1][j + j1]) res = input[i + i1][j + j1];
    }
  }
  return res;
}

auto CNN_Network::MaxPooling(size_t dimension) -> void {
  std::vector<S21Matrix> newLayer;
  int newRows = m_current_input[0].GetRow() / dimension;
  int newCols = m_current_input[0].GetColumn() / dimension;
  for (auto& elem : m_current_input) {
    S21Matrix temp(newRows, newCols);
    for (int i = 0; i < elem.GetRow(); i += dimension) {
      for (int j = 0; j < elem.GetColumn(); j += dimension) {
        temp[i][j] = FindMax(elem, i, j, dimension);
      }
    }
    newLayer.push_back(temp);
  }
  m_current_input = newLayer;
}
