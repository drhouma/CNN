#ifndef CUSTOM_CNN_NETWORK_H
#define CUSTOM_CNN_NETWORK_H
#include <string>
#include <vector>

#include "matrix/s21_matrix_oop.h"

/// input = image 28x28

enum Layers { CONVOLUTION = 1, KERNEL = 2 };

struct topology_member {
  size_t count;
  Layers what_layer;
};

class CNN_Network {
 public:
  auto Conv(size_t cernel_layer) -> void;
  auto MaxPooling(size_t dimension) -> void;

  auto SetTopology(const std::vector<topology_member>& topology) -> void;

  auto FeedInput(const S21Matrix& input) -> void;

 private:
  auto EvalCard(S21Matrix& input, S21Matrix& filter) -> S21Matrix;

  std::vector<topology_member> m_topology;

  std::vector<std::vector<S21Matrix>> m_kernels;
  std::vector<S21Matrix> m_conv_layers;

  std::vector<S21Matrix> m_current_input;
};

#endif