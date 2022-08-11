#ifndef CUSTOM_CNN_NETWORK_H
#define CUSTOM_CNN_NETWORK_H
#include <initializer_list>
#include <string>
#include <vector>

#include "MLP/MLP_network.h"
#include "matrix/s21_matrix_oop.h"

/// input = image 28x28

enum LayerType { INPUT, CONVOLUTION, MAX_POOLING, OUTPUT };

class CNN {
 public:
  auto Conv(size_t cernel_layer) -> void;
  auto MaxPooling(size_t dimension) -> void;
  auto Evaluate() -> std::vector<double>;

  auto FeedInput(const S21Matrix& input) -> void;

  auto InitKernels(const std::initializer_list<size_t>& topology);
  auto InitKernels(const std::vector<size_t>& topology);

  auto AddLayer(LayerType type, size_t layerSize) -> void;

 private:
  auto EvalCard(S21Matrix& input, S21Matrix& filter) -> S21Matrix;

  std::vector<LayerType> m_topology;
  std::vector<std::vector<S21Matrix>> m_kernels;

  std::vector<std::vector<S21Matrix>> m_layers;

  std::vector<S21Matrix> m_current_input;

  std::vector<double> m_output;
  static constexpr int m_h_layers_size = 100;
  static constexpr int m_output_layer_size = 100;
  s21::NetworkInterface* m_dense;

  // std::vector<S21Matrix> m_conv_layers;
};

#endif