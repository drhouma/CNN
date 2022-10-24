#ifndef CUSTOM_CNN_NETWORK_H
#define CUSTOM_CNN_NETWORK_H
#include <algorithm>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "MLP/MLP_network.h"
#include "matrix/s21_matrix_oop.h"

/// input = image 28x28

enum LayerType { INPUT, CONVOLUTION, MAX_POOLING, OUTPUT };


class Layer {
public:
  explicit Layer(size_t size, size_t row, size_t col);
  auto Conv(std::vector<S21Matrix>& kernels, Layer &output) -> void;
  auto MaxPooling(Layer &output, size_t dimension) -> void;
  auto input(S21Matrix &matrix) -> void;

  auto toVector() -> std::vector<double>;

  auto size() -> size_t;

  auto row() -> size_t;
  auto col() -> size_t;

private:
  std::vector<S21Matrix> m_layers;
};

class CNN {
 public:
  CNN();

  // auto BackProp(std::vector<double>& expectedVals) -> void;
  auto Conv(size_t kernel_layer) -> void;
  auto MaxPooling(size_t dimension) -> void;
  auto Evaluate() -> std::vector<double>;

  auto FeedInput(S21Matrix& input) -> void;

  auto InitKernels(const std::initializer_list<size_t>& topology);
  auto InitKernels(const std::vector<size_t>& topology);

  auto AddLayer(LayerType type, size_t layerSize) -> void;

  auto Predict() -> size_t;



 private:
  // auto EvalCard(S21Matrix& input, S21Matrix& filter) -> S21Matrix;

  // auto UpdateGradiend(std::vector<S21Matrix>& localGradient, int curK, int curL,
  //                     LayerType type) -> void;

  // auto UpdateWeights(std::vector<S21Matrix>& localGrads, int curK, int curL)
  //     -> void;

  auto InitWeightMatrix(S21Matrix& matrix) -> void;
  auto randomWeight() -> double;

  std::vector<LayerType> m_topology;
  std::vector<std::vector<S21Matrix>> m_kernels;
  // std::vector<std::vector<S21Matrix>> m_layers;

  std::vector<S21Matrix>* m_current_input;

  std::vector<double> m_output;

  static constexpr int m_h_layers_size = 100;
  static constexpr int m_output_layer_size = 10;


  std::vector<Layer> m_layers_norm;

  s21::NetworkInterface* m_dense;
  std::mt19937 m_generator;

  /// для итерации по слоям в addLayer
  int prevLayer = -1;

  size_t m_current_layer{0};

  double learningRate = 0.08;
};

#endif