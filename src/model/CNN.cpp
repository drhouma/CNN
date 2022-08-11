#include <omp.h>

#include "CNN_network.h"

/*--- Скармливает вектор значений нейросети ---*/
auto CNN::FeedInput(const S21Matrix& input) -> void {
  m_current_input.clear();
  m_current_input.push_back(input);
}

auto CNN::AddLayer(LayerType type, size_t layerSize) -> void {
  m_topology.push_back(type);

  if (type == INPUT) {
    m_layers.push_back(std::vector<S21Matrix>());
    m_layers.back().push_back(S21Matrix(layerSize, layerSize));
  } else if (type == CONVOLUTION) {
    m_layers.push_back(std::vector<S21Matrix>());
    // for ()
  }
}

/** @brief Вычисляет новое значение матрицы input относительно фильтра
 * @param input Матрица с входными данными
 * @param filter Матрица весов
 */
auto CNN::EvalCard(S21Matrix& input, S21Matrix& filter) -> S21Matrix {
  S21Matrix card(input);
  // #pragma omp parallel for
  for (int i = 0; i < card.row(); i++) {
    for (int j = 0; j < card.col(); j++) {
      double val = 0;
      for (int k1 = 0; k1 < filter.row(); k1++) {
        for (int k2 = 0; k2 < filter.col(); k2++) {
          if (i + k1 - 1 >= 0 && j + k2 - 1 >= 0) {
            val += input[i + k1 - 1][j + k2 - 1] * filter[k1][k2];
          }
        }
      }
    }
  }
}

/**
 * @brief вычисляет новый сверточный слой
 * @param kernel_layer слой фильтров, с которым будут производиться вычисления
 */
auto CNN::Conv(size_t kernel_layer) -> void {
  std::vector<S21Matrix> newLayer;
  for (auto& elem : m_current_input) {
    for (auto& kernel : m_kernels[kernel_layer]) {
      S21Matrix card = EvalCard(elem, kernel);
      newLayer.push_back(card);
    }
  }
  m_current_input = newLayer;
}

/**
 * @brief Находит максимальное значение рядом с input[i][j] на расстоянии
 * dimension от него
 *
 * @param input матрица, в которой ищется максимум
 * @param i строка матрицы
 * @param j столбец матрицы
 * @param dimension радиус поиска (диагональ не считается)
 * @return double max
 */
auto FindMax(S21Matrix& input, size_t i, size_t j, size_t dimension) -> double {
  double res = input[i][j];
  // #pragma omp parallel for
  for (int i1 = 0; i < dimension; i++) {
    for (int j1 = 0; j < dimension; j++) {
      if (res < input[i + i1][j + j1]) res = input[i + i1][j + j1];
    }
  }
  return res;
}
/**
 * @brief Масштабирует матрицы в векторе m_current_input
 * @param dimension - во сколько раз уменьшаются матрицы
 */
auto CNN::MaxPooling(size_t dimension) -> void {
  std::vector<S21Matrix> newLayer;
  int newRows = m_current_input[0].row() / dimension;
  int newCols = m_current_input[0].col() / dimension;
  // #pragma omp parallel for
  for (auto& elem : m_current_input) {
    S21Matrix temp(newRows, newCols);
    for (int i = 0; i < elem.row(); i += dimension) {
      for (int j = 0; j < elem.col(); j += dimension) {
        temp[i][j] = FindMax(elem, i, j, dimension);
      }
    }
    newLayer.push_back(temp);
  }
  m_current_input = newLayer;
}

/**
 * @brief Вычисляет результат через многослойный перцептрон
 */
auto CNN::Evaluate() -> std::vector<double> {
  size_t input_size = m_current_input.size() * m_current_input[0].row() *
                      m_current_input[0].row();
  std::vector<double> input(input_size);

  m_dense = new s21::MatrixNetwork();
  m_dense->SetLayers({input_size, m_h_layers_size, m_output_layer_size});
  int vector_index = 0;
  for (S21Matrix& elem : m_current_input) {
    for (int i = 0; i < elem.row(); i++) {
      for (int j = 0; j < elem.col(); j++) {
        input[vector_index] = elem[i][j];
        vector_index++;
      }
    }
  }
  m_dense->FeedInitValues(input);
  m_dense->FeedForward();
  return m_output = m_dense->GetResultVector();
}