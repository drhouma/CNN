#include "CNN.h"

#include <omp.h>

/*--- Скармливает вектор значений нейросети ---*/
auto CNN::FeedInput(const S21Matrix& input) -> void {
  m_layers[0][0].SetVals(input);
  m_current_input = &m_layers[0];
}

/**
 * @brief
 *
 * @param type
 * @param layerSize для типа [Convolution] - сколько фильтров применяется к слою
 * для типа [Input] - размерность матрицы с входными данными
 * для типа [Output] и [Max_Pooling] не играет роли
 */
auto CNN::AddLayer(LayerType type, size_t layerSize) -> void {
  m_topology.push_back(type);

  if (type == INPUT) {
    m_layers.push_back(std::vector<S21Matrix>());
    m_layers.back().push_back(S21Matrix(layerSize, layerSize));
  } else if (type == CONVOLUTION) {
    if (m_layers.empty()) {
      throw std::invalid_argument("daun");
    }
    m_layers.push_back(std::vector<S21Matrix>());
    for (int i = 0; i < m_current_input->size() * layerSize; i++) {
      int row = m_current_input->back().row();
      int col = m_current_input->back().col();
      m_layers.back().push_back(S21Matrix(row, col));
    }
    m_kernels.push_back(std::vector<S21Matrix>());
    for (int i = 0; i < layerSize; i++) {
      m_kernels.back().push_back(S21Matrix(3, 3));
    }
  } else if (type == MAX_POOLING) {
    if (m_layers.empty()) {
      throw std::invalid_argument("daun");
    }
    m_layers.push_back(std::vector<S21Matrix>());
    for (int i = 0; i < m_current_input->size() * layerSize; i++) {
      int row = m_current_input->back().row() / 2;
      int col = m_current_input->back().col() / 2;
      m_layers.back().push_back(S21Matrix(row, col));
    }
  } else {  // OUTPUT
    m_dense = new s21::MatrixNetwork();
    int row = m_current_input->back().row() / 2;
    int col = m_current_input->back().col() / 2;
    m_dense->SetLayers({m_current_input->size() * row * col, 140, 10});
  }

  m_current_input = &m_layers.back();
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
  std::vector<S21Matrix>* cur_output = m_current_input + 1;
  int i = 0;
  for (auto& elem : *m_current_input) {
    for (auto& kernel : m_kernels[kernel_layer]) {
      S21Matrix card = EvalCard(elem, kernel);
      (*cur_output)[i].SetVals(card);
      i++;
    }
  }
  m_current_input = m_current_input + 1;
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
  std::vector<S21Matrix>* output = m_current_input + 1;
  int newRows = (*m_current_input)[0].row() / dimension;
  int newCols = (*m_current_input)[0].col() / dimension;
  int counter = 0;
  // #pragma omp parallel for
  for (auto& elem : (*m_current_input)) {
    S21Matrix temp(newRows, newCols);
    for (int i = 0; i < elem.row(); i += dimension) {
      for (int j = 0; j < elem.col(); j += dimension) {
        temp[i][j] = FindMax(elem, i, j, dimension);
      }
    }

    (*output)[counter].SetVals(temp);
  }
  m_current_input = output;
}

/**
 * @brief Вычисляет результат через многослойный перцептрон
 */
auto CNN::Evaluate() -> std::vector<double> {
  size_t input_size = (*m_current_input).size() * (*m_current_input)[0].row() *
                      (*m_current_input)[0].row();
  std::vector<double> input(input_size);

  m_dense = new s21::MatrixNetwork();
  m_dense->SetLayers({input_size, m_h_layers_size, m_output_layer_size});
  int vector_index = 0;
  for (S21Matrix& elem : (*m_current_input)) {
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