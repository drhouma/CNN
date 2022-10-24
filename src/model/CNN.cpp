#include "CNN.h"

#include <omp.h>

CNN::CNN() {
  std::random_device randDevice;
  std::mt19937 tempGenerator(randDevice());
  m_generator = tempGenerator;
}

/*--- Скармливает вектор значений нейросети ---*/
auto CNN::FeedInput(S21Matrix& input) -> void {
  m_layers_norm[0].input(input);
}

auto CNN::randomWeight() -> double {
  double rand = ((int)m_generator() % 10000) * 0.0001;
  return rand;
}

auto CNN::InitWeightMatrix(S21Matrix& matrix) -> void {
  for (size_t i = 0; i < matrix.row(); i++) {
    for (size_t j = 0; j < matrix.col(); j++) {
      matrix(i, j) = randomWeight();
    }
  }
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
    m_layers_norm.push_back(Layer(1, layerSize, layerSize));
  } else if (type == CONVOLUTION) {
    if (m_layers_norm.empty()) {
      throw std::invalid_argument("daun");
    }
    size_t size = m_layers_norm[prevLayer].size() * layerSize;
    size_t row = m_layers_norm[prevLayer].row() - 2;
    size_t col = m_layers_norm[prevLayer].col() - 2;
    m_layers_norm.emplace_back(size, row, col);
    m_kernels.emplace_back();

    for (int i = 0; i < layerSize; i++) {
      m_kernels.back().push_back(S21Matrix(3, 3));
      InitWeightMatrix(m_kernels.back().back());
    }

  } else if (type == MAX_POOLING) {
    if (m_layers_norm.empty()) {
      throw std::invalid_argument("daun");
    }
    int size = m_layers_norm[prevLayer].size();
    int row = m_layers_norm[prevLayer].row() / 2;
    int col = m_layers_norm[prevLayer].col() / 2;
    m_layers_norm.emplace_back(size, row, col);

  } else {  // OUTPUT
    m_dense = new s21::MatrixNetwork();
    size_t size = m_layers_norm[prevLayer].size();
    size_t row = m_layers_norm[prevLayer].row();
    size_t col = m_layers_norm[prevLayer].col();
    m_dense->SetLayers({size * row * col, m_h_layers_size,
                        m_output_layer_size});
  }
  prevLayer++;
}

/** @brief Вычисляет новое значение матрицы input относительно фильтра
 * @param input Матрица с входными данными
 * @param filter Матрица весов
 */
auto EvalCard(S21Matrix& input, S21Matrix& filter, S21Matrix &output) -> S21Matrix {
  S21Matrix card(input.row() - 2, input.col() - 2);
  // #pragma omp parallel for
  for (int i = 0; i < card.row(); i++) {
    for (int j = 0; j < card.col(); j++) {
      double val = 0;
      for (int k1 = 0; k1 < filter.row(); k1++) {
        for (int k2 = 0; k2 < filter.col(); k2++) {
          output[i][j] += input[i + k1][j + k2] * filter[k1][k2];
        }
      }
    }
  }
  return card;
}

/**
 * @brief вычисляет новый сверточный слой
 * @param kernel_layer слой фильтров, с которым будут производиться вычисления
 */
auto CNN::Conv(size_t kernel_layer) -> void {
  // std::vector<S21Matrix>* cur_output = m_current_input + 1;
  // int i = 0;
  // for (auto& elem : *m_current_input) {
  //   for (auto& kernel : m_kernels[kernel_layer]) {
  //     // S21Matrix card = EvalCard(elem, kernel);
  //     // (*cur_output)[i].SetVals(card);
  //     i++;
  //   }
  // }
  // m_current_input = m_current_input + 1;
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
auto CNN::MaxPooling(size_t dim) -> void {
  m_layers_norm[m_current_layer].MaxPooling(m_layers_norm[m_current_layer + 1], dim);
}

/**
 * @brief Вычисляет результат через многослойный перцептрон
 */
auto CNN::Evaluate() -> std::vector<double> {
  std::vector<double> input{m_layers_norm[m_current_layer].toVector()};
  m_dense->FeedInitValues(input);
  m_dense->FeedForward();
  return m_output = m_dense->GetResultVector();
}

auto CNN::Predict() -> size_t {
  int conv_num = 0, result = -1;
  int curLayer{-1};  /*что бы пройти слой input*/
  for (auto elem : m_topology) {
    if (elem == CONVOLUTION) {
      // Conv(conv_num);
      m_layers_norm[curLayer].Conv(m_kernels[conv_num], m_layers_norm[curLayer + 1]);
      conv_num++;
    } else if (elem == MAX_POOLING) {
      m_layers_norm[curLayer].MaxPooling(m_layers_norm[curLayer + 1], 2);
    } else if (elem == OUTPUT) {
      std::vector<double> res = Evaluate();
      int what = res.size();
      result = distance(res.begin(), max_element(res.begin(), res.end()));
    }
    curLayer++;
  }
  return result;
}

// auto Deconv(S21Matrix& input, S21Matrix& kernel) -> S21Matrix {
//   S21Matrix tmp(input.row() + 2, input.col() + 2);

//   for (int i = 0; i < tmp.row(); i++) {
//     for (int j = 0; j < tmp.col(); j++) {
//       for (int ik = 0; ik < kernel.row(); ik++) {
//         for (int jk = 0; jk < kernel.col(); jk++) {
//           if (i + ik - 2 > -1 && i + ik - 2 < input.row() && j + jk - 2 > -1 &&
//               j + jk - 2 < input.col()) {
//             tmp[i][j] += input[i + ik - 2][j + jk - 2] * kernel[ik][jk];
//           }
//         }
//       }
//     }
//   }
//   return tmp;
// }
// auto ConvertVectorToMatrix(std::vector<double>::iterator& iter, S21Matrix& dst)
//     -> void {
//   for (int i = 0; i < dst.row(); i++) {
//     for (int j = 0; j < dst.col(); j++) {
//       dst[i][j] = *iter;
//       ++iter;
//     }
//   }
// }

// auto GetMaxElementIndex(S21Matrix& input, int& row, int& col) -> void {
//   int rowMax = row, colMax = col;
//   double max = input[row][col];
//   for (int i = 0; i < 2; i++) {
//     for (int j = 0; j < 2; j++) {
//       if (input[row + i][col + j] > max) {
//         rowMax = row + i;
//         colMax = col + j;
//       }
//     }
//   }
//   row = rowMax;
//   col = colMax;
// }
// auto CNN::UpdateGradiend(std::vector<S21Matrix>& localGradient, int curK,
//                          int curL, LayerType type) -> void {
//   std::vector<S21Matrix> newGrads;
//   if (type == CONVOLUTION) {
//     if (curK > 0) {
//       for (auto& elem : localGradient) {
//         S21Matrix tmp(elem.row() + 2, elem.col() + 2);
//         for (auto& kernel : m_kernels[curK]) {
//           tmp.SetVals(tmp + Deconv(elem, kernel));
//         }
//         newGrads.push_back(tmp);
//       }
//     }
//   } else if (type == MAX_POOLING) {
//     std::vector<S21Matrix> newGrads;
//     for (int i = 0; i < localGradient.size(); i++) {
//       S21Matrix* curMatrix = &localGradient[i];
//       S21Matrix tmp(curMatrix->row() * 2, curMatrix->col() * 2);
//       for (int row = 0; row < curMatrix->row(); row++) {
//         for (int col = 0; col < curMatrix->col(); col++) {
//           int newRow = row * 2, newCol = col * 2;
//           GetMaxElementIndex(*curMatrix, newRow, newCol);
//           tmp[newRow][newCol] = (*curMatrix)[row][col];
//         }
//       }
//       newGrads.push_back(tmp);
//     }
//     localGradient = newGrads;
//   }
// }

// auto CNN::UpdateWeights(std::vector<S21Matrix>& localGrads, int curK, int curL)
//     -> void {
//   for (auto& elem : m_layers[curL]) {
//     for (int p = 0; p < m_kernels[curK].size(); p++) {
//       auto& kernel = m_kernels[curK][p];
//       for (int i = 0; i < elem.row() - 2; i++) {
//         for (int j = 0; j < elem.col() - 2; j++) {
//           for (int ik = 0; ik < kernel.row(); ik++) {
//             for (int jk = 0; jk < kernel.col(); jk++) {
//               kernel[ik][jk] -=
//                   localGrads[p][i][j] * learningRate * elem[i + ik][j + jk];
//             }
//           }
//         }
//       }
//     }
//   }
//   m_current_input = m_current_input + 1;
// }

// auto CNN::BackProp(std::vector<double>& expectedVals) -> void {
//   std::vector<S21Matrix> localGradient;

//   int curKernel = m_kernels.size() - 1;
//   std::vector<double> localGrads = m_dense->CNNBackPropagation(expectedVals);
//   auto it = localGrads.begin();
//   for (int i = 0; i < m_layers.back().size(); i++) {
//     int row = m_layers.back()[0].row();
//     int col = m_layers.back()[0].col();
//     S21Matrix tmp(row, col);
//     ConvertVectorToMatrix(it, tmp);
//     localGradient.push_back(tmp);
//   }
//   /// если всего два слоя - input и output, то инструкций выше хватит для
//   /// backProp
//   for (int layer = m_topology.size() - 3; layer >= 0; layer--) {
//     UpdateGradiend(localGradient, curKernel, layer + 1, m_topology[layer]);

//     if (m_topology[layer + 1] == CONVOLUTION) {
//       UpdateWeights(localGradient, curKernel, layer);
//       curKernel--;
//     }
//   }
// }

Layer::Layer(size_t size, size_t row, size_t col) {
  m_layers.resize(size);
  for (int i = 0; i < size; i++) {
    m_layers[i].resize(row, col);
  }
}

auto Layer::MaxPooling(Layer &output, size_t dimension) -> void {
  size_t newRows = output.row();
  size_t newCols = output.col();
  int layer = 0;
  for (int layer = 0; layer < m_layers.size(); layer++) {
    for (int i = 0; i < newRows; i++) {
      for (int j = 0; j < newCols; j++) {
        output.m_layers[layer][i][j] = FindMax(m_layers[layer], i * dimension, j * dimension, dimension);
      }
    }
  }
}

auto Layer::Conv(std::vector<S21Matrix>& kernels, Layer& output) -> void {
  int i = 0;
  for (auto& elem : m_layers) {
    for (auto& kernel : kernels) {
      EvalCard(elem, kernel, output.m_layers[i]);
      i++;
    }
  }
}

auto Layer::size() -> size_t {return m_layers.size();}

auto Layer::row() -> size_t {return m_layers[0].row();}

auto Layer::col() -> size_t {return m_layers[0].col();}

auto Layer::toVector() -> std::vector<double> {
  std::vector<double> ret(m_layers.size() * m_layers[0].row() * m_layers[0].col());
  int vector_index = 0;
  for (S21Matrix& elem : m_layers) {
    for (int i = 0; i < elem.row(); i++) {
      for (int j = 0; j < elem.col(); j++) {
        ret[vector_index] = elem[i][j];
        vector_index++;
      }
    }
  }
  return ret;
}

auto Layer::input(S21Matrix &input) -> void {
  m_layers[0].SetVals(input);
}