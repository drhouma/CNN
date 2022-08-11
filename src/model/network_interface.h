#ifndef CNN_NETWORK_INTERFACE_H
#define CNN_NETWORK_INTERFACE_H
#include <algorithm>

#include "CNN_network.h"

class CNN_NetworkInterface {
 public:
  auto InitNetwork(const std::initializer_list<LayerType> &) -> bool;

  auto Predict() -> size_t;

  auto Train(const S21Matrix &input, const std::vector<double> expectedVals)
      -> void;

 private:
  std::vector<LayerType> m_topology;
  CNN m_net;
};
#endif