#ifndef CNN_NETWORK_INTERFACE_H
#define CNN_NETWORK_INTERFACE_H
#include <algorithm>

#include "CNN_network.h"

enum LayerType { CONVOLUTION, MAX_POOLING, OUTPUT };

class CNN_NetworkInterface {
 public:
  auto InitNetwork(const std::initializer_list<LayerType>&) -> bool;

  auto Predict() -> size_t;

 private:
  std::vector<LayerType> m_topology;
  CNN_Network m_net;
};
#endif