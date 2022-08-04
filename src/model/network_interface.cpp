#include "network_interface.h"

auto CNN_NetworkInterface::InitNetwork(
    const std::initializer_list<LayerType>& vals) -> bool {
  std::vector<size_t> kernel_topology;
  for (auto elem : vals) {
    m_topology.push_back(elem);
  }
}

auto CNN_NetworkInterface::Predict() -> size_t {
  int conv_num = 0;
  for (auto elem : m_topology) {
    if (elem == CONVOLUTION) {
      m_net.Conv(conv_num);
      conv_num++;
    } else if (elem == MAX_POOLING) {
      m_net.MaxPooling(2);
    } else if (elem == OUTPUT) {
      std::vector<double> res = m_net.Evaluate();
      return distance(res.begin(), max_element(res.begin(), res.end()));
    }
  }
}