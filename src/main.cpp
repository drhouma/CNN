#include "model/CNN.h"

int main() {
  CNN net;
  net.AddLayer(INPUT, 28);
  net.AddLayer(CONVOLUTION, 10);
  net.AddLayer(CONVOLUTION, 5);
  net.AddLayer(MAX_POOLING, 5);
  net.AddLayer(CONVOLUTION, 3);
  net.AddLayer(OUTPUT, 10);

  S21Matrix random(28, 28);
  random.FillRandom(0, 1);
  net.FeedInput(random);
  std::vector<double> va(10);
  // net.BackProp(va);
  std::cout << net.Predict();
}