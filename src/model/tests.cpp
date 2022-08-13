#include <gtest/gtest.h>

#include "CNN.h"

TEST(CNN_tests, Add_layer) {
  CNN net;
  net.AddLayer(INPUT, 28);
  net.AddLayer(CONVOLUTION, 10);
  net.AddLayer(MAX_POOLING, 28);
  net.AddLayer(CONVOLUTION, 5);
  net.AddLayer(MAX_POOLING, 28);
  net.AddLayer(OUTPUT, 28);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
