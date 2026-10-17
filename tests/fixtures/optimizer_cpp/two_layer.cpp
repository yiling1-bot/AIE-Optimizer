#include "dimensions.hpp"

static I8 limit_value(int value) {
  return (I8)(value > 127 ? 127 : value < -128 ? -128 : value);
}

static void first_stage(const I8 (&samples)[M][K], const I8 (&weights)[K][H],
                        I8 (& __restrict hidden)[M][H]) {
  for (int column = 0; column < H; ++column)
    for (int row = 0; row < M; ++row) {
      int accumulator = 0;
      for (int reduction = 0; reduction < K; ++reduction)
        accumulator += (int)samples[row][reduction] * (int)weights[reduction][column];
      hidden[row][column] = limit_value(accumulator);
    }
}

void network(const I8 (&second)[H][N], const I8 (&input)[M][K],
             I8 (& __restrict output)[M][N], const I8 (&first)[K][H]) {
  I8 temporary[M][H];
  first_stage(input, first, temporary);
  for (int row = 0; row < M; ++row)
    for (int column = 0; column < N; ++column) {
      int total = 0;
      for (int reduction = 0; reduction < H; ++reduction)
        total = total + temporary[row][reduction] * second[reduction][column];
      output[row][column] = limit_value(total);
    }
}
