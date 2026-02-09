#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
#pragma omp parallel for schedule(static)
  for (int j = iRowBlkA; j < std::min(A.nbRows, iRowBlkA + szBlock); ++j)
    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
      for (int i = iColBlkB; i < std::min(B.nbCols, iColBlkB + szBlock); i++)
        C(i, j) += A(i, k) * B(k, j);
}
const int szBlock = 32;
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  prodSubBlocks(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
  return C;
}
