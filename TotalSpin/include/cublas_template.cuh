#ifndef CUBLAS_TEMPLATE_CUH
#define CUBLAS_TEMPLATE_CUH

#include <cublas_v2.h>
#include <thrust/device_vector.h>

namespace cublas // using column-major matrix
{
// y = alpha*(A*X) + beta*y
inline void gemv(const cublasHandle_t & handle, const cublasOperation_t & trans, const int & m, const int & n, const double & alpha,
  const double * A, const double * x, const double & beta, double * y)
{
  cublasDgemv(handle, trans, m, n, &alpha, A, m, x, 1, &beta, y, 1);
}

inline void gemv(const cublasHandle_t & handle, const cublasOperation_t & trans, const int & m, const int & n, const float & alpha,
  const float * A, const float * x, const float & beta, float * y)
{
  cublasSgemv(handle, trans, m, n, &alpha, A, m, x, 1, &beta, y, 1);
}

} // namespace cublas

#endif
