/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_lapack_api.h
 * \brief Unified interface for LAPACK calls from within mxnet.
 *  Purpose is to hide the platform specific differences.
 */
#ifndef MXNET_C_LAPACK_API_H_
#define MXNET_C_LAPACK_API_H_

// Manually maintained list of LAPACK interfaces that can be used
// within MXNET. Conventions:
//    - Interfaces must be compliant with lapacke.h in terms of signature and
//      naming conventions so wrapping a function "foo" which has the
//      signature
//         lapack_int LAPACKE_foo(int, char, lapack_int, float* , lapack_int)
//      within lapacke.h should result in a wrapper with the following signature
//         int MXNET_LAPACK_foo(int, char, int, float* , int)
//      Note that function signatures in lapacke.h will always have as first
//      argument the storage order (row/col-major). All wrappers have to support
//      that argument. The underlying fortran functions will always assume a
//      column-major layout. It is the responsibility of the wrapper function
//      to handle the (usual) case that it is called with data in row-major
//      format, either by doing appropriate transpositions explicitly or using
//      transposition options of the underlying fortran function.
//    - It is ok to assume that matrices are stored in contiguous memory
//      (which removes the need to do special handling for lda/ldb parameters
//      and enables us to save additional matrix transpositions around
//      the fortran calls).
//    - It is desired to add some basic checking in the C++-wrappers in order
//      to catch simple mistakes when calling these wrappers.
//    - Must support compilation without lapack-package but issue runtime error in this case.

#include <algorithm>
#include "dmlc/logging.h"
#include "mshadow/tensor.h"

using namespace mshadow;

extern "C" {
  // Fortran signatures
  #define MXNET_LAPACK_FSIGNATURE1(func, dtype) \
    void func##_(char* uplo, int* n, dtype* a, int* lda, int *info);

  MXNET_LAPACK_FSIGNATURE1(spotrf, float)
  MXNET_LAPACK_FSIGNATURE1(dpotrf, double)
  MXNET_LAPACK_FSIGNATURE1(spotri, float)
  MXNET_LAPACK_FSIGNATURE1(dpotri, double)

  void dposv_(char *uplo, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb, int *info);

  void sposv_(char *uplo, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb, int *info);

  void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda, double *s,
      double *u, int *ldu, double *vt, int *ldvt, double *work,
      int *lwork, int *iwork, int *info);

  void sgesdd_(char *jobz, int *m, int *n, float *a, int *lda, float *s,
      float *u, int *ldu, float *vt, int *ldvt, float *work,
      int *lwork, int *iwork, int *info);
}

#define MXNET_LAPACK_ROW_MAJOR 101
#define MXNET_LAPACK_COL_MAJOR 102

#define CHECK_LAPACK_CONTIGUOUS(a, b) \
  CHECK_EQ(a, b) << "non contiguous memory for array in lapack call";

#define CHECK_LAPACK_UPLO(a) \
  CHECK(a == 'U' || a == 'L') << "neither L nor U specified as triangle in lapack call";

inline char loup(char uplo, bool invert) { return invert ? (uplo == 'U' ? 'L' : 'U') : uplo; }


/*!
 * \brief Transpose matrix data in memory
 *
 * Equivalently we can see it as flipping the layout of the matrix
 * between row-major and column-major.
 *
 * \param m number of rows of input matrix a
 * \param n number of columns of input matrix a
 * \param b output matrix
 * \param ldb leading dimension of b
 * \param a input matrix
 * \param lda leading dimension of a
 */
template <typename xpu, typename DType>
inline void flip(int m, int n, DType *b, int ldb, DType *a, int lda);

template <>
inline void flip<cpu, float>(int m, int n,
  float *b, int ldb, float *a, int lda) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      b[j * ldb + i] = a[i * lda + j];
}

template <>
inline void flip<cpu, double>(int m, int n,
  double *b, int ldb, double *a, int lda) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      b[j * ldb + i] = a[i * lda + j];
}


#if MXNET_USE_LAPACK

  #define MXNET_LAPACK_CWRAPPER1(func, dtype) \
  inline int MXNET_LAPACK_##func(int matrix_layout, char uplo, int n, dtype* a, int lda ) { \
    CHECK_LAPACK_CONTIGUOUS(n, lda); \
    CHECK_LAPACK_UPLO(uplo); \
    char o(loup(uplo, (matrix_layout == MXNET_LAPACK_ROW_MAJOR))); \
    int ret(0); \
    func##_(&o, &n, a, &lda, &ret); \
    return ret; \
  }
  MXNET_LAPACK_CWRAPPER1(spotrf, float)
  MXNET_LAPACK_CWRAPPER1(dpotrf, double)
  MXNET_LAPACK_CWRAPPER1(spotri, float)
  MXNET_LAPACK_CWRAPPER1(dpotri, double)

  // add matrix_layout handling to sposv
  inline int MXNET_LAPACK_sposv(int matrix_layout, char uplo, int n, int nrhs,
    float *a, int lda, float *b, int ldb) {
    int info;
    if (matrix_layout == MXNET_LAPACK_ROW_MAJOR) {
      // Transpose b to b_t of shape (nrhs, n)
      float *b_t = new float[nrhs * n];
      flip<cpu, float>(n, nrhs, b_t, n, b, ldb);
      sposv_(&uplo, &n, &nrhs, a, &lda, b_t, &n, &info);
      flip<cpu, float>(nrhs, n, b, ldb, b_t, n);
      delete [] b_t;
      return info;
    }
    sposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
  }

  // add matrix_layout handling to dposv
  inline int MXNET_LAPACK_dposv(int matrix_layout, char uplo, int n, int nrhs,
    double *a, int lda, double *b, int ldb) {
    int info;
    if (matrix_layout == MXNET_LAPACK_ROW_MAJOR) {
      // Transpose b to b_t of shape (nrhs, n)
      double *b_t = new double[nrhs * n];
      flip<cpu, double>(n, nrhs, b_t, n, b, ldb);
      dposv_(&uplo, &n, &nrhs, a, &lda, b_t, &n, &info);
      flip<cpu, double>(nrhs, n, b, ldb, b_t, n);
      delete [] b_t;
      return info;
    }
    dposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
  }

  // add matrix_layout handling to sgesdd
  inline int MXNET_LAPACK_sgesdd(int matrix_layout,
    char jobz, int m, int n, float *a, int lda,
    float *s, float *u, int ldu, float *vt, int ldvt) {
    int info, lwork;
    float *work, wkopt;
    int *iwork = new int[8 * std::min(m, n)];

    if (matrix_layout == MXNET_LAPACK_ROW_MAJOR) {
      lwork = -1;
      sgesdd_(&jobz, &n, &m, a, &lda, s, vt, &ldvt, u, &ldu,
        &wkopt, &lwork, iwork, &info);
      if (info != 0) {
        delete [] iwork;
        return info;
      }

      lwork = static_cast<int>(wkopt);
      work = new float[lwork];
      sgesdd_(&jobz, &n, &m, a, &lda, s, vt, &ldvt, u, &ldu,
        work, &lwork, iwork, &info);
    } else {
      lwork = -1;
      sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
        &wkopt, &lwork, iwork, &info);
      if (info != 0) {
        delete [] iwork;
        return info;
      }

      lwork = static_cast<int>(wkopt);
      work = new float[lwork];
      sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
        work, &lwork, iwork, &info);
    }

    delete [] work;
    delete [] iwork;
    return info;
  }

  // add matrix_layout handling to dgesdd
  inline int MXNET_LAPACK_dgesdd(int matrix_layout,
    char jobz, int m, int n, double *a, int lda,
    double *s, double *u, int ldu, double *vt, int ldvt) {
    int info, lwork;
    double *work, wkopt;
    int *iwork = new int[8 * std::min(m, n)];

    if (matrix_layout == MXNET_LAPACK_ROW_MAJOR) {
      lwork = -1;
      dgesdd_(&jobz, &n, &m, a, &lda, s, vt, &ldvt, u, &ldu,
        &wkopt, &lwork, iwork, &info);
      if (info != 0) {
        delete [] iwork;
        return info;
      }

      lwork = static_cast<int>(wkopt);
      work = new double[lwork];
      dgesdd_(&jobz, &n, &m, a, &lda, s, vt, &ldvt, u, &ldu,
        work, &lwork, iwork, &info);
    } else {
      lwork = -1;
      dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
        &wkopt, &lwork, iwork, &info);
      if (info != 0) {
        delete [] iwork;
        return info;
      }

      lwork = static_cast<int>(wkopt);
      work = new double[lwork];
      dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
        work, &lwork, iwork, &info);
    }

    delete [] work;
    delete [] iwork;
    return info;
  }

#else
  // use pragma message instead of warning
  #pragma message("Warning: lapack usage not enabled, linalg-operators will be not available." \
                  " Build with USE_LAPACK=1 to get lapack functionalities.")

  // Define compilable stubs.
  #define MXNET_LAPACK_CWRAPPER1(func, dtype) \
  inline int MXNET_LAPACK_##func(int matrix_layout, char uplo, int n, dtype* a, int lda ) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  #define MXNET_LAPACK_UNAVAILABLE(func) \
  inline int MXNET_LAPACK_##func(...) { \
    LOG(FATAL) << "MXNet build without lapack. Function " << #func << " is not available."; \
    return 1; \
  }

  MXNET_LAPACK_CWRAPPER1(spotrf, float)
  MXNET_LAPACK_CWRAPPER1(dpotrf, double)
  MXNET_LAPACK_CWRAPPER1(spotri, float)
  MXNET_LAPACK_CWRAPPER1(dpotri, double)

  MXNET_LAPACK_UNAVAILABLE(sposv)
  MXNET_LAPACK_UNAVAILABLE(dposv)
  MXNET_LAPACK_UNAVAILABLE(sgesdd)
  MXNET_LAPACK_UNAVAILABLE(dgesdd)


#endif

template <typename DType>
inline int MXNET_LAPACK_posv(int matrix_layout, char uplo, int n, int nrhs,
  DType *a, int lda, DType *b, int ldb);

template <>
inline int MXNET_LAPACK_posv<float>(int matrix_layout, char uplo, int n,
  int nrhs, float *a, int lda, float *b, int ldb) {
  return MXNET_LAPACK_sposv(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
}

template <>
inline int MXNET_LAPACK_posv<double>(int matrix_layout, char uplo, int n,
  int nrhs, double *a, int lda, double *b, int ldb) {
  return MXNET_LAPACK_dposv(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
}

template <typename DType>
inline int MXNET_LAPACK_gesdd(int matrix_layout,
  char jobz, int m, int n, DType *a, int lda,
  DType *s, DType *u, int ldu, DType *vt, int ldvt);

template <>
inline int MXNET_LAPACK_gesdd<float>(int matrix_layout,
  char jobz, int m, int n, float *a, int lda,
  float *s, float *u, int ldu, float *vt, int ldvt) {
  return MXNET_LAPACK_sgesdd(matrix_layout, jobz, m, n, a, lda,
    s, u, ldu, vt, ldvt);
}

template <>
inline int MXNET_LAPACK_gesdd<double>(int matrix_layout,
  char jobz, int m, int n, double *a, int lda,
  double *s, double *u, int ldu, double *vt, int ldvt) {
  return MXNET_LAPACK_dgesdd(matrix_layout, jobz, m, n, a, lda,
    s, u, ldu, vt, ldvt);
}


#endif  // MXNET_C_LAPACK_API_H_
