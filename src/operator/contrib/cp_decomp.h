/*!
 *  Copyright (c) 2017 by Contributors
 *  \file cp_decomp.h
 *  \brief Core function performing CP Decomposition
 *  \author Jencir Lee
 */
#ifndef MXNET_OPERATOR_CONTRIB_CP_DECOMP_H_
#define MXNET_OPERATOR_CONTRIB_CP_DECOMP_H_
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <utility>
#include <numeric>
#include <iostream>
#include "dmlc/logging.h"
#include "mshadow/tensor.h"
#include "./krprod.h"
#include "./unfold.h"
#include "../tensor/broadcast_reduce-inl.h"
#include "../tensor/la_op_inline.h"
#include "mxnet/c_lapack_api.h"


namespace mxnet {
namespace op {

using namespace mshadow;
using namespace mshadow::expr;

template <typename DType>
inline void print1DTensor_(const Tensor<cpu, 1, DType> &t);

template <typename DType>
inline void print2DTensor_(const Tensor<cpu, 2, DType> &t);

/*!
 * \brief Evaluates the reconstruction error of the CP Decomposition
 *
 * Reconstruction error = \lVert t - [eigvals; factors] \rVert_2
 *
 * \param t input tensor
 * \param eigvals eigen-value vector of the CP Decomposition
 * \param factors_T array of transposed factor matrices
 * \return the reconstruction error
 */
template <int order, typename DType>
inline DType CPDecompReconstructionError
  (const Tensor<cpu, order, DType> &t,
  const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T);

/*!
 * \brief Randomly initialise the transposed factor matrices for the CP Decomposition
 *
 * All factor matrices will be filled with random numbers from the standard Gaussian distribution. Optionally they could be initialised with left singular vectors of the unfolded tensor.
 *
 * \param factors_T array of transposed factor matrices to be initialised
 * \param orthonormal whether to orthogonalise the factor matrices
 * \param generator C++ random generator from <random>
 * \return 0 for success, non-zero otherwise
 */
template <typename RandomEngine, typename DType>
inline int CPDecompInitFactors
  (std::vector<Tensor<cpu, 2, DType> > factors_T,
  const std::vector<Tensor<cpu, 2, DType> > &unfoldings,
  bool orthonormal,
  RandomEngine *generator);

/*!
 * \brief Update one transposed factor matrix during one step of the ALS algorithm
 *
 * This function first solves a least-squre problem for factors_T[mode] whilst fixing all the other factor matrices
 *
 *    min \lVert unfolding - [I; factors] \rVert_2
 *
 * where I is all-one eigenvector, then normalise the obtained factors_T[mode] by row.
 *
 * This function is called by CPDecomp().
 *
 * \param eigvals eigen-value vector to be updated
 * \param factors_T array of transposed factor matrices
 * \param unfolding unfolded tensor along the specified mode
 * \param mode the index of the transposed factor matrix to be updated
 * \return 0 if success, non-zero otherwise
 */
template <typename DType>
inline int CPDecompUpdate
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T,
  const Tensor<cpu, 2, DType> &unfolding,
  int mode);

/*!
 * \brief Evaluates if the ALS algorithm has converged
 *
 * Evaluates if across two steps of the ALS algorithm, the relative norm of error between the eigenvectors or every row of every transposed factor matrix is below a constant eps.
 *
 * \param eigvals eigen-value vector at time t+1
 * \param factors_T transposed factor matrices at time t+1
 * \param oldEigvals eigen-value vector at time t
 * \param oldFactors_T transposed factor matrices at time t+1
 * \param eps relative error threshold
 * \return if all vectors converged
 */
template <typename DType>
inline bool CPDecompConverged
  (const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T,
  const Tensor<cpu, 1, DType> &oldEigvals,
  const std::vector<Tensor<cpu, 2, DType> > &oldFactors_T,
  DType eps);

/*!
 * \brief Sort the eigenvalue vector in descending order and the rows of transposed factor matrices accordingly
 *
 * \param eigvals eigenvalue vector obtained from CPDecomp
 * \param factors_T transposed factor matrices obtained from CPDecomp
 */
template <typename DType>
inline void CPDecompSortResults
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T);

/*!
 * \brief Perform CANDECOMP/PARAFAC Decomposition on CPU
 *
 * This function performs CP Decompsition for input tensor of arbitrary order and shape on CPU. At success, it populates `eigvals` and `factors_T` with the eigen-value vector sorted in descending order and transposed factor matrices, and returns 0; otherwise returns 1.
 *
 * Internally it uses an iterative algorithm with random initial matrices and may not necessarily converge to the same solution from run to run.
 *
 * \param eigvals eigen-value vector to be populated
 * \param factors_T array of the transposed factor matrices to be populated
 * \param in input tensor
 * \param k rank for the CP Decomposition
 * \param eps relative error thershold for checking convergence, default: 1e-6
 * \param max_iter maximum iterations for each run of the ALS algorithm, default: 100
 * \param restarts number of runs for the ALS algorithm, default: 5
 * \param init_orthonormal_factors whether to initialise the factor matrices with orthonormal columns, default: true
 * \param stream calculation stream (for GPU)
 * \return 0 if success, 1 otherwise
 */
template <int order, typename DType>
inline int CPDecomp
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T,
  const Tensor<cpu, order, DType> &in,
  int k,
  DType eps = 1e-6,
  int max_iter = 200,
  int restarts = 5,
  bool init_orthonormal_factors = true,
  Stream<cpu> *stream = NULL) {
  CHECK_GE(order, 2);
  CHECK_GE(k, 1);

  CHECK_EQ((int) eigvals.size(0), k);
  CHECK_EQ((int) factors_T.size(), order);
  CHECK_GE(k, 1);
  for (int i = 0; i < order; ++i) {
    CHECK_EQ((int) factors_T[i].size(0), k);
    CHECK_EQ(factors_T[i].size(1), in.size(i));
  }
  CHECK_GE(restarts, 1);

  // Return value
  int status;

  // Unfold the input tensor along specified mode
  // unfoldings[id_mode] is a matrix of shape
  // (in.size(id_mode), tensor_size / in.size(id_mode))
  std::vector<Tensor<cpu, 2, DType> > unfoldings;
  const int tensor_size = in.shape_.Size();
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    unfoldings.emplace_back
      (Shape2(in.size(id_mode), tensor_size / in.size(id_mode)));
    AllocSpace(&unfoldings[id_mode]);

    Unfold(unfoldings[id_mode], in, id_mode);
  }

  // Allocate space for old factor matrices A, B, C, etc,
  // transposed as well, of the same shapes as factors_T
  Tensor<cpu, 1, DType> currEigvals(Shape1(k)), oldEigvals(Shape1(k));
  AllocSpace(&currEigvals);
  AllocSpace(&oldEigvals);

  std::vector<Tensor<cpu, 2, DType> > currFactors_T;
  std::vector<Tensor<cpu, 2, DType> > oldFactors_T;
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    currFactors_T.emplace_back(factors_T[id_mode].shape_);
    oldFactors_T.emplace_back(factors_T[id_mode].shape_);
    AllocSpace(&currFactors_T[id_mode]);
    AllocSpace(&oldFactors_T[id_mode]);
  }

  // Initialise random generator
  std::random_device rnd_device;
  std::mt19937 generator(rnd_device());

  // Starting multi-runs of ALS
  DType reconstructionError = std::numeric_limits<DType>::infinity();
  DType currReconstructionError;
  for (int id_run = 0; id_run < restarts; ++id_run) {
    // Randomly initialise factor matrices
    // If any error go to the next run of the multi-starts
    status = CPDecompInitFactors(currFactors_T, unfoldings,
      init_orthonormal_factors, &generator);
    if (status != 0) {
      LOG(WARNING) << "Init error\n";
      continue;
    }

    // ALS
    //
    // If any step produces error, we terminate early the ALS
    // and go to the next run of the multi-starts
    int iter = 0;
    while (iter < max_iter
        && status == 0
        && (iter == 0 || !CPDecompConverged(currEigvals, currFactors_T,
                                     oldEigvals, oldFactors_T, eps))) {
      Copy(oldEigvals, currEigvals);
      for (int id_mode = 0; id_mode < order; ++id_mode)
        Copy(oldFactors_T[id_mode], currFactors_T[id_mode]);

      for (int id_mode = 0; id_mode < order; ++id_mode) {
        status = CPDecompUpdate(currEigvals, currFactors_T,
          unfoldings[id_mode], id_mode);
        if (status != 0) {
          LOG(WARNING) << "Iter " << iter << " Update error\n";
          break;
        }
      }

      ++iter;
    }

    // If any error we won't consider updating the optimal factor matrices
    // and the optimal reconstruction error for the current run
    if (status != 0)
      continue;

    // Update the optimal reconstruction error and factor matrices
    currReconstructionError = CPDecompReconstructionError
      (in, currEigvals, currFactors_T);
#if DEBUG
    print1DTensor_(currEigvals);
    std::cerr << "Reconstruction error: " << currReconstructionError << "\n";
#endif
    if (currReconstructionError < reconstructionError) {
      Copy(eigvals, currEigvals);
      for (int id_mode = 0; id_mode < order; ++id_mode)
        Copy(factors_T[id_mode], currFactors_T[id_mode]);
      reconstructionError = currReconstructionError;
    }
  }

  // Free up space
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    FreeSpace(&unfoldings[id_mode]);
    FreeSpace(&currFactors_T[id_mode]);
    FreeSpace(&oldFactors_T[id_mode]);
  }

  FreeSpace(&currEigvals);
  FreeSpace(&oldEigvals);

  // Return success only if the optimal reconstruction error
  // has been updated at least once
  if (reconstructionError < std::numeric_limits<DType>::infinity()) {
    CPDecompSortResults(eigvals, factors_T);
    return 0;
  } else {
    return 1;
  }
}

template <int order, typename DType>
inline DType CPDecompReconstructionError
  (const Tensor<cpu, order, DType> &t,
  const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T) {
  int k = eigvals.size(0);

  Shape<order> strides = t.shape_;
  strides[order - 1] = t.stride_;

  Shape<order> coord;
  DType sum_sq = 0, delta;
  DType reconstructedElement, c;
  for (int flat_id = 0;
      flat_id < static_cast<int>(t.shape_.Size()); ++flat_id) {
    coord = mxnet::op::broadcast::unravel(flat_id, t.shape_);

    reconstructedElement = 0;
    for (int i = 0; i < k; ++i) {
      c = eigvals[i];
      for (int id_mode = 0; id_mode < order; ++id_mode)
        c *= factors_T[id_mode][i][coord[id_mode]];

      reconstructedElement += c;
    }

    delta = t.dptr_[ravel_multi_index(coord, strides)] - reconstructedElement;
    sum_sq += delta * delta;
  }

  return std::sqrt(sum_sq);
}

template <typename RandomEngine, typename DType>
inline int CPDecompInitFactors
  (std::vector<Tensor<cpu, 2, DType> > factors_T,
  const std::vector<Tensor<cpu, 2, DType> > &unfoldings,
  bool orthonormal,
  RandomEngine *generator) {
  int status = 0;

  int order = static_cast<int>(factors_T.size());
  int k = factors_T[0].size(0);
  for (const auto &mat : factors_T)
    CHECK_EQ((int) mat.size(0), k);

  // TODO(jli05): implement seed for random generator
  std::normal_distribution<DType> normal(0.0, 1.0);

  int num_singular_values;
  DType *u, *s, *vt;

  // Make a copy of the unfolding as after the _gesdd call
  // the contents of it will be destroyed
  std::vector<TensorContainer<cpu, 2, DType> > _unfoldings;

  for (int id_mode = 0; id_mode < order; ++id_mode) {
    // fill factors_T[id_mode] with random numbers
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < static_cast<int>(factors_T[id_mode].size(1)); ++j)
        factors_T[id_mode][i][j] = normal(*generator);
    }

    // optionally, we fill factors_T[id_mode] with left singular
    // vectors of unfoldings[id_mode] to increase CPDecomp stability
    if (orthonormal) {
      num_singular_values = std::min
        (static_cast<int>(unfoldings[id_mode].size(0)),
        static_cast<int>(unfoldings[id_mode].size(1)));
      u = new DType[unfoldings[id_mode].size(0) * num_singular_values];
      s = new DType[num_singular_values];
      vt = new DType[num_singular_values * unfoldings[id_mode].size(1)];

      _unfoldings.emplace_back(unfoldings[id_mode].shape_);
      Copy(_unfoldings[id_mode], unfoldings[id_mode]);

      status = MXNET_LAPACK_gesdd<DType>(MXNET_LAPACK_ROW_MAJOR, 'S',
        static_cast<int>(_unfoldings[id_mode].size(0)),
        static_cast<int>(_unfoldings[id_mode].size(1)),
        _unfoldings[id_mode].dptr_, _unfoldings[id_mode].stride_,
        s, u, num_singular_values, vt,
        static_cast<int>(_unfoldings[id_mode].size(1)));
      if (status != 0) {
        delete [] u;
        delete [] s;
        delete [] vt;
        return status;
      }

      for (int i = 0; i < std::min(k, num_singular_values); ++i) {
        for (int j = 0; j < static_cast<int>(factors_T[id_mode].size(1)); ++j)
        factors_T[id_mode][i][j] = u[j * num_singular_values + i];
      }
      print2DTensor_(factors_T[id_mode]);

      delete [] u;
      delete [] s;
      delete [] vt;
    }
  }

  return status;
}

template <typename DType>
inline int CPDecompUpdate
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T,
  const Tensor<cpu, 2, DType> &unfolding,
  int mode) {
  int order = static_cast<int>(factors_T.size());
  int k = eigvals.size(0);

  for (auto &m : factors_T)
    CHECK_EQ(static_cast<int>(m.size(0)), k);

  // Compute dot(inv_krprod(ts_arr), unfolding.T()), where ts_arr
  // is the reversed factors_T, excluding the original factors_T[mode].
  //
  // dot(inv_krprod(ts_arr), unfolding.T()) just gives the updated
  // factors_T[mode], in the unnormalised form.

  // Prepare ts_arr by reversing factors_T, and excluding the original
  // factors_T[mode]
  int info;
  std::vector<Tensor<cpu, 2, DType> > ts_arr;
  for (int id_mode = order - 1; id_mode >= 0; --id_mode) {
    if (id_mode == mode)
      continue;
    ts_arr.push_back(factors_T[id_mode]);
  }

  // Compute inv_krprod for the tranposed factor matrices in ts_arr
  Tensor<cpu, 2, DType> inv_kr(Shape2(k, unfolding.size(1)));
  AllocSpace(&inv_kr);
  inv_khatri_rao(inv_kr, ts_arr, true);

  // Update the current transposed factor matrix
  factors_T[mode] = implicit_dot(inv_kr, unfolding.T());

  // Normalise by row
  for (int j = 0; j < k; ++j) {
    eigvals[j] = nrm2(factors_T[mode].size(1),
        factors_T[mode][j].dptr_, 1);
    factors_T[mode][j] /= eigvals[j];
  }

  FreeSpace(&inv_kr);
  return 0;
}

template <typename DType>
inline bool CPDecompConverged
  (const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T,
  const Tensor<cpu, 1, DType> &oldEigvals,
  const std::vector<Tensor<cpu, 2, DType> > &oldFactors_T,
  DType eps) {
  int k = eigvals.size(0);

  // Check if the relative norm of error is below eps for eigvals
  TensorContainer<cpu, 1, DType> eigval_diff(eigvals.shape_);
  eigval_diff = eigvals - oldEigvals;
  if (nrm2(k, eigval_diff.dptr_, 1)
      > eps * nrm2(k, oldEigvals.dptr_, 1))
    return false;

  // Check if the relative norm of error is below eps for every row
  // of every transposed factor matrix
  int d;
  for (int p = 0; p < static_cast<int>(factors_T.size()); ++p) {
    d = factors_T[p].size(1);
    TensorContainer<cpu, 2, DType> factors_diff(factors_T[p].shape_);
    factors_diff = factors_T[p] - oldFactors_T[p];

    for (int i = 0; i < k; ++i) {
      if (nrm2(d, factors_diff[i].dptr_, 1)
          > eps * nrm2(d, oldFactors_T[p][i].dptr_, 1))
        return false;
    }
  }

  return true;
}

// Argsort 1D Tensor in descending order
template <typename DType>
std::vector<int> sort_indexes(const Tensor<cpu, 1, DType> &v) {
  // initialize original index locations
  std::vector<int> idx(v.size(0));
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
}

template <typename DType>
inline void CPDecompSortResults
  (Tensor<cpu, 1, DType> eigvals,
  std::vector<Tensor<cpu, 2, DType> > factors_T) {
  int order = factors_T.size();
  int k = eigvals.size(0);

  // Create temporary tensors
  TensorContainer<cpu, 1, DType> eigvals_(eigvals.shape_);
  std::vector<TensorContainer<cpu, 2, DType> > factors_T_;
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    factors_T_.emplace_back(factors_T[id_mode].shape_);
  }

  // Args of eigvals in descending order
  std::vector<int> idx = sort_indexes(eigvals);

  // Sort the eigvals and rows of every transposed factor matrix into
  // the temporary tensors
  for (int i = 0; i < k; ++i) {
    eigvals_[i] = eigvals[idx[i]];
    for (int id_mode = 0; id_mode < order; ++id_mode) {
      for (int j = 0; j < static_cast<int>(factors_T[id_mode].size(1)); ++j)
        factors_T_[id_mode][i][j] = factors_T[id_mode][idx[i]][j];
    }
  }

  // Copy the temporary tensors into final results
  Copy(eigvals, eigvals_);
  for (int id_mode = 0; id_mode < order; ++id_mode) {
    Copy(factors_T[id_mode], factors_T_[id_mode]);
  }
}

template <typename DType>
inline void print1DTensor_(const Tensor<cpu, 1, DType> &t) {
  for (int i = 0; i < static_cast<int>(t.size(0)); ++i)
    std::cerr << t[i] << " ";
  std::cerr << "\n";
}

template <typename DType>
inline void print2DTensor_(const Tensor<cpu, 2, DType> &t) {
  for (int i = 0; i < static_cast<int>(t.size(0)); ++i) {
    for (int j = 0; j < static_cast<int>(t.size(1)); ++j)
      std::cerr << t[i][j] << " ";
    std::cerr << "\n";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_CP_DECOMP_H_
