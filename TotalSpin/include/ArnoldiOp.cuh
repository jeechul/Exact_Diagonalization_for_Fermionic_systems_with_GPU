#ifndef SPECTRA_ARNOLDI_OP_CUH
#define SPECTRA_ARNOLDI_OP_CUH

#include <Eigen/Core>
#include <cmath>  // std::sqrt
#include "cublas_template.cuh"
#include "thrust_util.cuh"

namespace cuSpectra {

///
/// \ingroup Internals
/// @{
///

///
/// \defgroup Operators Operators
///
/// Different types of operators.
///

///
/// \ingroup Operators
///
/// Operators used in the Arnoldi factorization.
///
template <typename Scalar, typename OpType, typename BOpType>
class ArnoldiOp
{
private:
    using Index = Eigen::Index;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ThrustLin = THRUST::Thrust_Lin<Scalar>;

    const OpType& m_op;
    const BOpType& m_Bop;
    mutable Vector m_cache;
    const ThrustLin thrust_lin;
    cublasHandle_t handle;

public:
    ArnoldiOp(const OpType& op, const BOpType& Bop) : 
        m_op(op), m_Bop(Bop), m_cache(op.rows()), thrust_lin(ThrustLin())
    {
        cublasCreate(&handle);
    }

    // Move constructor
    ArnoldiOp(ArnoldiOp&& other) :
        m_op(other.m_op), m_Bop(other.m_Bop)
    {
        // We emulate the move constructor for Vector using Vector::swap()
        m_cache.swap(other.m_cache);
    }

    inline Index rows() const { return m_op.rows(); }

    // In generalized eigenvalue problem Ax=lambda*Bx, define the inner product to be <x, y> = x'By.
    // For regular eigenvalue problems, it is the usual inner product <x, y> = x'y

    // Compute <x, y> = x'By
    // x and y are two vectors
    template <typename Arg1, typename Arg2>
    Scalar inner_product(const Arg1& x, const Arg2& y) const
    {
        m_Bop.perform_op(y.data(), m_cache.data());
       
        thrust::device_vector<Scalar> x_dev(x.size()), m_cache_dev(m_cache.size());
        thrust::copy(x.data(),x.data()+x.size(),x_dev.begin());
        thrust::copy(m_cache.data(),m_cache.data()+m_cache.size(),m_cache_dev.begin());

        return thrust_lin.thrust_Dotprod(x_dev,m_cache_dev);
    }

    // Compute res = <X, y> = X'By
    // X is a matrix, y is a vector, res is a vector
    template <typename Arg1, typename Arg2>
    void trans_product(const Arg1& x, const Arg2& y, Eigen::Ref<Vector> res) const
    {
        m_Bop.perform_op(y.data(), m_cache.data());
 
        Vector y_temp(x.cols());
        thrust::device_vector<Scalar> x_dev(x.size()), m_cache_dev(m_cache.size()), y_temp_dev(y_temp.size());

        thrust::copy(x.data(),x.data()+x.size(),x_dev.begin());
        thrust::copy(y_temp.data(),y_temp.data()+y_temp.size(),y_temp_dev.begin());
        thrust::copy(m_cache.data(),m_cache.data()+m_cache.size(),m_cache_dev.begin());

        const cublasOperation_t trans = CUBLAS_OP_T;
        Scalar alpha = 1.0, beta = 0.0;
        cublas::gemv(
            handle,trans,x.rows(),x.cols(),alpha,
            thrust::raw_pointer_cast(x_dev.data()),thrust::raw_pointer_cast(m_cache_dev.data()),beta,
            thrust::raw_pointer_cast(y_temp_dev.data())
        );

        thrust::copy(y_temp_dev.begin(),y_temp_dev.end(),y_temp.data());

        res.noalias() = y_temp;
    }

    // B-norm of a vector, ||x||_B = sqrt(x'Bx)
    template <typename Arg>
    Scalar norm(const Arg& x) const
    {
        thrust::device_vector<Scalar> x_dev(x.size());
        thrust::copy(x.data(),x.data()+x.size(),x_dev.begin());

        return std::sqrt(thrust_lin.norm(x_dev));
    }

    // The "A" operator to generate the Krylov subspace
    inline void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        m_op.perform_op(x_in, y_out);
    }
};

///
/// \ingroup Operators
///
/// Placeholder for the B-operator when \f$B = I\f$.
///
class IdentityBOp
{};

///
/// \ingroup Operators
///
/// Partial specialization for the case \f$B = I\f$.
///
template <typename Scalar, typename OpType>
class ArnoldiOp<Scalar, OpType, IdentityBOp>
{
private:
    using Index = Eigen::Index;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ThrustLin = THRUST::Thrust_Lin<Scalar>;

    const OpType& m_op; 
    const ThrustLin thrust_lin;
    cublasHandle_t handle;

public:
    ArnoldiOp(const OpType& op, const IdentityBOp& /*Bop*/) :
        m_op(op), thrust_lin(ThrustLin())
    {
        cublasCreate(&handle);
    }

    inline Index rows() const { return m_op.rows(); }

    // Compute <x, y> = x'y
    // x and y are two vectors
    template <typename Arg1, typename Arg2>
    Scalar inner_product(const Arg1& x, const Arg2& y) const
    {
        thrust::device_vector<Scalar> x_dev(x.size()), y_dev(y.size());
        thrust::copy(x.data(),x.data()+x.size(),x_dev.begin());
        thrust::copy(y.data(),y.data()+y.size(),y_dev.begin());

        Scalar res = thrust_lin.thrust_Dotprod(x_dev,y_dev);
        
        thrust::device_vector<Scalar>().swap(x_dev);
        thrust::device_vector<Scalar>().swap(y_dev);

        return res;
    }

    // Compute res = <X, y> = X'y
    // X is a matrix, y is a vector, res is a vector
    template <typename Arg1, typename Arg2>
    void trans_product(const Arg1& x, const Arg2& y, Eigen::Ref<Vector> res) const
    {
        Vector y_temp(x.cols());
        thrust::device_vector<Scalar> x_dev(x.size()), y_dev(y.size()), y_temp_dev(y_temp.size());

        thrust::copy(x.data(),x.data()+x.size(),x_dev.begin());
        thrust::copy(y.data(),y.data()+y.size(),y_dev.begin());
        thrust::copy(y_temp.data(),y_temp.data()+y_temp.size(),y_temp_dev.begin());

        // std::cerr << "#    x_rows, x_cols, y_dim : " << x.rows() << " " << x.cols() << " " << y.size() << " " << std::endl;
        const cublasOperation_t trans = CUBLAS_OP_T;
        Scalar alpha = 1.0, beta = 0.0;
        cublas::gemv(
            handle,trans,x.rows(),x.cols(),alpha,
            thrust::raw_pointer_cast(x_dev.data()),thrust::raw_pointer_cast(y_dev.data()),beta,
            thrust::raw_pointer_cast(y_temp_dev.data())
        );

        thrust::copy(y_temp_dev.begin(),y_temp_dev.end(),y_temp.data());

        res.noalias() = y_temp;

        thrust::device_vector<Scalar>().swap(x_dev);
        thrust::device_vector<Scalar>().swap(y_dev);
        thrust::device_vector<Scalar>().swap(y_temp_dev); 
    }

    // B-norm of a vector. For regular eigenvalue problems it is simply the L2 norm
    template <typename Arg>
    Scalar norm(const Arg& x) const
    {
        thrust::device_vector<Scalar> x_dev(x.size());
        thrust::copy(x.data(),x.data()+x.size(),x_dev.begin());

        Scalar res = std::sqrt(thrust_lin.norm(x_dev));

        thrust::device_vector<Scalar>().swap(x_dev);

        return res;
    }

    // The "A" operator to generate the Krylov subspace
    inline void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        m_op.perform_op(x_in, y_out);
    }
};

///
/// @}
///

}  // namespace cuSpectra

#endif  // SPECTRA_ARNOLDI_OP_CUH
