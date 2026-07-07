#pragma once

#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/complex.h>

namespace THRUST
{

///////////////////////
//---Thrust Linear---//
///////////////////////

template<typename L> 
struct Thrust_Lin
{
public:
	Thrust_Lin()
	{}

	thrust::device_vector<L> ax(L s, const thrust::device_vector<L> &v) const;
	thrust::device_vector<thrust::complex<L>> ax(L s, const thrust::device_vector<thrust::complex<L>> &v) const;
	thrust::device_vector<thrust::complex<L>> ax(thrust::complex<L> s, const thrust::device_vector<thrust::complex<L>> &v) const;

	thrust::device_vector<L> axpy(L s, thrust::device_vector<L> &v1, const thrust::device_vector<L> &v2) const;
	thrust::device_vector<thrust::complex<L>> axpy(L s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2) const;
	thrust::device_vector<thrust::complex<L>> axpy(thrust::complex<L> s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2) const;
	thrust::device_vector<thrust::complex<L>> axpy(thrust::complex<L> s, thrust::device_vector<L> &v1, const thrust::device_vector<thrust::complex<L>> &v2) const;

	L thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<L> &v2) const;
	thrust::complex<L> thrust_Dotprod(thrust::device_vector<thrust::complex<L>> &v1, thrust::device_vector<thrust::complex<L>> &v2) const;
	thrust::complex<L> thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<thrust::complex<L>> &v2) const;

	void normalize(thrust::device_vector<L> &v) const;
	void normalize(thrust::device_vector<thrust::complex<L>> &v) const;
	L norm(thrust::device_vector<L> &v) const;
	L norm(thrust::device_vector<thrust::complex<L>> &v) const;
};

template<typename L>
struct ABS
{
	__host__ __device__ L operator()(const L &x) const
	{ return thrust::abs(x); }
};

template<typename L>
struct ABS2
{
	__host__ __device__ L operator()(const thrust::complex<L> &x) const
	{ return (thrust::conj(x)*x).real(); }
};

template<typename L>
struct square
{
	__host__ __device__ L operator()(const L &x) const
	{ return x*x; }
};

template<typename L>
struct complex_prod : public thrust::binary_function<thrust::complex<L>,thrust::complex<L>,thrust::complex<L>>
{
	__host__ __device__ thrust::complex<L> operator()(const thrust::complex<L> &a, const thrust::complex<L> &b) const
	{ return thrust::conj(a)*b; }	
};

template<typename L>
struct real_comp_prod
{
	__host__ __device__ thrust::complex<L> operator()(const L &a, const thrust::complex<L> &b) const
	{ return a*b; }	
};

template<typename L>
struct ax_
{
	const L a;
	
	ax_(L a):a(a)
	{}

	__host__ __device__ L operator()(const L &x)
	{
		return a*x;	
	}
};

template<typename K, typename L>
struct axpy_
{
	const L a;
	
	axpy_(L a):a(a)
	{}

	__host__ __device__ L operator()(const K &x, const L &y)
	{
		return a*x+y;	
	}
};

template<typename L>
void Thrust_Lin<L>::normalize(thrust::device_vector<L> &v) const
{
	square<L> unary_op;
	thrust::plus<L> binary_op;
	L init = 0.0;
	L sum = thrust::transform_reduce(v.begin(),v.end(),unary_op,init,binary_op);
	
	thrust::transform(v.begin(),v.end(),v.begin(),ax_<L>(1./sqrt(sum)));
}

template<typename L>
void Thrust_Lin<L>::normalize(thrust::device_vector<thrust::complex<L>> &v) const
{
	ABS2<L> unary_op;
	thrust::plus<L> binary_op;
	L init = 0.0;
	L sum = thrust::transform_reduce(v.begin(),v.end(),unary_op,init,binary_op);
	
    thrust::complex<double> oneOVERnorm = thrust::complex<double>(1.0/sqrt(sum),0.0);
	thrust::transform(v.begin(),v.end(),v.begin(),ax_<thrust::complex<L>>(oneOVERnorm));
}

template<typename L>
L Thrust_Lin<L>::norm(thrust::device_vector<L> &v) const
{
	square<L> unary_op;
	thrust::plus<L> binary_op;
	L init = 0.0;
	L sum = thrust::transform_reduce(v.begin(),v.end(),unary_op,init,binary_op);

	return sum;
}

template<typename L>
L Thrust_Lin<L>::norm(thrust::device_vector<thrust::complex<L>> &v) const
{
	ABS2<L> unary_op;
	thrust::plus<L> binary_op;
	L init = 0.0;
	L sum = thrust::transform_reduce(v.begin(),v.end(),unary_op,init,binary_op);

	return sum;
}

template<typename L>
thrust::device_vector<L> Thrust_Lin<L>::ax(L s, const thrust::device_vector<L> &v) const
{
    int dim = v.size();
	thrust::device_vector<L> w(dim);
	thrust::transform(v.begin(),v.end(),w.begin(),ax_<L>(s)); 
	
	return w;	
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::ax(L s, const thrust::device_vector<thrust::complex<L>> &v) const
{
    int dim = v.size();
	thrust::device_vector<thrust::complex<L>> w(dim);
	thrust::transform(v.begin(),v.end(),w.begin(),ax_<thrust::complex<L>>(thrust::complex<L>(s,0.))); 
	
	return w;	
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::ax(thrust::complex<L> s, const thrust::device_vector<thrust::complex<L>> &v) const
{
    int dim = v.size();
	thrust::device_vector<thrust::complex<L>> w(dim);
	thrust::transform(v.begin(),v.end(),w.begin(),ax_<thrust::complex<L>>(s)); 
	
	return w;	
}

template<typename L>
thrust::device_vector<L> Thrust_Lin<L>::axpy(L s, thrust::device_vector<L> &v1, const thrust::device_vector<L> &v2) const 
{
    int dim = v1.size();
	thrust::device_vector<L> w(dim);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<L,L>(s));

	return w;
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::axpy(L s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2) const
{
    int dim = v1.size();
	thrust::device_vector<thrust::complex<L>> w(dim);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<thrust::complex<L>,thrust::complex<L>>(thrust::complex<L>(s,0.)));

	return w;
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::axpy(thrust::complex<L> s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2) const
{
    int dim = v1.size();
	thrust::device_vector<thrust::complex<L>> w(dim);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<thrust::complex<L>,thrust::complex<L>>(s));

	return w;
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::axpy(thrust::complex<L> s, thrust::device_vector<L> &v1, const thrust::device_vector<thrust::complex<L>> &v2) const
{
    int dim = v1.size();
	thrust::device_vector<thrust::complex<L>> w(dim);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<L,thrust::complex<L>>(s));

	return w;
}

template<typename L>
L Thrust_Lin<L>::thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<L> &v2) const
{
	return thrust::inner_product(v1.begin(),v1.end(),v2.begin(),0.0);
}

template<typename L>
thrust::complex<L> Thrust_Lin<L>::thrust_Dotprod(thrust::device_vector<thrust::complex<L>> &v1, thrust::device_vector<thrust::complex<L>> &v2) const
{
	thrust::complex<L> w = thrust::inner_product(v1.begin(),v1.end(),v2.begin(),
			thrust::complex<L>(0.0,0.0),thrust::plus<thrust::complex<L>>(),complex_prod<L>());
	return w;
}

template<typename L>
thrust::complex<L> Thrust_Lin<L>::thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<thrust::complex<L>> &v2) const
{
	thrust::complex<L> w = thrust::inner_product(v1.begin(),v1.end(),v2.begin(),
			thrust::complex<L>(0.0,0.0),thrust::plus<thrust::complex<L>>(),real_comp_prod<L>());
	return w;
}

} // namespace THRUST
