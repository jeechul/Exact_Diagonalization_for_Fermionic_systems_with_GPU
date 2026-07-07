#ifndef _HUBMATRIX_H_
#define _HUBMATRIX_H_

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
#include "MAGMA_LAPACK.h"
#include "fermi_operator.h"
#include "EIGEN_cusolver.h"

#define CUDAERR(x){\
	if(x != cudaSuccess) {\
		std::cout << "Error! : #" << __LINE__ << std::endl;\
	}\
}\

/////////////////////////////
//---Hubbard Hamiltonian---//
/////////////////////////////

struct Hubbard
{
	thrust::device_vector<double> AxUP_D;
	thrust::device_vector<int> AjUP_D;
	thrust::device_vector<double> AxDN_D;
	thrust::device_vector<int> AjDN_D;
	thrust::device_vector<int> stateUP;
	thrust::device_vector<int> stateDN;
public:
	unsigned long dimUP,dimDN;
    int BPS,N; 
	const int numcolsUP,numcolsDN; 	
	const double U;
	const bool is_diluted;
	int dilution;

	Hubbard(const H_TBq& HqUP, const H_TBq& HqDN, const double U=0.0):
		U(U), numcolsUP(z*HqUP.Q+1), numcolsDN(z*HqDN.Q+1), is_diluted(false)
	{
		dimUP = HqUP.count;
		dimDN = HqDN.count;
		N = HqUP.N;
		dilution = (1<<N)-1;

		for(int i=1;i<dimDN;i++){
			if(dimDN%i==0){
				if(dimDN/i<=1024){
					BPS=i;
					break;
				}
			}
		}
	
		AxUP_D = HqUP.Ax;
		AjUP_D = HqUP.Aj;
		AxDN_D = HqDN.Ax;
		AjDN_D = HqDN.Aj;
	
		stateUP = HqUP.state;
		stateDN = HqDN.state;
	}	
	
	Hubbard(const H_TBq& HqUP, const H_TBq& HqDN, const double U, const int dilution):
		U(U), dilution(dilution), numcolsUP(z*HqUP.Q+1), numcolsDN(z*HqDN.Q+1), is_diluted(true)
	{
		dimUP = HqUP.count;
		dimDN = HqDN.count;
		N = HqUP.N;

		for(int i=1;i<dimDN;i++){
			if(dimDN%i==0){
				if(dimDN/i<=1024){
					BPS=i;
					break;
				}
			}
		}

		AxUP_D = HqUP.Ax;
		AjUP_D = HqUP.Aj;
		AxDN_D = HqDN.Ax;
		AjDN_D = HqDN.Aj;
	
		stateUP = HqUP.state;
		stateDN = HqDN.state;
	}

	template<typename M>
	void Hubbard_mv(thrust::device_vector<M> &x, thrust::device_vector<M> &y);
	void Write(thrust::device_vector<double> &Hmat);	
	void Write_pinned(double* Hmat);
	void initialize(double* Hmat);	

	~Hubbard()
	{
		AxUP_D.clear(); thrust::device_vector<double>().swap(AxUP_D);
		AjUP_D.clear(); thrust::device_vector<int>().swap(AjUP_D);
		AxDN_D.clear(); thrust::device_vector<double>().swap(AxDN_D);
		AjDN_D.clear(); thrust::device_vector<int>().swap(AjDN_D);
		stateUP.clear(); thrust::device_vector<int>().swap(stateUP);
		stateDN.clear(); thrust::device_vector<int>().swap(stateDN);
	}
};

template<typename M>
__global__ void Hubbard_mv_(const double * AxUP, const int * AjUP, const double * AxDN, const int * AjDN, 
		const int * stateUP, const int * stateDN, M * x, M * y, 
        const unsigned long dimUP, const unsigned long dimDN, const int numcolsUP, const int numcolsDN, 
		const double U, const int dilution, const int N, const int blocksPerSubvector)
{
	M sum=0;
	int threadsPerBlock = dimDN/blocksPerSubvector;
	extern __shared__ double shared[];
	double * Axs = shared;
	int * Ajs = (int*)&Axs[numcolsUP];

	if(blockIdx.x<dimUP*blocksPerSubvector){
		int sv = blockIdx.x/blocksPerSubvector;
		int id = (blockIdx.x%blocksPerSubvector)*threadsPerBlock+threadIdx.x;
		int gid = blockIdx.x*blockDim.x+threadIdx.x;

		if(threadIdx.x < numcolsUP){
			Axs[threadIdx.x] = AxUP[threadIdx.x*dimUP+sv];
			Ajs[threadIdx.x] = AjUP[threadIdx.x*dimUP+sv];
		}
		__syncthreads();

		if(id<dimDN){
            // H0up(x)Idn*x_vec  
            //sum_i(|sv>_up AxUP_{sv,i} up_<i|)(|id>_dn dn_<id|)(x_{i,id}|i>_up |id>_dn)
			for(int i=0;i<numcolsUP;i++){
				if(Axs[i]!=0) sum += Axs[i]*x[ Ajs[i]*dimDN+id ];
			}

            // Iup(x)H0dn*x_vec
            // sum_i (|sv>_up up_<sv|)(|id>_dn AxDN_{id,i} dn_<i|)(x_{sv,i}|sv>_up |i>_dn)
			for(int i=0;i<numcolsDN;i++){
				double Aij = AxDN[i*dimDN+id];
				int col = AjDN[i*dimDN+id];
				sum += Aij*x[sv*dimDN+col];
			}

			if(U!=0){
				int doubles = (stateUP[gid/dimDN]&stateDN[gid%dimDN])&dilution;
				double intE = U*count1(doubles);
				sum += intE*x[sv*dimDN+id];
			}
			
            y[gid] = sum; 
		}
	}
}

template<typename M>
void Hubbard::Hubbard_mv(thrust::device_vector<M> &x, thrust::device_vector<M> &y)
{
	double * ptr_AxUP = thrust::raw_pointer_cast(AxUP_D.data());
	int * ptr_AjUP = thrust::raw_pointer_cast(AjUP_D.data());
	double * ptr_AxDN = thrust::raw_pointer_cast(AxDN_D.data());
	int * ptr_AjDN = thrust::raw_pointer_cast(AjDN_D.data());
	int * ptr_stateUP = thrust::raw_pointer_cast(stateUP.data());
	int * ptr_stateDN = thrust::raw_pointer_cast(stateDN.data());
	M * ptr_x = thrust::raw_pointer_cast(x.data());
	M * ptr_y = thrust::raw_pointer_cast(y.data());

	int numBlocks = dimUP*BPS;
	int threadsPerBlock = dimDN/BPS;
	size_t shared = numcolsUP*sizeof(double)+numcolsUP*sizeof(int);
	Hubbard_mv_<M><<<numBlocks,threadsPerBlock,shared>>>(ptr_AxUP, ptr_AjUP, ptr_AxDN, ptr_AjDN, 
			ptr_stateUP, ptr_stateDN, ptr_x, ptr_y, dimUP, dimDN, numcolsUP, numcolsDN, 
			U, dilution, N, BPS);
	cudaDeviceSynchronize();
}

__global__ void write_(double * Hmat, const double * AxUP, const int * AjUP, const double * AxDN, const int * AjDN, 
		const int * stateUP, const int * stateDN, const unsigned long dimUP, const unsigned long dimDN, const int numcolsUP, const int numcolsDN, 
		const double U, const int dilution, const int blocksPerSubvector)
{
	int threadsPerBlock = dimDN/blocksPerSubvector;
	extern __shared__ double shared[];
	double * Axs = shared;
	int * Ajs = (int*)&Axs[numcolsUP];

	if(blockIdx.x<dimUP*blocksPerSubvector){
		int sv = blockIdx.x/blocksPerSubvector;
		int id = (blockIdx.x%blocksPerSubvector)*threadsPerBlock+threadIdx.x;
		int gid = blockIdx.x*blockDim.x+threadIdx.x;

		if(threadIdx.x < numcolsUP){
			Axs[threadIdx.x] = AxUP[threadIdx.x*dimUP+sv];
			Ajs[threadIdx.x] = AjUP[threadIdx.x*dimUP+sv];
		}
		__syncthreads();

		if(id<dimDN){
			for(int i=0;i<numcolsUP;i++){
				if(Axs[i]!=0) Hmat[ (sv*dimDN+id)*dimDN*dimUP+Ajs[i]*dimDN+id ] += Axs[i];
			}
			
			for(int i=0;i<numcolsDN;i++){
				double Aij = AxDN[i*dimDN+id];
				int col = AjDN[i*dimDN+id];
				if(Aij!=0) Hmat[ (sv*dimDN+id)*dimDN*dimUP+sv*dimDN+col ] += Aij;
			}

			if(U!=0){
				int doubles = (stateUP[gid/dimDN]&stateDN[gid%dimDN])&dilution;
				double intE = U*count1(doubles);
				Hmat[gid*dimUP*dimDN+gid] += intE;
			}
		}
	}
}

void Hubbard::Write(thrust::device_vector<double> &Hmat)
{
	double * ptr_Hmat = thrust::raw_pointer_cast(Hmat.data());
	double * ptr_AxUP = thrust::raw_pointer_cast(AxUP_D.data());
	int * ptr_AjUP = thrust::raw_pointer_cast(AjUP_D.data());
	double * ptr_AxDN = thrust::raw_pointer_cast(AxDN_D.data());
	int * ptr_AjDN = thrust::raw_pointer_cast(AjDN_D.data());
	int * ptr_stateUP = thrust::raw_pointer_cast(stateUP.data());
	int * ptr_stateDN = thrust::raw_pointer_cast(stateDN.data());

	int numBlocks = dimUP*BPS;
	int threadsPerBlock = dimDN/BPS;
	size_t shared = numcolsUP*sizeof(double)+numcolsUP*sizeof(int);
	write_<<<numBlocks,threadsPerBlock,shared>>>(ptr_Hmat, ptr_AxUP, ptr_AjUP, ptr_AxDN, ptr_AjDN, 
			ptr_stateUP, ptr_stateDN, dimUP, dimDN, numcolsUP, numcolsDN, U, dilution, BPS);
	cudaDeviceSynchronize();
}

void Hubbard::Write_pinned(double* Hmat)
{
	double * ptr_AxUP = thrust::raw_pointer_cast(AxUP_D.data());
	int * ptr_AjUP = thrust::raw_pointer_cast(AjUP_D.data());
	double * ptr_AxDN = thrust::raw_pointer_cast(AxDN_D.data());
	int * ptr_AjDN = thrust::raw_pointer_cast(AjDN_D.data());
	int * ptr_stateUP = thrust::raw_pointer_cast(stateUP.data());
	int * ptr_stateDN = thrust::raw_pointer_cast(stateDN.data());

	int numBlocks = dimUP*BPS;
	int threadsPerBlock = dimDN/BPS;
	size_t shared = numcolsUP*sizeof(double)+numcolsUP*sizeof(int);
	write_<<<numBlocks,threadsPerBlock,shared>>>(Hmat, ptr_AxUP, ptr_AjUP, ptr_AxDN, ptr_AjDN, 
			ptr_stateUP, ptr_stateDN, dimUP, dimDN, numcolsUP, numcolsDN, U, dilution, BPS);
	cudaDeviceSynchronize();
}

__global__ void initialize_(double * Hmat)
{
	int gid = blockIdx.x*blockDim.x+threadIdx.x;

	Hmat[gid] = 0.0;
}

void Hubbard::initialize(double* Hmat)
{
	unsigned int sup_size = dimUP*dimUP;
	unsigned int sub_size = dimDN*dimDN;
	int bps;
 
	for(int i=1;i<sub_size;i++){
		if(sub_size%i==0){
			if(sub_size/i<=1024){
				bps=i;
				break;
			}
		}
	}

	unsigned int numBlocks = sup_size*bps;
	unsigned int threadsPerBlock = sub_size/bps;
	initialize_<<<numBlocks,threadsPerBlock>>>(Hmat);
	cudaDeviceSynchronize();
}

///////////////////////
//---Thrust Linear---//
///////////////////////

template<typename L> 
struct Thrust_Lin
{
public:
	int dimUP,dimDN;

	Thrust_Lin(int dimUP, int dimDN):dimUP(dimUP),dimDN(dimDN)
	{}

	thrust::device_vector<L> ax(L s, const thrust::device_vector<L> &v);
	thrust::device_vector<thrust::complex<L>> ax(L s, const thrust::device_vector<thrust::complex<L>> &v);
	thrust::device_vector<thrust::complex<L>> ax(thrust::complex<L> s, const thrust::device_vector<thrust::complex<L>> &v);

	thrust::device_vector<L> axpy(L s, thrust::device_vector<L> &v1, const thrust::device_vector<L> &v2);
	thrust::device_vector<thrust::complex<L>> axpy(L s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2);
	thrust::device_vector<thrust::complex<L>> axpy(thrust::complex<L> s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2);
	thrust::device_vector<thrust::complex<L>> axpy(thrust::complex<L> s, thrust::device_vector<L> &v1, const thrust::device_vector<thrust::complex<L>> &v2);

	L thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<L> &v2);
	L thrust_Dotprod(thrust::device_vector<thrust::complex<L>> &v1, thrust::device_vector<thrust::complex<L>> &v2);
	thrust::complex<L> thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<thrust::complex<L>> &v2);

	void normalize(thrust::device_vector<L> &v);
	L norm(thrust::device_vector<L> &v);
};

template<typename L>
struct square
{
	__host__ __device__ L operator()(const L &x) const
	{ return x*x; }
};

template<typename L>
struct complex_prod //: public thrust::binary_function<thrust::complex<L>,thrust::complex<L>,thrust::complex<L>>
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
void Thrust_Lin<L>::normalize(thrust::device_vector<L> &v)
{
	square<L> unary_op;
	thrust::plus<L> binary_op;
	L init = 0.0;
	L sum = thrust::transform_reduce(v.begin(),v.end(),unary_op,init,binary_op);
	
	thrust::transform(v.begin(),v.end(),v.begin(),ax_<L>(1./sqrt(sum)));
}

template<typename L>
L Thrust_Lin<L>::norm(thrust::device_vector<L> &v)
{
	square<L> unary_op;
	thrust::plus<L> binary_op;
	L init = 0.0;
	L sum = thrust::transform_reduce(v.begin(),v.end(),unary_op,init,binary_op);

	return sum;
}

template<typename L>
thrust::device_vector<L> Thrust_Lin<L>::ax(L s, const thrust::device_vector<L> &v)
{
	thrust::device_vector<L> w(dimUP*dimDN);
	thrust::transform(v.begin(),v.end(),w.begin(),ax_<L>(s)); 
	
	return w;	
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::ax(L s, const thrust::device_vector<thrust::complex<L>> &v)
{
	thrust::device_vector<thrust::complex<L>> w(dimUP*dimDN);
	thrust::transform(v.begin(),v.end(),w.begin(),ax_<thrust::complex<L>>(thrust::complex<L>(s,0.))); 
	
	return w;	
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::ax(thrust::complex<L> s, const thrust::device_vector<thrust::complex<L>> &v)
{
	thrust::device_vector<thrust::complex<L>> w(dimUP*dimDN);
	thrust::transform(v.begin(),v.end(),w.begin(),ax_<thrust::complex<L>>(s)); 
	
	return w;	
}

template<typename L>
thrust::device_vector<L> Thrust_Lin<L>::axpy(L s, thrust::device_vector<L> &v1, const thrust::device_vector<L> &v2)
{
	thrust::device_vector<L> w(dimUP*dimDN);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<L,L>(s));

	return w;
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::axpy(L s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2)
{
	thrust::device_vector<thrust::complex<L>> w(dimUP*dimDN);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<thrust::complex<L>,thrust::complex<L>>(thrust::complex<L>(s,0.)));

	return w;
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::axpy(thrust::complex<L> s, thrust::device_vector<thrust::complex<L>> &v1, const thrust::device_vector<thrust::complex<L>> &v2)
{
	thrust::device_vector<thrust::complex<L>> w(dimUP*dimDN);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<thrust::complex<L>,thrust::complex<L>>(s));

	return w;
}

template<typename L>
thrust::device_vector<thrust::complex<L>> Thrust_Lin<L>::axpy(thrust::complex<L> s, thrust::device_vector<L> &v1, const thrust::device_vector<thrust::complex<L>> &v2)
{
	thrust::device_vector<thrust::complex<L>> w(dimUP*dimDN);
	thrust::transform(v1.begin(),v1.end(),v2.begin(),w.begin(),axpy_<L,thrust::complex<L>>(s));

	return w;
}

template<typename L>
L Thrust_Lin<L>::thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<L> &v2)
{
	return thrust::inner_product(v1.begin(),v1.end(),v2.begin(),0.0);
}

template<typename L>
L Thrust_Lin<L>::thrust_Dotprod(thrust::device_vector<thrust::complex<L>> &v1, thrust::device_vector<thrust::complex<L>> &v2)
{
	thrust::complex<L> w = thrust::inner_product(v1.begin(),v1.end(),v2.begin(),
			thrust::complex<L>(0.0,0.0),thrust::plus<thrust::complex<L>>(),complex_prod<L>());
	return w.real();
}

template<typename L>
thrust::complex<L> Thrust_Lin<L>::thrust_Dotprod(thrust::device_vector<L> &v1, thrust::device_vector<thrust::complex<L>> &v2)
{
	thrust::complex<L> w = thrust::inner_product(v1.begin(),v1.end(),v2.begin(),
			thrust::complex<L>(0.0,0.0),thrust::plus<thrust::complex<L>>(),real_comp_prod<L>());
	return w;
}

///////////////////////////
//---Lanczos algorithm---//
///////////////////////////

template<typename T>
struct Lanczos : public Thrust_Lin<T>
{
public:	
	thrust::host_vector<T> b;
	thrust::host_vector<T> a;
	int dimUP,dimDN;
		
	Lanczos(int dimUP, int dimDN) : dimUP(dimUP), dimDN(dimDN), Thrust_Lin<T>(dimUP,dimDN)
	{
		b.push_back(0.0);
	}

	void random_vector(thrust::device_vector<T> &f);
	
	void tridiagonal(Hubbard &Hub, thrust::device_vector<T> f);
	void eigen(Hubbard &Hub, thrust::device_vector<T> &f, T &eval, thrust::device_vector<T> &evec);
	
	void clear();

	~Lanczos()
	{
		b.clear(); thrust::host_vector<T>().swap(b);
		a.clear(); thrust::host_vector<T>().swap(a);
	}
};

struct GenRand
{
	unsigned int seed;

	__host__ __device__ GenRand()
	{
		//seed = time(NULL);
		seed = 235; 
	}

	__host__ __device__ double operator()(int idx)
	{
		thrust::default_random_engine randEng(seed);
		thrust::uniform_real_distribution<double> uniDist(-1.0,1.0);
		randEng.discard(idx);
		return uniDist(randEng);
	}
};

template<typename T>
void Lanczos<T>::clear()
{
	a.clear(); 
	b.clear(); 
	b.push_back(0.0);
}

template<typename T>
void Lanczos<T>::random_vector(thrust::device_vector<T> &f)
{
	int randN = f.size();
	thrust::transform(thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(randN),
			f.begin(),GenRand());
	this->normalize(f);
}

template<typename T> 
void Lanczos<T>::tridiagonal(Hubbard &Hub, thrust::device_vector<T> f)
{
    int N,maxN=200;
    thrust::device_vector<T> v0(f), v1(f); 
    b[0] = sqrt(this->norm(f));
    for (N=0;N<maxN;++N)
    {
        v1 = f;
        v1 = this->ax(1.0/b.back(),v1);
        Hub.Hubbard_mv(v1,f);
        if (N!=0) f = this->axpy(-b.back(),v0,f);
        a.push_back(this->thrust_Dotprod(v1,f));
        f = this->axpy(-a.back(),v1,f);
        b.push_back(sqrt(this->norm(f)));
        v0 = v1;
		//std::cout<<N<<"\t"<<a[N]<<"\t"<<b[N+1]<<std::endl;
    }
}

template<typename T>
void Lanczos<T>::eigen(Hubbard &Hub, thrust::device_vector<T> &f, T &eval, thrust::device_vector<T> &evec)
{
    int N,maxN=1000;
    thrust::device_vector<T> v0(f), v1(f); 
    evec=f; 
	thrust::host_vector<T> z, m;

    const double eps = std::numeric_limits<double>::epsilon();
    double anorm, tolconv = 0.01*sqrt(eps);

    b[0] = sqrt(this->norm(f));
    for (N=0;N<maxN;++N)
    {
        v1 = f;
        v1 = this->ax(1.0/b.back(),v1);
        Hub.Hubbard_mv(v1,f);
        if (N!=0) f = this->axpy(-b.back(),v0,f);
        a.push_back(this->thrust_Dotprod(v1,f));
        f = this->axpy(-a.back(),v1,f);
        b.push_back(sqrt(this->norm(f)));
        v0 = v1;
		//std::cout<<N<<"\t"<<a[N]<<"\t"<<b[N+1]<<std::endl;
        
        if (N==0) anorm = std::abs(a[0]+b[1]);
        else anorm = std::max(anorm,b[N]+std::abs(a[N]+b[N+1]));

        if (N%10==9)
        {
            z.resize((N+1)*(N+1)); m.resize(N+1);
			LAPACKf77::dsted(N+1,&a[0],&b[1],&m[0],&z[0]); // cpu lapack routin
			//std::cout<<"eval "<<N<<"th : "<<m[0]<<std::endl;
			eval = m[0];
			if(std::abs(b.back()*z[N])<tolconv*anorm && std::abs(z[0])>tolconv*anorm){ 
				break;
			}
        }
    }

    f = evec;
    thrust::fill(evec.begin(),evec.end(),0.0);
    for (int i=0;i<N+1;++i)
    {
        v1 = f;
        v1 = this->ax(1.0/b[i],v1);
        Hub.Hubbard_mv(v1,f);
        if (i!=0) f = this->axpy(-b[i],v0,f);
        f = this->axpy(-a[i],v1,f);
        v0 = v1;
        evec = this->axpy(z[i],v1,evec);
    }
    this->normalize(evec);
}

////////////////////////////////////
//------ Time evolution ----------//
////////////////////////////////////

template<typename C>
struct Krylov : public Thrust_Lin<C>
{
	thrust::device_vector<thrust::complex<C>> cHf;
	thrust::device_vector<thrust::complex<C>> cq;
	thrust::device_vector<thrust::complex<C>> cf_;
public:
	thrust::host_vector<C> b;
	thrust::host_vector<C> a;
	thrust::host_vector<C> m;
	thrust::host_vector<C> z;
	int dimUP,dimDN;

	Krylov(int dimUP, int dimDN):dimUP(dimUP),dimDN(dimDN),Thrust_Lin<C>(dimUP,dimDN)
	{
		cf_.resize(dimUP*dimDN,0.0);
		cHf.resize(dimUP*dimDN);
		cq.resize(dimUP*dimDN);
	
		b.push_back(0.0);
	}	
	
	void tridiagonal(Hubbard &Hub, thrust::device_vector<thrust::complex<C>> f);
	void Make_tridiagonal_eigen(Hubbard &Hub, thrust::device_vector<thrust::complex<C>> phi0);
	void Krylov_time_evol(Hubbard &Hub, thrust::device_vector<thrust::complex<C>> psi0, thrust::device_vector<thrust::complex<C>> &psi, double dt);

	void clear();

	~Krylov()
	{
		cHf.clear(); thrust::device_vector<thrust::complex<C>>().swap(cHf);	
		cq.clear(); thrust::device_vector<thrust::complex<C>>().swap(cq);
		cf_.clear(); thrust::device_vector<thrust::complex<C>>().swap(cf_);
		b.clear(); thrust::host_vector<C>().swap(b);
		a.clear(); thrust::host_vector<C>().swap(a);
		m.clear(); thrust::host_vector<C>().swap(m);
		z.clear(); thrust::host_vector<C>().swap(z);
	}
};

template<typename C> 
void Krylov<C>::tridiagonal(Hubbard &Hub, thrust::device_vector<thrust::complex<C>> f)
{
	for(int i=0;i<1000;i++){
		Hub.Hubbard_mv(f,cHf);
		cq = this->axpy(-b.back(),cf_,cHf);
		a.push_back(this->thrust_Dotprod(cq,f));
		//std::cout<<"a "<<i<<"th : "<<a[i]<<std::endl;
		cq = this->axpy(-a.back(),f,cq);
		b.push_back(sqrt(this->thrust_Dotprod(cq,cq)));
		//std::cout<<"b "<<i<<"th : "<<b[i]<<std::endl;	
		
		if(b.back()<1e-10) break;
		cf_ = f;
		f = this->ax(1.0/b.back(),cq);
	}

	cHf.clear(); cHf.resize(dimUP*dimDN);	
	cq.clear(); cq.resize(dimUP*dimDN);	
}

template<typename C>
void Krylov<C>::Make_tridiagonal_eigen(Hubbard &Hub, thrust::device_vector<thrust::complex<C>> phi0)
{
	tridiagonal(Hub,phi0);
	
	int Ntri = a.size();
	z.resize(Ntri*Ntri); m.resize(Ntri);

	LAPACK::dsted(Ntri,&a[0],&b[1],1,Ntri,&m[0],&z[0]);
}

template<typename C>
void Krylov<C>::Krylov_time_evol(Hubbard &Hub, thrust::device_vector<thrust::complex<C>> phi0, thrust::device_vector<thrust::complex<C>> &phi, double dt)
{
	// After Make_tridiagonal_eigen ... //

	int Ntri = a.size();
	thrust::complex<C> Tri = thrust::complex<C>(0,0);

	for(int i=0;i<Ntri;i++){
		for(int j=0;j<Ntri;j++){
			Tri += thrust::complex<C>(cos(m[j]*dt),-sin(m[j]*dt))*z[j*Ntri]*z[j*Ntri+i];
		} 
		//std::cout<<"Tri : "<<Tri<<std::endl;

		phi = this->axpy(Tri,phi0,phi);
		Tri = thrust::complex<C>(0,0);	

		Hub.Hubbard_mv(phi0,cHf);
		cq = this->axpy(-b[i],cf_,cHf);
		cq = this->axpy(-a[i],phi0,cq);
		cf_ = phi0;
		phi0 = this->ax(1./b[i+1],cq);	
	}
	
	cHf.clear(); cHf.resize(dimUP*dimDN);	
	cq.clear(); cq.resize(dimUP*dimDN);	
}

template<typename C>
void Krylov<C>::clear()
{
	a.clear();
	b.clear();
	b.push_back(0.0);

	m.clear();
	z.clear();
}

template<typename C>
struct tED : public Thrust_Lin<C>
{
	
public:
	int dimUP,dimDN;
	thrust::device_vector<C> W;
	thrust::device_vector<C> V;

	thrust::host_vector<C> h_W;

	tED(int dimUP, int dimDN):dimUP(dimUP),dimDN(dimDN),Thrust_Lin<C>(dimUP,dimDN)
	{
		W.resize(dimUP*dimDN);
		h_W.resize(dimUP*dimDN);
	}

	void Make_eigen(const thrust::device_vector<C> &Hmat);
	void tED_time_evol(thrust::device_vector<thrust::complex<C>> phi0,thrust::device_vector<thrust::complex<C>> &phi, double dt);

	void Make_eigen_pinned(C* Hmat);
	void tED_time_evol_pinned(C* Hmat, thrust::device_vector<thrust::complex<C>> phi0,thrust::device_vector<thrust::complex<C>> &phi, double dt);

	void clear();
};

template<typename C>
void tED<C>::Make_eigen(const thrust::device_vector<C> &Hmat)
{
	// After Hubbard Write ..

	V = Hmat;	
	
	C *ptr_W = thrust::raw_pointer_cast(W.data()); 
	C *ptr_V = thrust::raw_pointer_cast(V.data()); 

	cusolverDsyevd(dimUP*dimDN,ptr_V,ptr_W);
}

template<typename C>
void tED<C>::Make_eigen_pinned(C* Hmat)
{
	// After Hubbard Write ..
	
	cusolverDsyevdMG(dimUP*dimDN,Hmat,&h_W[0]);
	//cusolverDsyevd(dimUP*dimDN,Hmat,thrust::raw_pointer_cast(W.data()));
}

template<typename C>
void tED<C>::tED_time_evol(thrust::device_vector<thrust::complex<C>> phi0, thrust::device_vector<thrust::complex<C>> &phi, double dt)
{
	thrust::complex<C> Cn, braket;
	thrust::device_vector<C> n_bra(dimUP*dimDN);

	for(int i=0;i<dimUP*dimDN;i++){
		thrust::copy(V.begin()+i*dimUP*dimDN,V.begin()+(i+1)*dimUP*dimDN,n_bra.begin());
		braket = this->thrust_Dotprod(n_bra,phi0);

		Cn = thrust::complex<C>(cos(W[i]*dt),-sin(W[i]*dt))*braket;
		phi = this->axpy(Cn,n_bra,phi);
	}			
}

template<typename C>
void tED<C>::tED_time_evol_pinned(C* Hmat, thrust::device_vector<thrust::complex<C>> phi0, thrust::device_vector<thrust::complex<C>> &phi, double dt)
{
	thrust::complex<C> Cn, braket;
	thrust::device_vector<C> n_bra(dimUP*dimDN);

	for(int i=0;i<dimUP*dimDN;i++){
		thrust::copy(Hmat+i*dimUP*dimDN,Hmat+(i+1)*dimUP*dimDN,n_bra.begin());
		braket = this->thrust_Dotprod(n_bra,phi0);

		Cn = thrust::complex<C>(cos(h_W[i]*dt),-sin(h_W[i]*dt))*braket;
		phi = this->axpy(Cn,n_bra,phi);
	}			
}

template<typename C>
void tED<C>::clear()
{
	W.clear(); W.resize(dimUP*dimDN);
	h_W.clear(); h_W.resize(dimUP*dimDN);
	V.clear();
}

////////////////////////////////////
//------Calculate Correlator------//
////////////////////////////////////

template<typename T>
void correlator_continued_fraction(Lanczos<T> &lanctri, Hubbard &H, const int &nw, 
	const thrust::host_vector< thrust::complex<T> > &w, const T &energy, 
	thrust::device_vector<T> &phi, thrust::host_vector< thrust::complex<T> > &g)
{
	T norm = lanctri.norm(phi);
	lanctri.normalize(phi);
	lanctri.tridiagonal(H,phi);

	int n = (lanctri.a).size(); 
	for(int idx=0;idx<nw;idx++){
		thrust::complex<T> gw=1.0/(w[idx]+energy-lanctri.a[n-1]);
		for(int i=n-2;i>=0;i--) gw=1.0/(w[idx]+energy-lanctri.a[i]-gw*lanctri.b[i+1]*lanctri.b[i+1]);
		g[idx] = norm*gw;
	}
}

template<typename T>
void compute_sigma_omega(Lanczos<T> &lanctri, Hubbard &H, Hubbard &O, 
	const T &E, thrust::device_vector<T> &evec, 
	const thrust::host_vector< thrust::complex<T> > &w, 
	thrust::host_vector< thrust::complex<T> > &Lambda)
{
	lanctri.clear();
	thrust::device_vector<T> phi(lanctri.dimUP*lanctri.dimDN);
	O.Hubbard_mv(evec,phi);
	correlator_continued_fraction(lanctri,H,w.size(),w,E,phi,Lambda);

	phi.clear(); thrust::device_vector<double>().swap(phi);
}

#endif









