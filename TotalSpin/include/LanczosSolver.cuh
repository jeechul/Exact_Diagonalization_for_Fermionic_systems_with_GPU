#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include "thrust_util.cuh"
#include "../../MBL_test/include/MAGMA_LAPACK.h"

#define CUDAERR(x){\
	if(x != cudaSuccess) {\
		std::cout << "Error! : #" << __LINE__ << std::endl;\
	}\
}\

///////////////////////////
//---Lanczos algorithm---//
///////////////////////////

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

struct GenComplexRand
{
	unsigned int seed;

	__host__ __device__ GenComplexRand()
	{
		//seed = time(NULL);
		seed = 235; 
	}

	__host__ __device__ thrust::complex<double> operator()(int idx)
	{
		thrust::default_random_engine randEng(seed);
		thrust::uniform_real_distribution<double> uniDist(-1.0,1.0);
		randEng.discard(idx);
		return thrust::complex<double>(uniDist(randEng),uniDist(randEng));
	}
};

template<typename T, typename ModelType>
struct Lanczos : public THRUST::Thrust_Lin<T>
{
private:	
	thrust::host_vector<T> b;
	thrust::host_vector<T> a;
public:		
    Lanczos() : THRUST::Thrust_Lin<T>()
	{
		b.push_back(0.0);
	}

	void random_vector(thrust::device_vector<T> &f);	
	void random_vector(thrust::device_vector<thrust::complex<T>> &f);	
	void tridiagonal(ModelType &Hub, thrust::device_vector<T> f);
	void eigen(ModelType &Hub, thrust::device_vector<T> &f, T &eval, thrust::device_vector<T> &evec);
	void eigen(ModelType &Hub, thrust::device_vector<thrust::complex<T>> &f, T &eval, thrust::device_vector<thrust::complex<T>> &evec);
	void clear();

	~Lanczos()
	{
		b.clear(); thrust::host_vector<T>().swap(b);
		a.clear(); thrust::host_vector<T>().swap(a);
	}
};

template<typename T, typename ModelType>
void Lanczos<T, ModelType>::clear()
{
	a.clear(); 
	b.clear(); 
	b.push_back(0.0);
}

template<typename T, typename ModelType>
void Lanczos<T, ModelType>::random_vector(thrust::device_vector<T> &f)
{
	int randN = f.size();
	thrust::transform(thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(randN),
			f.begin(),GenRand());
	this->normalize(f);
}

template<typename T, typename ModelType>
void Lanczos<T, ModelType>::random_vector(thrust::device_vector<thrust::complex<T>> &f)
{
	int randN = f.size();
	thrust::transform(thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(randN),
			f.begin(),GenComplexRand());
	this->normalize(f);
}

template<typename T, typename ModelType> 
void Lanczos<T, ModelType>::tridiagonal(ModelType &Hub, thrust::device_vector<T> f)
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

template<typename T, typename ModelType>
void Lanczos<T, ModelType>::eigen(ModelType &Hub, thrust::device_vector<T> &f, T &eval, thrust::device_vector<T> &evec)
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

template<typename T, typename ModelType>
void Lanczos<T, ModelType>::eigen(ModelType &Hub, thrust::device_vector<thrust::complex<T>> &f, T &eval, thrust::device_vector<thrust::complex<T>> &evec)
{
    int N,maxN=1000;
    thrust::device_vector<thrust::complex<T>> v0(f), v1(f); 
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
        a.push_back((this->thrust_Dotprod(v1,f)).real());
        //std::cout << this->thrust_Dotprod(v1,f) << std::endl;
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

