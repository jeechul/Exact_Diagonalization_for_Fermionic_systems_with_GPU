#ifndef __EE_MATRIX_GPU__
#define __EE_MATRIX_GPU__

#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <assert.h>
#include "fermi_operator.h"
#include "entangle_entropy.h"

using comTh=thrust::complex<double>;

__host__ __device__ void idx2ABidx(int idx, int *ABidx, int *state, int *addressA, int *addressB, int A, int B)
{
	// ABidx = {idx of A state, idx of B state} // 

	int stateAB = state[idx];
	int Astate = stateAB&A, Bstate = stateAB&B;
	ABidx[0] = addressA[Astate];
	ABidx[1] = addressB[Bstate];	
}

template<typename T>
struct EEmatHubAB_GPU
{
public:
	EEmatAB<T> EEmatUP,EEmatDN;
	int rowD, colD, BPS, Aup, Adn, Bup, Bdn;

	EEmatHubAB_GPU(int *Asite, int Asize, const H_TBq &HqUP, const H_TBq &HqDN):EEmatUP(Asite,Asize,HqUP),EEmatDN(Asite,Asize,HqDN)
	{
		rowD = EEmatUP.dimA*EEmatDN.dimA;
		colD = EEmatUP.dimB*EEmatDN.dimB;	
		Aup = EEmatUP.A; Adn = EEmatDN.A;
		Bup = EEmatUP.B; Bdn = EEmatDN.B;

		int dimDN = EEmatDN.dimAB;

		for(int i=1;i<dimDN;i++){
			if(dimDN%i==0){
				if(dimDN/i<=1024){
					BPS=i;
					break;
				}
			}
		}
	}

	void Make_Write(thrust::device_vector<T> &psi, thrust::device_vector<T> &Hmat);
	void Make_Write_pinned(thrust::device_vector<T> &psi, T* Hmat);

	void initialize(T* Hmat);
};

template<typename T>
__global__ void make_write_(T *psi, T *Hmat, int *stateUP, int *stateDN, int *addressAUP, 
	int *addressADN, int *addressBUP, int *addressBDN, int dimDN, int dimUP, 
	int dimADN, int dimBDN, int dimBUP, int Aup, int Adn, int Bup, int Bdn, int BPS)
{
	int threadsPerBlock = dimDN/BPS;
	int UPABid[2], DNABid[2];

	if(blockIdx.x<dimUP*BPS){
		int idxUP = blockIdx.x/BPS;
		int idxDN = (blockIdx.x%BPS)*threadsPerBlock+threadIdx.x;
	
		idx2ABidx(idxUP,UPABid,stateUP,addressAUP,addressBUP,Aup,Bup);
		idx2ABidx(idxDN,DNABid,stateDN,addressADN,addressBDN,Adn,Bdn);
	
		T Mx = psi[idxUP*dimDN+idxDN];
		int row = UPABid[0]*dimADN+DNABid[0];
		int col = UPABid[1]*dimBDN+DNABid[1];

		Hmat[row*dimBDN*dimBUP+col] = Mx;
	}
} 

template<typename T>
__global__ void initialize_(T *Hmat)
{
	int gid = blockIdx.x*blockDim.x+threadIdx.x;	

	Hmat[gid] = 0.0;
}

template<typename T>
void EEmatHubAB_GPU<T>::Make_Write(thrust::device_vector<T> &psi, thrust::device_vector<T> &Hmat)
{
	int dimDN = EEmatDN.dimAB, dimUP = EEmatUP.dimAB;
	int dimADN = EEmatDN.dimA, dimBDN = EEmatDN.dimB;
	int dimBUP = EEmatUP.dimB;

	thrust::device_vector<int> stateUP = EEmatUP.state, stateDN = EEmatDN.state;  
	thrust::device_vector<int> addressAUP = EEmatUP.addressA, addressADN = EEmatDN.addressA;  
	thrust::device_vector<int> addressBUP = EEmatUP.addressB, addressBDN = EEmatDN.addressB;

	T *ptr_psi = thrust::raw_pointer_cast(psi.data());  
	T *ptr_Hmat = thrust::raw_pointer_cast(Hmat.data()); 
 
	int *ptr_stateUP = thrust::raw_pointer_cast(stateUP.data());  
	int *ptr_stateDN = thrust::raw_pointer_cast(stateDN.data());  
	int *ptr_addressAUP = thrust::raw_pointer_cast(addressAUP.data());  
	int *ptr_addressADN = thrust::raw_pointer_cast(addressADN.data());  
	int *ptr_addressBUP = thrust::raw_pointer_cast(addressBUP.data());  
	int *ptr_addressBDN = thrust::raw_pointer_cast(addressBDN.data());  

	int numBlocks = dimUP*BPS;
	int threadsPerBlock = dimDN/BPS;
	make_write_<T><<<numBlocks,threadsPerBlock>>>(ptr_psi,ptr_Hmat,ptr_stateUP,ptr_stateDN,
		ptr_addressAUP,ptr_addressADN,ptr_addressBUP,ptr_addressBDN,dimDN,dimUP,
		dimADN,dimBDN,dimBUP,Aup,Adn,Bup,Bdn,BPS); 		
}

template<typename T>
void EEmatHubAB_GPU<T>::Make_Write_pinned(thrust::device_vector<T> &psi, T* Hmat)
{
	int dimDN = EEmatDN.dimAB, dimUP = EEmatUP.dimAB;
	int dimADN = EEmatDN.dimA, dimBDN = EEmatDN.dimB;
	int dimBUP = EEmatUP.dimB;

	thrust::device_vector<int> stateUP = EEmatUP.state, stateDN = EEmatDN.state;  
	thrust::device_vector<int> addressAUP = EEmatUP.addressA, addressADN = EEmatDN.addressA;  
	thrust::device_vector<int> addressBUP = EEmatUP.addressB, addressBDN = EEmatDN.addressB;

	T *ptr_psi = thrust::raw_pointer_cast(psi.data());  
 
	int *ptr_stateUP = thrust::raw_pointer_cast(stateUP.data());  
	int *ptr_stateDN = thrust::raw_pointer_cast(stateDN.data());  
	int *ptr_addressAUP = thrust::raw_pointer_cast(addressAUP.data());  
	int *ptr_addressADN = thrust::raw_pointer_cast(addressADN.data());  
	int *ptr_addressBUP = thrust::raw_pointer_cast(addressBUP.data());  
	int *ptr_addressBDN = thrust::raw_pointer_cast(addressBDN.data());  

	int numBlocks = dimUP*BPS;
	int threadsPerBlock = dimDN/BPS;
	make_write_<T><<<numBlocks,threadsPerBlock>>>(ptr_psi,Hmat,ptr_stateUP,ptr_stateDN,
		ptr_addressAUP,ptr_addressADN,ptr_addressBUP,ptr_addressBDN,dimDN,dimUP,
		dimADN,dimBDN,dimBUP,Aup,Adn,Bup,Bdn,BPS); 	
	cudaDeviceSynchronize();	
}

template<typename T>
void EEmatHubAB_GPU<T>::initialize(T* Hmat)
{
	int dimAUP = EEmatUP.dimA, dimADN = EEmatDN.dimA, dimBUP = EEmatUP.dimB, dimBDN = EEmatDN.dimB, bps;
	unsigned int sub_size = dimADN*dimBDN;
	unsigned int sup_size = dimAUP*dimBUP;

	for(int i=1;i<sub_size;i++){
		if(sub_size%i==0){
			if(sub_size/i<=1024){
				bps=i;
				break;
			}
		}
	}
	
	int numBlocks = sup_size*bps;
	int threadsPerBlock = sub_size/bps;
	initialize_<T><<<numBlocks,threadsPerBlock>>>(Hmat);
	cudaDeviceSynchronize();	
}

template<typename T>
struct EEmatHubSpin_GPU
{
public:
	H_TBq HqUP, HqDN;
	int rowD, colD, BPS;

	EEmatHubSpin_GPU(const H_TBq &HqUP, const H_TBq &HqDN):HqUP(HqUP), HqDN(HqDN)
	{
		colD = HqUP.count; rowD = HqDN.count;	
	
		for(int i=1;i<rowD;i++){
			if(rowD%i==0){
				if(rowD/i<=1024){
					BPS=i;
					break;
				}
			}
		}
	}	

	void Make_Write(thrust::device_vector<T> &psi, thrust::device_vector<T> &Hmat);
	void Make_Write_pinned(thrust::device_vector<T> &psi, T* Hmat);

	void initialize(T* Hmat);
};

template<typename T>
__global__ void make_write_spin(T *psi, T *Hmat, int colD, int rowD, int BPS)
{
	int threadsPerBlock = rowD/BPS;
	
	if(threadIdx.x<colD*BPS){
		int sv = blockIdx.x/BPS;
		int id = (blockIdx.x%BPS)*threadsPerBlock+threadIdx.x;

		Hmat[sv+id*rowD] = psi[sv*rowD+id];
	}		
}

template<typename T>
__global__ void initialize_spin(T *Hmat, int colD, int rowD, int BPS)
{
	int threadsPerBlock = rowD/BPS;
	
	if(threadIdx.x<colD*BPS){
		int sv = blockIdx.x/BPS;
		int id = (blockIdx.x%BPS)*threadsPerBlock+threadIdx.x;

		Hmat[sv+id*rowD] = 0.0;
	}		
}

template<typename T>
void EEmatHubSpin_GPU<T>::Make_Write(thrust::device_vector<T> &psi, thrust::device_vector<T> &Hmat)
{
	T *ptr_psi = thrust::raw_pointer_cast(psi.data());
	T *ptr_Hmat = thrust::raw_pointer_cast(Hmat.data());

	int numBlocks = colD*BPS;
	int threadsPerBlock = rowD/BPS;
	make_write_spin<T><<<numBlocks,threadsPerBlock>>>(ptr_psi,ptr_Hmat,colD,rowD,BPS);
}

template<typename T>
void EEmatHubSpin_GPU<T>::Make_Write_pinned(thrust::device_vector<T> &psi, T* Hmat)
{
	T *ptr_psi = thrust::raw_pointer_cast(psi.data());

	int numBlocks = colD*BPS;
	int threadsPerBlock = rowD/BPS;
	make_write_spin<T><<<numBlocks,threadsPerBlock>>>(ptr_psi,Hmat,colD,rowD,BPS);
	cudaDeviceSynchronize();	
}

template<typename T>
void EEmatHubSpin_GPU<T>::initialize(T* Hmat)
{
	int numBlocks = colD*BPS;
	int threadsPerBlock = rowD/BPS;
	initialize_spin<T><<<numBlocks,threadsPerBlock>>>(Hmat,colD,rowD,BPS);
	cudaDeviceSynchronize();	
}

#endif











