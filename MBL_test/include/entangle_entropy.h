#ifndef __EE_MATRIX__
#define __EE_MATRIX__

#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <assert.h>
#include "fermi_operator.h"

using comTh=thrust::complex<double>;

template<typename T>
class EEmatAB
{
	int Asize,Q;
	int *Asite;

	void LabelingAB()
	{
		// making stateA,B addressA,B //
		int stateAB,Astate,Bstate;

		for(int n=0;n<dimAB;n++){
			stateAB = state[n];
			Astate = stateAB&A;
			Bstate = stateAB&B;

			if(addressA[Astate]==-1){
				stateA[dimA] = Astate;
				addressA[Astate] = dimA;
				dimA++;
			}

			if(addressB[Bstate]==-1){
				stateB[dimB] = Bstate;
				addressB[Bstate] = dimB;
				dimB++;
			}
		}
	}
public:
	int dimA,dimB,dimAB,A,B; // A,B : 1-bit at sites of A,B.;

	thrust::host_vector<int> state,address;
	thrust::host_vector<int> stateA,addressA;
	thrust::host_vector<int> stateB,addressB;
	thrust::host_vector<T> Mx;
	thrust::host_vector<int> Mrow,Mcol;
	
	EEmatAB(int *Asite, int Asize, const H_TBq &Hq):Asite(Asite),Asize(Asize),dimA(0),dimB(0)
	{
		state = Hq.state;
		address = Hq.address;
		Q = Hq.Q;
		dimAB = Hq.count;
		int dim = Hq.dim;
		int N = Hq.N;

		A = 0;
		for(int i=0;i<Asize;i++) A += 1<<Asite[i];
		B = ((1<<N)-1)^A;

		stateA.resize(dim); addressA.resize(dim,-1);
		stateB.resize(dim); addressB.resize(dim,-1);

		LabelingAB(); 
	}

	void Make(const thrust::device_vector<T> &psi)
	{
		int stateAB,Astate,Bstate;
		for(int n=0;n<dimAB;n++){
			stateAB = state[n];
			Astate = stateAB&A;
			Bstate = stateAB&B;

			Mx.push_back(psi[n]);
			Mrow.push_back(addressA[Astate]);
			Mcol.push_back(addressB[Bstate]);
		}			
	}

	void Write(thrust::device_vector<T>& Mat)
	{
		int row,col;

		for(int i=0;i<dimAB;i++){
			row = Mrow[i]; col = Mcol[i];
			Mat[row*dimB+col] = Mx[i];
		}		
	}

	void idx2ABidx(int idx, int *ABidx)
	{
		// ABidx = {idx of A state, idx of B state} // 

		int stateAB = state[idx];
		int Astate = stateAB&A, Bstate = stateAB&B;
		ABidx[0] = addressA[Astate];
		ABidx[1] = addressB[Bstate];			
	}

	void Make_Write(thrust::device_vector<T> &psi, thrust::device_vector<T>& Mat)
	{
		int ABidx[2];

		for(int n=0;n<dimAB;n++){
			idx2ABidx(n,ABidx);

			int row = ABidx[0];
			int col = ABidx[1];

			Mat[row*dimB+col] = psi[n];
		}
	}

	void clear()
	{
		Mx.clear();
		Mrow.clear();
		Mcol.clear();
	}		
};

template<typename T>
class EEmatHubAB
{
public:
	EEmatAB<T> EEmatUP,EEmatDN;
	thrust::host_vector<T> Hx;
	thrust::host_vector<int> Hrow;
	thrust::host_vector<int> Hcol;
	int rowD, colD;

	EEmatHubAB(int *Asite, int Asize, const H_TBq &HqUP, const H_TBq &HqDN):EEmatUP(Asite,Asize,HqUP), EEmatDN(Asite,Asize,HqDN)
	{
		rowD = EEmatUP.dimA*EEmatDN.dimA;
		colD = EEmatUP.dimB*EEmatDN.dimB;
	}

	void Make(const thrust::device_vector<T> &psi)
	{
		int dimDN = EEmatDN.dimAB, dimUP = EEmatUP.dimAB;
		int dimADN = EEmatDN.dimA, dimBDN = EEmatDN.dimB;
		int idxUP[2], idxDN[2];

		for(int i=0;i<dimUP;i++){
			EEmatUP.idx2ABidx(i,idxUP);

			for(int j=0;j<dimDN;j++){
				EEmatDN.idx2ABidx(j,idxDN);		

				Hx.push_back(psi[i*dimDN+j]);
				int row = idxUP[0]*dimADN+idxDN[0];
				int col = idxUP[1]*dimBDN+idxDN[1];
				Hrow.push_back(row);
				Hcol.push_back(col);
			}
		}
	}

	void Write(thrust::device_vector<T> &Hmat)
	{
		int row,col,dimUP=EEmatUP.dimAB,dimDN=EEmatDN.dimAB;
		int dimBDN = EEmatDN.dimB, dimBUP = EEmatUP.dimB;

		for(int i=0;i<dimUP*dimDN;i++){
			row = Hrow[i]; col = Hcol[i];
			Hmat[row*dimBDN*dimBUP+col] = Hx[i];
		}
	}

	void Make_Write(const thrust::device_vector<T> &psi, thrust::device_vector<T> &Hmat)
	{
		int dimDN = EEmatDN.dimAB, dimUP = EEmatUP.dimAB;
		int dimADN = EEmatDN.dimA, dimBDN = EEmatDN.dimB;
		int dimBUP = EEmatUP.dimB;
		int idxUP[2],idxDN[2];
	
		for(int i=0;i<dimUP;i++){
			EEmatUP.idx2ABidx(i,idxUP);

			for(int j=0;j<dimDN;j++){
				EEmatDN.idx2ABidx(j,idxDN);		

				double Mx = psi[i*dimDN+j];
				int row = idxUP[0]*dimADN+idxDN[0];
				int col = idxUP[1]*dimBDN+idxDN[1];

				Hmat[row*dimBDN*dimBUP+col] = Mx;
			}
		}
	}

	void clear()
	{
		Hx.clear();
		Hrow.clear();
		Hcol.clear();
	}
};

template<typename T>
class EEmatHubSpin
{
public:
	H_TBq HqUP,HqDN;
	int rowD, colD;

	EEmatHubSpin(const H_TBq &HqUP, const H_TBq &HqDN):HqUP(HqUP),HqDN(HqDN)
	{ colD = HqUP.count; rowD = HqDN.count; }

	void Make_Write(const thrust::device_vector<T> &psi, thrust::device_vector<T> &Hmat)
	{
		int dimUP = colD, dimDN = rowD;

		for(int i=0;i<dimUP;i++){
			for(int j=0;j<dimDN;j++){
				Hmat[i+j*dimDN] = psi[i*dimDN+j];
			}
		}		
	}	
};

//--------- Cuda SVD -------------//

void cusolverDsvd(int m, int n, double *d_A, double *S)
{
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	const int lda = m;

	//double *d_A = NULL;
        double *d_S = NULL;
        double *d_U = NULL;
        double *d_VT = NULL;
        int *devInfo = NULL;
        double *d_work = NULL;
        double *d_rwork = NULL;

	int lwork = 0;
        int info_gpu = 0;
	int S_dim = std::min(m,n);

	// step 1: create cusolverDn/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	// step 2: copy A and B to device
	//cudaMalloc ((void**)&d_A  , sizeof(double)*lda*n);
	cudaMalloc ((void**)&d_S  , sizeof(double)*S_dim);
	cudaMalloc ((void**)&d_U  , sizeof(double)*lda*m);
	cudaMalloc ((void**)&d_VT  , sizeof(double)*lda*n);
	cudaMalloc ((void**)&devInfo, sizeof(int));

	//cudaMemcpy(d_A, A, sizeof(double)*lda*n, cudaMemcpyHostToDevice);

	// step 3: query working space of SVD
	cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH,m,n,&lwork);
	cudaMalloc((void**)&d_work , sizeof(double)*lwork);	
	assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

	// step 4: compute SVD
	signed char jobu = 'N'; // all m columns of U
        signed char jobvt = 'N'; // all n columns of VT
        cusolver_status = cusolverDnDgesvd (cusolverH,jobu,jobvt,m,n,d_A,lda,d_S,d_U,
        lda,  // ldu
        d_VT,
        lda, // ldvt,
        d_work,lwork,d_rwork,devInfo);
        cudaDeviceSynchronize();
	assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cudaMemcpy(S , d_S , sizeof(double)*n    , cudaMemcpyDeviceToHost);
	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

	cudaFree(devInfo);
	cudaFree(d_work);
	cudaFree(d_rwork);
	cudaFree(d_S);
	cudaFree(d_VT);
	cudaFree(d_U);
	cusolverDnDestroy(cusolverH);
	//cudaDeviceReset();
}

void cusolverZsvd(int m, int n, cuDoubleComplex *d_A, double *S)
{
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	const int lda = m;

	//cuDoubleComplex *d_A = NULL;
        double *d_S = NULL;
        cuDoubleComplex *d_U = NULL;
        cuDoubleComplex *d_VT = NULL;
        int *devInfo = NULL;
        cuDoubleComplex *d_work = NULL;
        double *d_rwork = NULL;

	int lwork = 0;
        int info_gpu = 0;
	int S_dim = std::min(m,n);

	// step 1: create cusolverDn/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	
	// step 2: copy A and B to device
	//cudaMalloc ((void**)&d_A  , sizeof(double)*lda*n);
	cudaMalloc ((void**)&d_S  , sizeof(double)*S_dim);
	cudaMalloc ((void**)&d_U  , sizeof(cuDoubleComplex)*lda*m);
	cudaMalloc ((void**)&d_VT  , sizeof(cuDoubleComplex)*lda*n);
	cudaMalloc ((void**)&devInfo, sizeof(int));

	//cudaMemcpy(d_A, A, sizeof(double)*lda*n, cudaMemcpyHostToDevice);
	
	// step 3: query working space of SVD
	cusolver_status = cusolverDnZgesvd_bufferSize(cusolverH,m,n,&lwork);
	cudaMalloc((void**)&d_work , sizeof(cuDoubleComplex)*lwork);	
	assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

	// step 4: compute SVD
	signed char jobu = 'N'; // all m columns of U
        signed char jobvt = 'N'; // all n columns of VT
        cusolver_status = cusolverDnZgesvd (cusolverH,jobu,jobvt,m,n,d_A,lda,d_S,d_U,
        lda,  // ldu
        d_VT,
        lda, // ldvt,
        d_work,lwork,d_rwork,devInfo);
        cudaDeviceSynchronize();
	assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cudaMemcpy(S , d_S , sizeof(double)*n    , cudaMemcpyDeviceToHost);
	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

	cudaFree(devInfo);
	cudaFree(d_work);
	cudaFree(d_rwork);
	cudaFree(d_S);
	cudaFree(d_VT);
	cudaFree(d_U);
	cusolverDnDestroy(cusolverH);
	//cudaDeviceReset();
}

#endif







