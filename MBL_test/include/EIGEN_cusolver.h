#ifndef __EIGEN__
#define __EIGEN__

#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <cusolverMg.h>
#include <assert.h>
//#include "fermi_operator.h"
#include "util.cuh"

void cusolverDsyevd(const int m, double *d_A, double *d_W)
{
	cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	const int lda = m;
	//double *d_A = NULL;
    //double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork = 0;

    int info_gpu = 0;

	// step 1: create cusolver/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	//cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    //cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaMalloc ((void**)&devInfo, sizeof(int));

	//cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	
	// step 3: query working space of syevd
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and vectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH,jobz,uplo,m,d_A,lda,d_W,&lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);

	// step 4: compute spectrum
	cusolver_status = cusolverDnDsyevd(cusolverH,jobz,uplo,m,d_A,lda,d_W,d_work,lwork,devInfo);
    cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	//cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	//cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

	//cudaFree(d_W);
	//cudaFree(d_A);
    cudaFree(devInfo);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
}

void cusolverZheevd(const int m, cuDoubleComplex *d_A, double *d_W)
{
	cusolverDnHandle_t cusolverH = NULL;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	
	const int lda = m;
	//double *d_A = NULL;
        //double *d_W = NULL;
        int *devInfo = NULL;
        cuDoubleComplex *d_work = NULL;
        int lwork = 0;

        int info_gpu = 0;

	// step 1: create cusolver/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// step 2: copy A and B to device
	//cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
        //cudaMalloc ((void**)&d_W, sizeof(double) * m);
        cudaMalloc ((void**)&devInfo, sizeof(int));

	//cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
	
	// step 3: query working space of zheevd
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and vectors.
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        cusolver_status = cusolverDnZheevd_bufferSize(cusolverH,jobz,uplo,m,d_A,lda,d_W,&lwork);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);

	// step 4: compute spectrum
	cusolver_status = cusolverDnZheevd(cusolverH,jobz,uplo,m,d_A,lda,d_W,d_work,lwork,devInfo);
        cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	//cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
	//cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
        cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

	//cudaFree(d_W);
	//cudaFree(d_A);
        cudaFree(devInfo);
        cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
}

void cusolverDsyevdj(const int m, double *d_A, double *d_W)
{
	cusolverDnHandle_t cusolverH = NULL;
 	cudaStream_t stream = NULL;
 	syevjInfo_t syevj_params = NULL;
 	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	const int lda = m;
	//double *d_A = NULL;
        //double *d_W = NULL;
        int *devInfo = NULL;
        double *d_work = NULL;
        int lwork = 0;

        int info_gpu = 0;

	/* configuration of syevj */
 	const double tol = 1.e-7;
 	const int max_sweeps = 15;
 	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
 	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	/* step 1: create cusolver handle, bind a stream */
 	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of syevj */
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXsyevjSetTolerance(
	syevj_params,
	tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXsyevjSetMaxSweeps(
	syevj_params,
	max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMalloc ((void**)&devInfo, sizeof(int));

	/* step 4: query working space of syevj */
	status = cusolverDnDsyevj_bufferSize(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	&lwork,
	syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	
	/* step 5: compute eigen-pair */
	status = cusolverDnDsyevj(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	d_work,
	lwork,
	devInfo,
	syevj_params);
	cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

        cudaFree(devInfo);
        cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
	cudaStreamDestroy(stream);
	cusolverDnDestroySyevjInfo(syevj_params);
}

void cusolverZheevdj(const int m, cuDoubleComplex *d_A, double *d_W)
{
	cusolverDnHandle_t cusolverH = NULL;
 	cudaStream_t stream = NULL;
 	syevjInfo_t syevj_params = NULL;
 	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	const int lda = m;
	//double *d_A = NULL;
        //double *d_W = NULL;
        int *devInfo = NULL;
        cuDoubleComplex *d_work = NULL;
        int lwork = 0;

        int info_gpu = 0;

	/* configuration of syevj */
 	const double tol = 1.e-7;
 	const int max_sweeps = 15;
 	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
 	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	/* step 1: create cusolver handle, bind a stream */
 	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of syevj */
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXsyevjSetTolerance(
	syevj_params,
	tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXsyevjSetMaxSweeps(
	syevj_params,
	max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMalloc ((void**)&devInfo, sizeof(int));

	/* step 4: query working space of syevj */
	status = cusolverDnZheevj_bufferSize(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	&lwork,
	syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
	
	/* step 5: compute eigen-pair */
	status = cusolverDnZheevj(
	cusolverH,
	jobz,
	uplo,
	m,
	d_A,
	lda,
	d_W,
	d_work,
	lwork,
	devInfo,
	syevj_params);
	cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(0 == info_gpu);

        cudaFree(devInfo);
        cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
	cudaStreamDestroy(stream);
	cusolverDnDestroySyevjInfo(syevj_params);
}

void cusolverDsyevdMG(const int N, double *A, double *W)
{
	cusolverMgHandle_t handle = NULL;
        cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat = cudaSuccess;

	/* maximum number of GPUs */

	const int MAX_NUM_DEVICES = 8;

	int nbGpus = 0;
	int deviceList[MAX_NUM_DEVICES];

	const int IA  = 1;
	const int JA  = 1;
	const int T_A = 512; /* tile size */
	const int lda = N;
	int  info = 0;

	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

	cudaLibMgMatrixDesc_t descrA;
	cudaLibMgGrid_t gridA;
	cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

	double **array_d_A = NULL;

	int64_t lwork = 0 ; /* workspace: number of elements per device */
	double **array_d_work = NULL;

	status = cusolverMgCreate(&handle);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat = cudaGetDeviceCount( &nbGpus );
	assert( cudaSuccess == cudaStat );

	nbGpus = (nbGpus < MAX_NUM_DEVICES)? nbGpus : MAX_NUM_DEVICES;
	//std::cout<<"\tthere are "<<nbGpus<<" GPUs \n"<<std::endl;
	for(int j = 0 ; j < nbGpus ; j++){
		deviceList[j] = j;
		//cudaDeviceProp prop;
		//cudaGetDeviceProperties(&prop, j+4);
		//std::cout<<"\tdevice "<<j+4<<", "<<prop.name<<", cc "<<prop.major<<"."<<prop.minor<<" \n"<<std::endl;
	}

	status = cusolverMgDeviceSelect(handle,nbGpus,deviceList);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	//assert( 0 == enablePeerAccess( nbGpus, deviceList ) );

	status = cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList, mapping );
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* (global) A is N-by-N */
	status = cusolverMgCreateMatrixDesc(
	&descrA,
	N,   /* nubmer of rows of (global) A */
	N,   /* number of columns of (global) A */
	N,   /* number or rows in a tile */
	T_A, /* number of columns in a tile */
	CUDA_R_64F,
	gridA 
	);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	array_d_A = (double**)malloc(sizeof(double*)*nbGpus);
	assert(NULL != array_d_A);
	
	/* A := 0 */
	createMat<double>(
	nbGpus,
	deviceList,
	N,   /* number of columns of global A */
	T_A, /* number of columns per column tile */
	lda, /* leading dimension of local A */
	array_d_A
	);

	memcpyH2D<double>(
	nbGpus,
	deviceList,
	N,
	N,
	/* input */
	A,
	lda,
	/* output */
	N,   /* number of columns of global A */
	T_A, /* number of columns per column tile */
	lda, /* leading dimension of local A */
	array_d_A,   /* host pointer array of dimension nbGpus */
	IA,
	JA
	);

	status = cusolverMgSyevd_bufferSize(
        handle,
        (cusolverEigMode_t)jobz,
        CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
        N,
        (void**)array_d_A,
        IA, /* base-1 */
        JA, /* base-1 */
        descrA,
        (void*)W,
        CUDA_R_64F,
        CUDA_R_64F,
        &lwork
	);
   	assert(CUSOLVER_STATUS_SUCCESS == status);

	array_d_work = (double**)malloc(sizeof(double*)*nbGpus);
	assert( NULL != array_d_work);

	/* array_d_work[j] points to device workspace of device j */
	workspaceAlloc(
	nbGpus,
	deviceList,
	sizeof(double)*lwork, /* number of bytes per device */
	(void**)array_d_work
	);

	/* sync all devices */
	cudaStat = cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat);

	status = cusolverMgSyevd(
	handle,
	(cusolverEigMode_t)jobz,
	CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
	N,
	(void**)array_d_A,  /* exit: eigenvectors */
	IA,
	JA,
	descrA,
	(void**)W,  /* exit: eigenvalues */
	CUDA_R_64F,
	CUDA_R_64F,
	(void**)array_d_work,
	lwork,
	&info  /* host */
	);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* sync all devices */
	cudaStat = cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat);

	/* check if SYEVD converges */
	assert(0 == info);

	memcpyD2H<double>(
        nbGpus,
        deviceList,
        N,
        N,
	/* input */
        N,   /* number of columns of global A */
        T_A, /* number of columns per column tile */
        lda, /* leading dimension of local A */
        array_d_A,
        IA,
        JA,
	/* output */
        A,   /* N-y-N eigenvectors */
        lda
    	);

	destroyMat(
        nbGpus,
        deviceList,
        N,   /* number of columns of global A */
        T_A, /* number of columns per column tile */
        (void**)array_d_A );

	workspaceFree( nbGpus, deviceList, (void**)array_d_work );

	free(array_d_A);
	free(array_d_work);
	cusolverMgDestroy(handle);
}

#endif
