#include <iostream>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <thrust/complex.h>
#include "../include/fermi_operator.h"
#include "../include/HubMatrix_GPU.cuh"
#include "../include/entangle_entropy_GPU.cuh"

void Line16(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist);
void Square13(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist);
void Square16(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist);
void Square20(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist);

int main(int argc, char * argv[])
{
	const int N = atoi(argv[1]);
	const int QUP = atoi(argv[2]);
	const int QDN = atoi(argv[3]);
	const double U = atof(argv[4]);

	cudaSetDevice(7);

	H_TBq HqUP(N,QUP);
	H_TBq HqDN(N,QDN);
	const int64_t dimUP = HqUP.count, dimDN = HqDN.count;
	const int64_t dimHub = dimUP*dimDN;
	
	int Asize = 8;
	int Asite[8] = {0,1,2,3,4,5,6,7};
	
	EEmatHubAB_GPU<double> EEhub(Asite,Asize,HqUP,HqDN);
	const int64_t rowEE = EEhub.rowD, colEE = EEhub.colD;
	double *EEmat, *EEmat_D;
	cudaHostAlloc((void **)&EEmat,(rowEE*colEE)*sizeof(double),cudaHostAllocMapped|cudaHostAllocPortable);
	cudaHostGetDevicePointer((void **)&EEmat_D,(void *)EEmat,0);

	/*
	EEmatHubSpin_GPU<thrust::complex<double>> EEhub(HqUP,HqDN);
	int rowEE = EEhub.rowD, colEE = EEhub.colD;
	thrust::device_vector<thrust::complex<double>> EEmat(rowEE*colEE,0);
	*/

	tED<double> time_ED(dimUP,dimDN);
	std::cout<<"total Hmat size : "<<dimHub*dimHub<<std::endl;
	double *Hmat, *Hmat_D;
	cudaHostAlloc((void **)&Hmat,(dimHub*dimHub)*sizeof(double),cudaHostAllocMapped|cudaHostAllocPortable);
	cudaHostGetDevicePointer((void**)&Hmat_D,(void *)Hmat,0);

	// Eigen state vector //
	thrust::device_vector<double> psi(dimHub,0);

	// random number //
	std::random_device rn;
	std::mt19937 rand(rn());
	std::uniform_real_distribution<double> dist(-1.0,1.0);	

	// out File //
	std::string file_name = "./Hubbard_model/Hub_AreaLaw_L"+std::to_string(N)+"_Q"+std::to_string(QUP+QDN)+"_U"+std::to_string((int)U)+"_pinned.out";
	std::ofstream outF(file_name);
	outF.precision(10);

	int Nx = 10, x_conf = 100; double x_coe, SE=0;
	for(int i=0;i<x_conf*Nx;i++)
	{
		int xid = i%x_conf, nid = i/x_conf;
		x_coe = (nid+1)*1.0;

		Square16(HqUP,x_coe,rand,dist);
		Square16(HqDN,x_coe,rand,dist);
		Hubbard Hub(HqUP,HqDN,U);
		Hub.initialize(Hmat_D);
		Hub.Write_pinned(Hmat_D);	
		time_ED.Make_eigen_pinned(Hmat);

		cudaSetDevice(7);

		thrust::copy(Hmat+dimHub/2*dimHub,Hmat+(dimHub/2+1)*dimHub,psi.begin());	

		EEhub.initialize(EEmat_D);
		EEhub.Make_Write_pinned(psi,EEmat_D);
		double S[rowEE];
		cusolverDsvd(colEE,rowEE,EEmat_D,S);
		for(int j=0;j<rowEE;j++){ 
			if(S[j]!=0) SE += -S[j]*log(S[j]);
		}

		if(xid == 0 && nid != 0) outF<<"\n"<<std::endl;
		outF<<x_coe<<"\t"<<SE<<std::endl;		
		std::cout<<x_coe<<"\t"<<SE<<std::endl;		

		psi.clear(); psi.resize(dimHub,0);
		SE = 0;

		HqUP.clear();
		HqDN.clear();
		time_ED.clear();	
	}	

	cudaFreeHost(EEmat);
	cudaFreeHost(Hmat);

	return 0;
}

void Line16(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist)
{
	const int t = -1;

	Hq.Make_H(0,1,t);
	Hq.Make_H(1,2,t);
	Hq.Make_H(2,3,t);
	Hq.Make_H(3,4,t);
	Hq.Make_H(4,5,t);
	Hq.Make_H(5,6,t);
	Hq.Make_H(6,7,t);
	Hq.Make_H(7,8,t);
	Hq.Make_H(8,9,t);
	Hq.Make_H(9,10,t);
	Hq.Make_H(10,11,t);
	Hq.Make_H(11,12,t);
	Hq.Make_H(12,13,t);
	Hq.Make_H(13,14,t);
	Hq.Make_H(14,15,t);

	for(int i=0;i<16;i++) Hq.Make_N(i,x_coe*dist(rand));
};

void Square16(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist)
{
	const int t = -1;

	// x direction
	Hq.Make_H(0,1,t);
	Hq.Make_H(1,2,t);
	Hq.Make_H(2,3,t);

	Hq.Make_H(4,5,t);
	Hq.Make_H(5,6,t);
	Hq.Make_H(6,7,t);

	Hq.Make_H(8,9,t);
	Hq.Make_H(9,10,t);
	Hq.Make_H(10,11,t);

	Hq.Make_H(12,13,t);
	Hq.Make_H(13,14,t);
	Hq.Make_H(14,15,t);

	// y direction
	Hq.Make_H(0,4,t);
	Hq.Make_H(4,8,t);
	Hq.Make_H(8,12,t);

	Hq.Make_H(1,5,t);
	Hq.Make_H(5,9,t);
	Hq.Make_H(9,13,t);

	Hq.Make_H(2,6,t);
	Hq.Make_H(6,10,t);
	Hq.Make_H(10,14,t);

	Hq.Make_H(3,7,t);
	Hq.Make_H(7,11,t);
	Hq.Make_H(11,15,t);

	for(int i=0;i<16;i++) Hq.Make_N(i,x_coe*dist(rand));
};

void Square13(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist)
{
	const int t = -1;

	// x direction
	Hq.Make_H(0,1,t);
	Hq.Make_H(1,2,t);
	Hq.Make_H(2,3,t);

	Hq.Make_H(4,5,t);
	Hq.Make_H(5,6,t);

	Hq.Make_H(7,8,t);
	Hq.Make_H(8,9,t);
	Hq.Make_H(9,10,t);

	// y direction
	Hq.Make_H(1,4,t);
	Hq.Make_H(4,7,t);
	Hq.Make_H(7,11,t);

	Hq.Make_H(2,5,t);
	Hq.Make_H(5,8,t);

	Hq.Make_H(3,12,t);
	Hq.Make_H(3,6,t);
	Hq.Make_H(6,9,t);

	for(int i=0;i<13;i++) Hq.Make_N(i,x_coe*dist(rand));
};

void Square20(H_TBq &Hq, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist)
{
	const int t = -1;

	// x direction
	Hq.Make_H(0,1,t);
	Hq.Make_H(1,2,t);
	Hq.Make_H(2,3,t);
	Hq.Make_H(3,4,t);

	Hq.Make_H(5,6,t);
	Hq.Make_H(6,7,t);
	Hq.Make_H(7,8,t);

	Hq.Make_H(9,10,t);
	Hq.Make_H(10,11,t);
	Hq.Make_H(11,12,t);

	Hq.Make_H(13,14,t);
	Hq.Make_H(14,15,t);
	Hq.Make_H(15,16,t);
	Hq.Make_H(16,17,t);

	// y direction
	Hq.Make_H(1,5,t);
	Hq.Make_H(5,9,t);
	Hq.Make_H(9,13,t);
	Hq.Make_H(13,18,t);

	Hq.Make_H(2,6,t);
	Hq.Make_H(6,10,t);
	Hq.Make_H(10,14,t);

	Hq.Make_H(3,7,t);
	Hq.Make_H(7,11,t);
	Hq.Make_H(11,15,t);

	Hq.Make_H(4,8,t);
	Hq.Make_H(8,12,t);
	Hq.Make_H(12,16,t);
	Hq.Make_H(4,19,t);

	for(int i=0;i<20;i++) Hq.Make_N(i,x_coe*dist(rand));
};



