#include <iostream>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <thrust/complex.h>
#include "../include/fermi_operator.h"
#include "../include/HubMatrix_GPU.cuh"
#include "../include/entangle_entropy.h"

void Line12(H_TBq &Hq, double U, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist);

int main(int argc, char * argv[])
{
	const int N = atoi(argv[1]);
	const int Q = atoi(argv[2]);
	const double U = atof(argv[3]);
	const double x_coe = atof(argv[4]);	

	cudaSetDevice(1);

	H_TBq Hq(N,Q);
	int dim = Hq.count;
	
	int Asize = 6;
	int Asite[6] = {0,1,2,3,4,5};
	
	EEmatAB<thrust::complex<double>> EEfermi(Asite,Asize,Hq);
	int rowEE = EEfermi.dimA, colEE = EEfermi.dimB;
	thrust::device_vector<thrust::complex<double>> EEmat(rowEE*colEE,0);
	
	tED<double> time_ED(dim,1.);
	thrust::device_vector<double> Hmat(dim*dim,0);

	// initialize state //
	thrust::device_vector<thrust::complex<double>> psi0(dim,0),psi(dim,0);
	int state0 = 63; // state0 =  000000111111
	int Hstate = Hq.address[state0];
	psi0[Hstate] = thrust::complex<double>(1.0,0);	

	// random number //
	std::random_device rn;
	std::mt19937 rand(rn());
	std::uniform_real_distribution<double> dist(-1.0,1.0);	

	// out File //
	std::string file_name = "sf_EE_U"+std::to_string((int)U)+"_x"+std::to_string((int)x_coe)+".out";
	std::ofstream outF(file_name);
	outF.precision(10);

	int Nt = 500, x_conf = 500; double dt = 0.02, SE=0;
	for(int i=0;i<x_conf*Nt;i++)
	{
		int tid = i%Nt, xid = i/Nt;

		if(tid==0){
			Line12(Hq,U,x_coe,rand,dist);
			Hq.Write(Hmat);	
			time_ED.Make_eigen(Hmat);
		}

		time_ED.tED_time_evol(psi0,psi,exp((tid+1)*dt)-1.);

		EEfermi.Make_Write(psi,EEmat);
		double S[rowEE];
		cuDoubleComplex * ptr_EEmat = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(EEmat.data()));
		cusolverZsvd(colEE,rowEE,ptr_EEmat,S);
		for(int j=0;j<rowEE;j++) SE += -S[j]*log(S[j]);

		if(tid == 0 && xid != 0) outF<<"\n"<<std::endl;
		outF<<xid<<"\t"<<exp((tid+1)*dt)-1.<<"\t"<<SE<<std::endl;		

		psi.clear(); psi.resize(dim,0);
		EEmat.clear(); EEmat.resize(rowEE*colEE,0);
		SE = 0;
		if(tid==Nt-1){ 
			Hq.clear();
			Hmat.clear(); Hmat.resize(dim*dim,0);
			time_ED.clear();
		}	
	}	

	return 0;
}

void Line12(H_TBq &Hq, double U, double x_coe, std::mt19937 &rand, std::uniform_real_distribution<double> &dist)
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

	Hq.Insert_2Nop(0,1,U);
	Hq.Insert_2Nop(1,2,U);
	Hq.Insert_2Nop(2,3,U);
	Hq.Insert_2Nop(3,4,U);
	Hq.Insert_2Nop(4,5,U);
	Hq.Insert_2Nop(5,6,U);
	Hq.Insert_2Nop(6,7,U);
	Hq.Insert_2Nop(7,8,U);
	Hq.Insert_2Nop(8,9,U);
	Hq.Insert_2Nop(9,10,U);
	Hq.Insert_2Nop(10,11,U);

	for(int i=0;i<12;i++) Hq.Make_N(i,x_coe*dist(rand));
};




