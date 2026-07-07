#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <thrust/complex.h>
#include "../include/fermi_operator.h"
#include "../include/HubMatrix_GPU.cuh"
#include "../include/entangle_entropy_GPU.cuh"
 
void Squared_4by4(H_TBq& Hq);
void Squared_3by4(H_TBq& Hq);

int main(int argc, char * argv[])
{
	const int N = atoi(argv[1]);
	const int QUP = atoi(argv[2]);
	const int QDN = atoi(argv[3]);
	const double u = atof(argv[4]);

	clock_t start,end;
	start = clock();

	cudaSetDevice(7);

	//------- GPU Hubbard ----------//

	H_TBq HqUP(N,QUP);
	H_TBq HqDN(N,QDN);
	Squared_3by4(HqUP);
	Squared_3by4(HqDN);

	int dimUP = HqUP.count, dimDN = HqDN.count;
	thrust::device_vector<double> f(dimUP*dimDN),evec(dimUP*dimDN);
	double eval;

	Hubbard Hub(HqUP,HqDN,u);
	Lanczos<double>	Lanc(dimUP,dimDN);

	std::cout<<"Reduced Hilbert space dim : "<<dimUP*dimDN<<std::endl;

	Lanc.random_vector(f);	
	Lanc.eigen(Hub,f,eval,evec);

	thrust::host_vector<double> evec_h = evec;
	std::cout<<"Evec : "<<std::endl; view_mat(&evec_h[0],10,10); 
	std::cout<<"Eval : "<<eval<<std::endl;

	Lanc.clear(); 
	
	end = clock();
	std::cout<<"GPU time : "<<(end-start)/CLOCKS_PER_SEC<<std::endl;
/*
	//----------- EE calculation for GS -----------//

	int Asize = 8;
	int Asite[8] = {0,1,2,3,4,5,6,7};	

	EEmatHubAB_GPU<double> EEhub(Asite,Asize,HqUP,HqDN);

	int rowD = EEhub.rowD, colD = EEhub.colD;
	std::cout<<"rowD, colD : "<<rowD<<", "<<colD<<std::endl;

	int dimAUP = (EEhub.EEmatUP).dimA, dimBUP = (EEhub.EEmatUP).dimB;
	std::cout<<"dimAUP, dimBUP : "<<dimAUP<<", "<<dimBUP<<std::endl;

	start = clock();
	thrust::device_vector<double> EEmat(rowD*colD,0);
	EEhub.Make_Write(evec,EEmat);
	end = clock();
	std::cout<<"Write time : "<<(end-start)/CLOCKS_PER_SEC<<std::endl;

	double S[rowD];
	start = clock();
	cusolverDsvd(colD,rowD,thrust::raw_pointer_cast(EEmat.data()),S);
	end = clock();
	std::cout<<"SVD time : "<<(end-start)/CLOCKS_PER_SEC<<std::endl;

	std::ofstream outF1("SVD_value_test.out");
	outF1.precision(10);	
	for(int i=0;i<rowD;i++) outF1<<i<<"\t"<<S[i]<<std::endl;	
*/
	return 0;
}

void Squared_4by4(H_TBq& Hq)
{
	const double t = -1.0;

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
}

void Squared_3by4(H_TBq& Hq)
{
	const double t = -1.0;

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
	
	// y direction

	Hq.Make_H(0,4,t);
	Hq.Make_H(4,8,t);

	Hq.Make_H(1,5,t);
	Hq.Make_H(5,9,t);

	Hq.Make_H(2,6,t);
	Hq.Make_H(6,10,t);

	Hq.Make_H(3,7,t);
	Hq.Make_H(7,11,t);
}





