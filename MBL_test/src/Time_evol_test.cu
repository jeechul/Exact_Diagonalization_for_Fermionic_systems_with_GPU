#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <thrust/complex.h>
#include "../include/fermi_operator.h"
#include "../include/HubMatrix_GPU.cuh"

void Squared_4by4(H_TBq& Hq);

int main(int argc, char * argv[])
{
	const int N = atoi(argv[1]);
	const int QUP = atoi(argv[2]);
	const int QDN = atoi(argv[3]);
	const double U = atof(argv[4]);

	clock_t start,end;

	cudaSetDevice(0);

	//------- GPU Hubbard ----------//

	H_TBq HqUP(N,QUP);
	H_TBq HqDN(N,QDN);
	Squared_4by4(HqUP);
	Squared_4by4(HqDN);

	int dimUP = HqUP.count, dimDN = HqDN.count;
	thrust::device_vector<thrust::complex<double>> phi0(dimUP*dimDN,0),phi(dimUP*dimDN,0);

	// initialize phi0 //
	int UPstate = 15, DNstate = 15; // states = 000000001111
	int Hubstate = HqUP.address[UPstate]*dimDN+HqDN.address[DNstate];
	phi0[Hubstate] = thrust::complex<double>(1.,0);

	// Time evolution //

	Hubbard Hub(HqUP,HqDN,U);
	Krylov<double> Kry(dimUP,dimDN);

	std::cout<<"Reduced Hilbert space dim : "<<dimUP*dimDN<<"\n"<<std::endl;

	double dt = 1.;
	int Nt = 1000, Ntb=10;
/*
	//-------- Krylov time evolution ----------//

	start = clock(); // time start

	Kry.Make_tridiagonal_eigen(Hub,phi0);
	std::cout<<"tridiagonal done\n"<<std::endl;

	std::cout<<"|phi(t)> : "<<std::endl;
	for(int i=0;i<Nt;i++){
		Kry.Krylov_time_evol(Hub,phi0,phi,(i%Ntb+1)*dt);
		std::cout<<i<<"-th wave vector : ";
		for(int j=0;j<2;j++) std::cout<<phi[j]<<" "; std::cout<<std::endl;

		if(i%Ntb == Ntb-1){
			phi0 = phi;
			Kry.clear();
			Kry.Make_tridiagonal_eigen(Hub,phi0);
		}
		
		phi.clear(); phi.resize(dimUP*dimDN,0);
	}

	end = clock(); // time end
	std::cout<<"Krylov time : "<<(end-start)/CLOCKS_PER_SEC<<"\n"<<std::endl;
*/
	//--------- Full diagonalization ----------//

	start = clock(); // time start

	thrust::device_vector<double> Hmat(dimUP*dimDN*dimUP*dimDN,0);
	Hub.Write(Hmat);
	std::cout<<"Hub matrix write done\n"<<std::endl;

	tED<double> time_ED(dimUP,dimDN);
	time_ED.Make_eigen(Hmat);
	std::cout<<"Hub matix eigen solve done\n"<<std::endl;	
	
	phi0.clear(); phi0.resize(dimUP*dimDN,0);
	phi0[Hubstate] = thrust::complex<double>(1.,0);

	std::cout<<"|phi(t)> : "<<std::endl;
	for(int i=0;i<=Nt;i++){	
		time_ED.tED_time_evol(phi0,phi,(i+1)*dt);	
		std::cout<<i<<"-th wave vector : ";
		for(int j=0;j<2;j++) std::cout<<phi[j]<<" "; std::cout<<std::endl;
		phi.clear(); phi.resize(dimUP*dimDN,0);
	}

	end = clock(); // time end
	std::cout<<"tED time : "<<(end-start)/CLOCKS_PER_SEC<<std::endl;

	return 0;
}

void Squared_4by4(H_TBq& Hq){
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


