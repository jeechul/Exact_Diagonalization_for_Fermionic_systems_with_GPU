#ifndef __TB_MODEL__
#define __TB_MODEL__

#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

const int z=10;

template<typename T>
inline void view_mat(T * mat,int coldim,int rowdim)
{ 
	for(int i=0;i<rowdim;i++){ for(int j=0;j<coldim;j++) cout<<mat[i*coldim+j]<<" "; cout<<endl; } 
}

__host__ __device__ int count1(int state)
{
	int i;
	for(i=0;state!=0;i++) state &= (state-1);
	return i;
}

__host__ __device__ int count1_bt_sites(int basis, int i_site, int j_site) // j_site > i_site
{
	int basis_;

	if(j_site>i_site){
		basis_ = basis>>(i_site+1);
		basis_ = basis_%(1<<(j_site-i_site-1));
	}
	else{
		basis_ = basis>>(j_site+1);
		basis_ = basis_%(1<<(i_site-j_site-1));
	}

	return count1(basis_);	 
}

class H_TBq
{
	void Labeling()
	{
		for(int n=0;n<dim;n++){
			if(count1(n)==Q){
				state[count] = n;
				address[n] = count;
				count += 1;
			}
		}
	}

	void Full_Labeling()
	{
		for(int n=0;n<dim;n++){
			state[count] = n;
			address[n] = count;
			count += 1;
		}
	}

	void stacking(double * Ax_idx, int * Aj_idx, int b, double sign, double hopping)
	{
		if(*Ax_idx != 0.0){ 
			stacking(Ax_idx+count,Aj_idx+count,b,sign,hopping);
		}
		else{
		       	*Ax_idx += sign*hopping;
		       	*Aj_idx = b;
		}
	}
public:
	int N,Q,dim; //N # of sites, Q # of particles, dim size of Hilbert space 
	thrust::host_vector<int> state,address,Aj;
	thrust::host_vector<double> Ax;
	int count;

	H_TBq(const int &N, const int &Q):N(N),Q(Q),count(0)
	{
		dim = 1<<N;
		state.resize(dim);
		address.resize(dim);
		Labeling(); 
		Ax.resize(count*(z*Q+1),0);
		Aj.resize(count*(z*Q+1),0);
	}

	H_TBq(const int &N):N(N),Q(N),count(0)
	{
		dim = 1<<N;
		state.resize(dim);
		address.resize(dim);
		Full_Labeling(); 
		Ax.resize(count*(z*Q+1),0);
		Aj.resize(count*(z*Q+1),0);
	}
	
	void Make_H(int i_site, int j_site, double t) // j_site > i_site
	{
		int dimq = count,fermi_sign; // dimq size of reduced Hilbert space
		for(int a=0;a<dimq;a++){
			int n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,j_site);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;
			if(occ[i_site] != occ[j_site]){
				int m = (1<<i_site)|(1<<j_site);
				int l = n^m; // new state through Cdagger_i*C_j+Cdagger_j*C_i
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),t);
			}	
		}
	}

	void Make_J(int i_site, int j_site, double t)
	{
		int dimq = count,fermi_sign;
		for(int a=0;a<dimq;a++){
			int n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,j_site);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;
			if(occ[i_site]==0 && occ[j_site]==1){ // t*Cdag_i*C_j
				int m = (1<<i_site)|(1<<j_site);
				int l = n^m;
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),t); 
			}
			else if(occ[i_site]==1 && occ[j_site]==0){ // -t*Cdag_j*C_i
				int m = (1<<i_site)|(1<<j_site);
				int l = n^m;
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),-t);
			}	
		}
	}

	void Make_N(int i_site, double v)
	{
		int dimq = count;
		for(int a=0;a<dimq;a++){
			int n = state[a];
			if( (n>>i_site)%2==1 ){ 
				stacking(&Ax[0]+a,&Aj[0]+a,a,1.,v); // v*Cdagger_i*C_i
			}
		}
	}

	void Insert_1Fop(const char i_dag, int i_site, double t) // use only full Hilbert space
	{
		int dimq = count,fermi_sign,i_bit;

		switch(i_dag)
		{
			case '+' : 
				i_bit = 0; break;
			case '-' : 
				i_bit = 1; break;
			default : cout<<"wrong sign!"<<endl; break;
		}	
	
		for(int a=0;a<dimq;a++){
			int n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,N);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;		

			if(occ[i_site]==i_bit){ // t*C^(i_dag)_i
				int m = (1<<i_site);
				int l = n^m;
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),t); 
			}
		}
	}

	void Insert_2Fop(const char i_dag, int i_site, const char j_dag, int j_site, double t) // dagger priority > ascending-order priority
	{
		int dimq = count,fermi_sign,i_bit,j_bit;

		switch(i_dag)
		{
			case '+' : 
				i_bit = 0; break;
			case '-' : 
				i_bit = 1; break;
			default : cout<<"wrong sign!"<<endl; break;
		}	

		switch(j_dag)
		{
			case '+' : 
				j_bit = 0; break;
			case '-' : 
				j_bit = 1; break;
			default : cout<<"wrong sign!"<<endl; break;
		}

		for(int a=0;a<dimq;a++){
			int n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,j_site);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;		

			if(occ[i_site]==i_bit && occ[j_site]==j_bit){ // t*C^{+}_{i}*C^{-}_{j} if i_dag='+', j_dag='-'
				int m = (1<<i_site)|(1<<j_site);          // t*C^{+}_{j}*C^{-}_{i} if j_dag='+', i_dag='-'
				int l = n^m;                              // t*C^{dag}_{i}*C^{dag}_{j} if i_dag=j_dag, i>j
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),t); 
			}
		}
	}

	void Insert_2Nop(int i_site, int j_site, double v) // j_site > i_site
	{
		int dimq = count;
		for(int a=0;a<dimq;a++){
			int n = state[a];
			if( (n>>i_site)%2==1 && (n>>j_site)%2==1 ){ 
				stacking(&Ax[0]+a,&Aj[0]+a,a,1.,v); // v*N_i*N_j
			}
		}
	}

	void Write(thrust::device_vector<double> &Mat)
	{
		for(int i=0;i<(z*Q+1);i++){
			for(int j=0;j<count;j++){ 
				int idx = i*count+j;
				if(Ax[idx]!=0) Mat[j*count+Aj[idx]] += Ax[idx];
			}
		}
	}

	void clear()
	{
		Ax.clear(); Ax.resize(count*(z*Q+1),0);
		Aj.clear(); Aj.resize(count*(z*Q+1),0);
	}
};

void Squared_2by2(H_TBq &Hq)
{
	int t = -1.0;

	Hq.Make_H(0,1,t);
	Hq.Make_H(2,3,t);
	Hq.Make_H(0,2,t);
	Hq.Make_H(1,3,t);
}

void Squared_2by2_J(H_TBq &Hq)
{
	int t = 1.0;

	Hq.Make_J(0,1,t);
	Hq.Make_J(2,3,t);
	Hq.Make_J(0,2,t);
	Hq.Make_J(1,3,t);
}

#endif


