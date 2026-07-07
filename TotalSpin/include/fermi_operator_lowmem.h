#ifndef __TB_MODEL_LOW_MEM__
#define __TB_MODEL_LOW_MEM__

#include <iostream>
#include <cmath>
#include <map>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
using LONG = unsigned long long;

namespace LOW_MEM
{

const int z=10;

template<typename T>
inline void view_mat(T * mat,int coldim,int rowdim)
{ 
	for(int i=0;i<rowdim;i++){ for(int j=0;j<coldim;j++) cout<<mat[i*coldim+j]<<" "; cout<<endl; } 
}

__host__ __device__ int count1(LONG state)
{
	int i;
	for(i=0;state!=0;i++) state &= (state-1);
	return i;
}

__host__ __device__ int count1_bt_sites(LONG basis, int i_site, int j_site) // j_site > i_site
{
	LONG basis_;

	if(j_site>i_site){
		basis_ = basis>>(i_site+1);
		basis_ = basis_%(1ULL<<(j_site-i_site-1));
	}
	else{
		basis_ = basis>>(j_site+1);
		basis_ = basis_%(1ULL<<(i_site-j_site-1));
	}

	return count1(basis_);	 
}

class H_TBq
{
	void Labeling()
	{
        std::vector<bool> occ(N,false);
        for (int i=0;i<Q;++i) occ[i] = true;
        
        do{
            LONG n=0;
            for (int i=0;i<N;++i) {
                if (occ[i]) n += (1ULL << i);
            }
            state.push_back(n);
            address[n] = count;
            count += 1;
        }while(prev_permutation(occ.begin(),occ.end()));
	}

	void stacking(double * Ax_idx, int * Aj_idx, int b, double sign, double hopping)
	{
        if ((Ax_idx-&Ax[0]) >= Ax.size()){
            std::cerr << "over Ax size !!" << std::endl;
            exit(1); 
        }

		if(*Ax_idx != 0.0){ 
			stacking(Ax_idx+count,Aj_idx+count,b,sign,hopping);
		}
		else{
		       	*Ax_idx += sign*hopping;
		       	*Aj_idx = b;
		}
	}
public:
	int N,Q; //N : # of sites, Q : # of particles, count : size of reduced Hilbert space 
	thrust::host_vector<LONG> state;
    std::map<LONG,int> address;
    thrust::host_vector<int> Aj;
	thrust::host_vector<double> Ax;
	int count;

	H_TBq(const int &N, const int &Q):N(N),Q(Q),count(0)
	{
		Labeling(); 
		Ax.resize(count*(z*Q+1),0);
		Aj.resize(count*(z*Q+1),0);
	}

	void Make_H(int i_site, int j_site, double t) // j_site > i_site
	{
		int dimq = count,fermi_sign; // dimq : size of reduced Hilbert space
		for(int a=0;a<dimq;a++){
			LONG n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,j_site);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;
			if(occ[i_site] != occ[j_site]){
				LONG m = (1ULL<<i_site)|(1ULL<<j_site);
				LONG l = n^m; // new state through Cdagger_i*C_j+Cdagger_j*C_i
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),t);
			}	
		}
	}

	void Make_J(int i_site, int j_site, double t)
	{
		int dimq = count,fermi_sign;
		for(int a=0;a<dimq;a++){
			LONG n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,j_site);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;
			if(occ[i_site]==0 && occ[j_site]==1){ // t*Cdag_i*C_j
				LONG m = (1ULL<<i_site)|(1ULL<<j_site);
				LONG l = n^m;
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),t); 
			}
			else if(occ[i_site]==1 && occ[j_site]==0){ // -t*Cdag_j*C_i
				LONG m = (1ULL<<i_site)|(1ULL<<j_site);
				LONG l = n^m;
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),-t);
			}	
		}
	}

	void Make_N(int i_site, double v)
	{
		int dimq = count;
		for(int a=0;a<dimq;a++){
			LONG n = state[a];
			if( (n>>i_site)%2==1 ){ 
				stacking(&Ax[0]+a,&Aj[0]+a,a,1.,v); // v*Cdagger_i*C_i
			}
		}
	}

	void Insert_1Fop(const char i_dag, int i_site, double t) // MUST call it only for full Hilbert space
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
			LONG n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,N);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;		

			if(occ[i_site]==i_bit){ // t*C^(i_dag)_i
				LONG m = (1ULL<<i_site);
				LONG l = n^m;
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
			LONG n = state[a];
			fermi_sign = count1_bt_sites(n,i_site,j_site);
			int occ[N];
			for(int i=0;i<N;i++) occ[i] = (n>>i)%2;		

			if(occ[i_site]==i_bit && occ[j_site]==j_bit){        // t*C^{+}_{i}*C^{-}_{j} if i_dag='+', j_dag='-'
				LONG m = (1ULL<<i_site)|(1ULL<<j_site);          // t*C^{+}_{j}*C^{-}_{i} if j_dag='+', i_dag='-'
				LONG l = n^m;                                    // t*C^{dag}_{i}*C^{dag}_{j} if i_dag=j_dag, i>j
				int b = address[l];
				stacking(&Ax[0]+a,&Aj[0]+a,b,pow(-1.,fermi_sign),t); 
			}
		}
	}

	void Insert_2Nop(int i_site, int j_site, double v) // j_site > i_site
	{
		int dimq = count;
		for(int a=0;a<dimq;a++){
			LONG n = state[a];
			if( (n>>i_site)%2==1 && (n>>j_site)%2==1 ){ 
				stacking(&Ax[0]+a,&Aj[0]+a,a,1.,v); // v*N_i*N_j
			}
		}
	}

	void Write(thrust::device_vector<double> &Mat) const
	{
		for(int i=0;i<count;++i){
			for(int j=0;j<(z*Q+1);++j){ 
				int idx = j*count+i;
				if(Ax[idx]!=0) Mat[i*count+Aj[idx]] += Ax[idx];
			}
		}
	}

    void Write(double * Mat) const
	{
		for(int i=0;i<count;++i){
			for(int j=0;j<(z*Q+1);++j){ 
				int idx = j*count+i;
				if(Ax[idx]!=0) Mat[i*count+Aj[idx]] += Ax[idx];
			}
		}
	}

	void clear()
	{
		Ax.clear(); Ax.resize(count*(z*Q+1),0);
		Aj.clear(); Aj.resize(count*(z*Q+1),0);
	}

    using Scalar = double;

    int rows() const { return count; };
    int cols() const { return count; };

    void perform_op(const double * x, double * y) const
    {
        for (int i=0;i<count;++i)
        {
            double y_temp = 0;
            for (int j=0;j<(z*Q+1);++j)
            {
                int idx = j*count+i;
                if (Ax[idx]!=0) y_temp += Ax[idx]*x[Aj[idx]];
            }
            y[i] = y_temp;
        }
    }
};

} // namespace LOW_MEM

#endif


