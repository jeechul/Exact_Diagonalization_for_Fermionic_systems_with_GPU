#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include "../../MBL_test/include/fermi_operator.h"
#include "fermi_operator_lowmem.h"

#define CUDAERR(x){\
	if(x != cudaSuccess) {\
		std::cout << "Error! : #" << __LINE__ << std::endl;\
	}\
}\

namespace kernel
{
template<typename FloatType>
__global__ void Hubbard_mv_(const int * stateUP, const int * stateDN, const int * addressUP, const int * addressDN,
        FloatType * x, FloatType * y, const int dimUP, const int dimDN,
        const int N, const int blocksPerSubvector)
{
	FloatType sum=0;
	int threadsPerBlock = dimDN/blocksPerSubvector;

	if(blockIdx.x<dimUP*blocksPerSubvector){	
        int sv = blockIdx.x/blocksPerSubvector;
		int id = (blockIdx.x%blocksPerSubvector)*threadsPerBlock+threadIdx.x;
		int gid = blockIdx.x*blockDim.x+threadIdx.x;

		if(id<dimDN){
            // 0.5*(S^+ S^-)+0.5*(S^- S^+)
            // = -sum_{j>i}(c^{+}_{j,up}c_{i,up}c^{+}_{i,dn}c_{j,dn}
            //              + c^{+}_{i,up}c_{j,up}c^{+}_{j,dn}c_{i,dn})
            //   +0.5*sum_{i}[ (1-n_{i,dn})n_{i,up}+(1-n_{i,up})n_{i,dn} ]
            int up = stateUP[sv], dn = stateDN[id];

            for(int i=0;i<N;++i){
                int iup = (up>>i)%2, idn = (dn>>i)%2;
                if (iup != idn){
                    for (int j=i+1;j<N;++j){
                        int jup = (up>>j)%2, jdn = (dn>>j)%2;
                        int exchanger = (1<<i)|(1<<j);
                        if (iup == jdn && idn == jup){
                            int up_ = up^exchanger, dn_ = dn^exchanger;
                            int fermiSign = count1_bt_sites(up_,i,j)+count1_bt_sites(dn_,i,j);
                            if (fermiSign%2==0){
                                sum -= x[ addressUP[up_]*dimDN+addressDN[dn_] ];
                            } else {
                                sum += x[ addressUP[up_]*dimDN+addressDN[dn_] ];
                            }
                        }
                    }
                }
			}

            int doubles = up^dn;
            double coef = 0.5*count1(doubles); 
            sum += coef*x[sv*dimDN+id];

            y[gid] = sum; 
		}
	}
}

template<typename FloatType>
__global__ void perform_op_(const double * AxUP, const int * AjUP, const double * AxDN, const int * AjDN, 
		const LONG * stateUP, const LONG * stateDN, FloatType * x, FloatType * y, 
        const int dimUP, const int dimDN, const int numcolsUP, const int numcolsDN, 
		const double U, const LONG dilution, const int N, const int blocksPerSubvector)
{
	FloatType sum=0;
	int threadsPerBlock = dimDN/blocksPerSubvector;
	extern __shared__ double shared[];
	double * Axs = shared;
	int * Ajs = (int*)&Axs[numcolsUP];

	if(blockIdx.x<dimUP*blocksPerSubvector){
		int sv = blockIdx.x/blocksPerSubvector;
		int id = (blockIdx.x%blocksPerSubvector)*threadsPerBlock+threadIdx.x;
		int gid = blockIdx.x*blockDim.x+threadIdx.x;

		if(threadIdx.x < numcolsUP){
			Axs[threadIdx.x] = AxUP[threadIdx.x*dimUP+sv];
			Ajs[threadIdx.x] = AjUP[threadIdx.x*dimUP+sv];
		}
		__syncthreads();

		if(id<dimDN){
            // H0up(x)Idn*x_vec  
            //sum_i(|sv>_up AxUP_{sv,i} up_<i|)(|id>_dn dn_<id|)(x_{i,id}|i>_up |id>_dn)
			for(int i=0;i<numcolsUP;i++){
				if(Axs[i]!=0) sum += Axs[i]*x[ Ajs[i]*dimDN+id ];
			}

            // Iup(x)H0dn*x_vec
            // sum_i (|sv>_up up_<sv|)(|id>_dn AxDN_{id,i} dn_<i|)(x_{sv,i}|sv>_up |i>_dn)
			for(int i=0;i<numcolsDN;i++){
				double Aij = AxDN[i*dimDN+id];
				int col = AjDN[i*dimDN+id];
				sum += Aij*x[sv*dimDN+col];
			}

			if(U!=0){
				LONG doubles = (stateUP[gid/dimDN]&stateDN[gid%dimDN])&dilution;
				double intE = U*count1(doubles);
				sum += intE*x[sv*dimDN+id];
			}
			
            y[gid] = sum; 
		}
	}
}

} // namespace kernel


namespace TotalSpin
{
/////////////////////////////
//---Hubbard Hamiltonian---//
/////////////////////////////

struct Hubbard
{
private:
	thrust::device_vector<int> stateUP_dev, stateDN_dev;
	thrust::device_vector<int> addressUP_dev, addressDN_dev;
    const unsigned int N, dimUP, dimDN; 
    int BPS;
    
public:
	Hubbard(const H_TBq& HqUP, const H_TBq& HqDN):
		N(HqUP.N), dimUP(HqUP.count), dimDN(HqDN.count)
	{
		for(int i=1;i<dimDN;i++){
			if(dimDN%i==0){
				if(dimDN/i<=1024){
					BPS=i;
					break;
				}
			}
		}

		stateUP_dev = HqUP.state;
		stateDN_dev = HqDN.state;
        addressUP_dev = HqUP.address;
		addressDN_dev = HqDN.address;
	}

	template<typename FloatType>
	void Hubbard_mv(thrust::device_vector<FloatType> &x, thrust::device_vector<FloatType> &y);
};

template <typename FloatType>
void Hubbard::Hubbard_mv(thrust::device_vector<FloatType> &x, thrust::device_vector<FloatType> &y)
{
    const int * stateUP = thrust::raw_pointer_cast(stateUP_dev.data());
    const int * stateDN = thrust::raw_pointer_cast(stateDN_dev.data());
    const int * addressUP = thrust::raw_pointer_cast(addressUP_dev.data());
    const int * addressDN = thrust::raw_pointer_cast(addressDN_dev.data());
    FloatType * ptr_x = thrust::raw_pointer_cast(x.data());
    FloatType * ptr_y = thrust::raw_pointer_cast(y.data());

    const int numBlocks = dimUP*BPS;
    const int threadsPerBlock = dimDN/BPS;
    kernel::Hubbard_mv_<FloatType><<<numBlocks,threadsPerBlock>>>(
        stateUP, stateDN, addressUP, addressDN,
        ptr_x, ptr_y, dimUP, dimDN, N, BPS);
    cudaDeviceSynchronize();
}

} // namespace TotalSpin


namespace cuSpectra
{

template<typename FloatType>
struct Hubbard
{
	thrust::device_vector<double> AxUP_D;
	thrust::device_vector<int> AjUP_D;
	thrust::device_vector<double> AxDN_D;
	thrust::device_vector<int> AjDN_D;
	thrust::device_vector<LONG> stateUP;
	thrust::device_vector<LONG> stateDN;
public:
    using Scalar = FloatType;

	int dimUP,dimDN,BPS,N,dim; 
	const int numcolsUP,numcolsDN; 	
	const double U;
	const bool is_diluted;
	LONG dilution;

	Hubbard(const H_TBq& HqUP, const H_TBq& HqDN, const double U=0.0):
		U(U), numcolsUP(z*HqUP.Q+1), numcolsDN(z*HqDN.Q+1), is_diluted(false)
	{
		dimUP = HqUP.count;
		dimDN = HqDN.count;
		N = HqUP.N;
        dim = dimUP*dimDN;
		dilution = (1ULL<<N)-1;

		for(int i=1;i<dimDN;i++){
			if(dimDN%i==0){
				if(dimDN/i<=1024){
					BPS=i;
					break;
				}
			}
		}
	
		AxUP_D = HqUP.Ax;
		AjUP_D = HqUP.Aj;
		AxDN_D = HqDN.Ax;
		AjDN_D = HqDN.Aj;
	
		stateUP = HqUP.state;
		stateDN = HqDN.state;
	}	
	
	Hubbard(const H_TBq& HqUP, const H_TBq& HqDN, const double U, const LONG dilution):
		U(U), dilution(dilution), numcolsUP(z*HqUP.Q+1), numcolsDN(z*HqDN.Q+1), is_diluted(true)
	{
		dimUP = HqUP.count;
		dimDN = HqDN.count;
		N = HqUP.N;
        dim = dimUP*dimDN;

		for(int i=1;i<dimDN;i++){
			if(dimDN%i==0){
				if(dimDN/i<=1024){
					BPS=i;
					break;
				}
			}
		}

		AxUP_D = HqUP.Ax;
		AjUP_D = HqUP.Aj;
		AxDN_D = HqDN.Ax;
		AjDN_D = HqDN.Aj;
	
		stateUP = HqUP.state;
		stateDN = HqDN.state;
	}

    Hubbard(const LOW_MEM::H_TBq& HqUP, const LOW_MEM::H_TBq& HqDN, const double U=0.0):
		U(U), numcolsUP(LOW_MEM::z*HqUP.Q+1), numcolsDN(LOW_MEM::z*HqDN.Q+1), is_diluted(false)
	{
		dimUP = HqUP.count;
		dimDN = HqDN.count;
		N = HqUP.N;
        dim = dimUP*dimDN;
		dilution = (1ULL<<N)-1;

		for(int i=1;i<dimDN;i++){
			if(dimDN%i==0){
				if(dimDN/i<=1024){
					BPS=i;
					break;
				}
			}
		}
	
		AxUP_D = HqUP.Ax;
		AjUP_D = HqUP.Aj;
		AxDN_D = HqDN.Ax;
		AjDN_D = HqDN.Aj;
	
		stateUP = HqUP.state;
		stateDN = HqDN.state;
	}	

    int rows() const { return dim; }
    int cols() const { return dim; }

    void perform_op(const FloatType *x, FloatType *y) const
    {
        const double * ptr_AxUP = thrust::raw_pointer_cast(AxUP_D.data());
        const int * ptr_AjUP = thrust::raw_pointer_cast(AjUP_D.data());
        const double * ptr_AxDN = thrust::raw_pointer_cast(AxDN_D.data());
        const int * ptr_AjDN = thrust::raw_pointer_cast(AjDN_D.data());
        const LONG * ptr_stateUP = thrust::raw_pointer_cast(stateUP.data());
        const LONG * ptr_stateDN = thrust::raw_pointer_cast(stateDN.data());

        thrust::device_vector<FloatType> x_dev(dim), y_dev(dim);
        FloatType * ptr_x = thrust::raw_pointer_cast(x_dev.data());
        FloatType * ptr_y = thrust::raw_pointer_cast(y_dev.data());
        
        //thrust::copy(x,x+dim,x_dev.begin());
        cudaMemcpy(ptr_x,x,sizeof(FloatType)*dim,cudaMemcpyHostToDevice);
        //thrust::copy(y,y+dim,y_dev.begin());
        cudaMemcpy(ptr_y,y,sizeof(FloatType)*dim,cudaMemcpyHostToDevice);

        int numBlocks = dimUP*BPS;
        int threadsPerBlock = dimDN/BPS;
        size_t shared = numcolsUP*sizeof(double)+numcolsUP*sizeof(int);
        kernel::perform_op_<<<numBlocks,threadsPerBlock,shared>>>(
            ptr_AxUP, ptr_AjUP, ptr_AxDN, ptr_AjDN, 
            ptr_stateUP, ptr_stateDN, ptr_x, ptr_y, dimUP, dimDN, numcolsUP, numcolsDN, 
            U, dilution, N, BPS);
        cudaDeviceSynchronize();
        
        //thrust::copy(y_dev.begin(),y_dev.end(),y);
        cudaMemcpy(y,ptr_y,sizeof(FloatType)*dim,cudaMemcpyDeviceToHost);
    }

    ~Hubbard()
	{
		AxUP_D.clear(); thrust::device_vector<double>().swap(AxUP_D);
		AjUP_D.clear(); thrust::device_vector<int>().swap(AjUP_D);
		AxDN_D.clear(); thrust::device_vector<double>().swap(AxDN_D);
		AjDN_D.clear(); thrust::device_vector<int>().swap(AjDN_D);
		stateUP.clear(); thrust::device_vector<LONG>().swap(stateUP);
		stateDN.clear(); thrust::device_vector<LONG>().swap(stateDN);
	}
};

} // namespace cuSpectra

















