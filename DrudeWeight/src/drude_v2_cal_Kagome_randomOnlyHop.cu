#include <iostream>
#include <cmath>
#include <ctime>
#include <bitset>
#include <fstream>
#include <random>
#include "../../MBL_test/include/HubMatrix_GPU.cuh"
#include "../../MBL_test/include/fermi_operator.h"
#include "../../MBL_test/include/argparse.h"

#define __OUTFILE__

const int N = 24;
const int NOMEGA = 8000;
const double OMEGA_STEP = 0.004;
const double DELTA = 0.01;
const double OMEGA_CUTOFF = 0.01;

const double VOL = 16.*sqrt(3); // 24 sites
//const double VOL = 2.*sqrt((sqrt(13)+sqrt(7)+2)*(sqrt(13)+sqrt(7)-2)*(sqrt(13)-sqrt(7)+2)*(-sqrt(13)+sqrt(7)+2)); // 18 sites	
//const double VOL = 8.*sqrt(3); // 12 sites

void KagomeLattice12(H_TBq& H, const double * rand);
void KagomeKinetic12(H_TBq& Kx, H_TBq& Ky, const double * rand);
void KagomeCurrent12(H_TBq& Jx, H_TBq& Jy, const double * rand);

void KagomeLattice18(H_TBq& H, const double * rand);
void KagomeKinetic18(H_TBq& Kx, H_TBq& Ky, const double * rand);
void KagomeCurrent18(H_TBq& Jx, H_TBq& Jy, const double * rand);

void KagomeLattice24(H_TBq& H, const double * rand);
void KagomeKinetic24(H_TBq& Kx, H_TBq& Ky, const double * rand);
void KagomeCurrent24(H_TBq& Jx, H_TBq& Jy, const double * rand);

void drude_weight_Jx(Lanczos<double>& lanc, Hubbard& Hmat, Hubbard& Jxmat, Hubbard& Kxmat,
	const double &eval, thrust::device_vector<double> &evec, double &drude_weight_xx, bool SigmaOut_on=false, std::string tag="xx");

int main(int argc, char* argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction"));
    options.push_back(pair_t("Q", "# of electrons"));
    options.push_back(pair_t("X", "disorder strength (0<= X <1)"));
    options.push_back(pair_t("Nsam", "# of disorder samples"));
    options.push_back(pair_t("dev", "device number"));
    options.push_back(pair_t("path", "directory to load and save files"));
   	// env; default value
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("Nsam", "1"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    const int Nsam = parser.find<int>("Nsam"),
        dev = parser.find<int>("dev");
    const double U = parser.find<double>("U"),
        X = parser.find<double>("X");
    const auto Q = parser.mfind<int>("Q");
    const int Qup = Q[0], Qdn = Q[1];
    // print info of the registered args
    parser.print(std::cout);

    cudaSetDevice(dev);

	std::random_device rn;
	std::mt19937 generator(rn());
	std::uniform_real_distribution<double> dist(-1.0,1.0);
    std::vector<double> randomSites(N);

	H_TBq H0UP(N,Qup);
	H_TBq H0DN(N,Qdn);

	H_TBq KxUP(N,Qup);
	H_TBq KxDN(N,Qdn);
	H_TBq KyUP(N,Qup);
	H_TBq KyDN(N,Qdn);

	H_TBq JxUP(N,Qup);
	H_TBq JxDN(N,Qdn);
	H_TBq JyUP(N,Qup);
	H_TBq JyDN(N,Qdn);

	unsigned int dimUP = H0UP.count, dimDN = H0DN.count;
	std::cout<<"# Reduced Hilbert space : "<<dimUP*dimDN<<std::endl;
	thrust::device_vector<double> f(dimUP*dimDN),V(dimUP*dimDN,0);
    double E;

    double drude_weight_xx, drude_weight_yy;
	
    const std::string filename = parser.find<>("path") + "/DrudeWeight_v2_Kagome-L" + std::to_string(N) + "U" + parser.find<>("U") 
        + "Q" + parser.find<>("Q")
        + "X" + parser.find<>("X") + ".dat";
#ifdef __OUTFILE__
    std::ofstream outfile;
    if (!std::ifstream(filename).is_open())
    {
        outfile.open(filename);
        outfile << "#       E       Dxx        Dyy        D      " << std::endl;
    }
    else
        outfile.open(filename,std::ios::app);
    outfile.precision(10);
#else
    std::cout << "#       E      Dxx       Dyy        D       " << std::endl;
    std::cout.precision(10);
#endif
	for(int sam=0;sam<Nsam;++sam)
	{
        for (int i=0;i<N;++i)
            randomSites[i] = 1.0+X*dist(generator);

      	KagomeLattice24(H0UP,&randomSites[0]);
        KagomeLattice24(H0DN,&randomSites[0]);
        
        KagomeKinetic24(KxUP,KyUP,&randomSites[0]);
        KagomeKinetic24(KxDN,KyDN,&randomSites[0]);

        KagomeCurrent24(JxUP,JyUP,&randomSites[0]);
        KagomeCurrent24(JxDN,JyDN,&randomSites[0]);

        Hubbard Hmat(H0UP,H0DN,U);
        Hubbard Kxmat(KxUP,KxDN);
        Hubbard Jxmat(JxUP,JxDN);
        Hubbard Kymat(KyUP,KyDN);
        Hubbard Jymat(JyUP,JyDN);
      	
		Lanczos<double> lanc(dimUP,dimDN);
		lanc.random_vector(f);
        lanc.eigen(Hmat,f,E,V);
        
        const bool sigmaout_on=false;
        drude_weight_Jx(lanc,Hmat,Jxmat,Kxmat,E,V,drude_weight_xx,sigmaout_on);
        drude_weight_Jx(lanc,Hmat,Jymat,Kymat,E,V,drude_weight_yy);
#ifdef __OUTFILE__
        outfile << "\t" << E << "\t" << drude_weight_xx << "\t" << drude_weight_yy << 
        "\t" << (drude_weight_xx+drude_weight_yy)/2. << std::endl;
        std::cout << "\r# ---- " << std::setw(4) << sam+1 << "/" << std::setw(4) << Nsam << std::flush;
#else
        std::cout << "\t" << E << "\t" << drude_weight_xx << "\t" << drude_weight_yy << 
        "\t" << (drude_weight_xx+drude_weight_yy)/2. << std::endl;
#endif
		V.assign(dimUP*dimDN,0.0);
        H0UP.clear();
        H0DN.clear();
        KxUP.clear();
        KxDN.clear();
        KyUP.clear();
        KyDN.clear();
        JxUP.clear();
        JxDN.clear();
        JyUP.clear();
        JyDN.clear();
	}
    std::cout<<std::endl;
#ifdef __OUTFILE__
	outfile.close();
#endif
	return 0;
}

void drude_weight_Jx(Lanczos<double>& lanc, Hubbard& Hmat, Hubbard& Jxmat, Hubbard& Kxmat,
	const double &eval, thrust::device_vector<double> &evec, double &drude_weight_xx, bool SigmaOut_on, std::string tag)
{
	thrust::host_vector< thrust::complex<double> > omega(NOMEGA,0.0);
    thrust::host_vector<double> sigma(NOMEGA,0.0);
    thrust::host_vector< thrust::complex<double> > Lambda(NOMEGA,0.0);
    thrust::device_vector<double> KV(lanc.dimUP*lanc.dimDN);

    for(int i=0;i<NOMEGA;i++) omega[i] = thrust::complex<double>(OMEGA_STEP*(double)(i+1),DELTA);
    compute_sigma_omega(lanc,Hmat,Jxmat,eval,evec,omega,Lambda);

    std::ofstream sigmaout;
    std::string filename = "sigma_"+tag+".out";
    if (SigmaOut_on) sigmaout.open(filename);

    for(int i=0;i<NOMEGA;i++){
        sigma[i] = -1.0/(M_PI*omega[i].real())*Lambda[i].imag();
        if(SigmaOut_on) 
            sigmaout<<omega[i].real()<<"\t"<<sigma[i]/VOL<<std::endl;
    }
	if(SigmaOut_on){ 
        sigmaout<<"\n"<<std::endl;
        sigmaout.close();
    }

    double Vertex = 0.0;
    for(int i=0;i<NOMEGA;i++){
            if(omega[i].real() > OMEGA_CUTOFF) Vertex += sigma[i]*OMEGA_STEP;
    }

    Kxmat.Hubbard_mv(evec,KV);
    double Kinetic = lanc.thrust_Dotprod(evec,KV);

    drude_weight_xx = (-Kinetic-2.0*Vertex)/VOL;

	KV.clear(); thrust::device_vector<double>().swap(KV);
}

void KagomeLattice24(H_TBq& H, const double * rand)
{
    int first[48] = {0,1,6,8,15,16,21,23,3,4,9,11,12,13,18,20,
        3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22,
        1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23};
    int second[48] = {1,6,8,15,16,21,23,0,4,9,11,12,13,18,20,3,
        2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3,
        2,4,5,1,8,10,11,7,14,16,17,13,20,22,23,19};

    for (int i=0; i<48; ++i)
    {
        H.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }
}

void KagomeKinetic24(H_TBq& Kx, H_TBq& Ky, const double * rand)
{
    int firstX[48] = {0,1,6,8,15,16,21,23,3,4,9,11,12,13,18,20, // A->B dir
                      3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22, // A->C dir
                      1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23}; // C->B dir
    int secondX[48] = {1,6,8,15,16,21,23,0,4,9,11,12,13,18,20,3,
                       2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3,
                       2,4,5,1,8,10,11,7,14,16,17,13,20,22,23,19};
    double ProjX[48] = {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                        0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                        0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25};

    int firstY[32] = {3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22, // A->C dir
                      2,4,5,1,8,10,11,7,14,16,17,13,20,22,23,19}; // B->C dir
    int secondY[32] = {2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3,
                       1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23};
    double Py = 3/4.;
    double ProjY[32] = {Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,
                        Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py};

    for (int i=0; i<48; ++i)
    {
        Kx.Make_H(firstX[i],secondX[i],-rand[firstX[i]]*rand[secondX[i]]*ProjX[i]);
    }

    for (int i=0; i<32; ++i) 
    {
        Ky.Make_H(firstY[i],secondY[i],-rand[firstY[i]]*rand[secondY[i]]*ProjY[i]);
    }
}

void KagomeCurrent24(H_TBq& Jx, H_TBq& Jy, const double * rand)
{
    int firstX[48] = {0,1,6,8,15,16,21,23,3,4,9,11,12,13,18,20, // A->B dir
                      3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22, // A->C dir
                      1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23}; // C->B dir
    int secondX[48] = {1,6,8,15,16,21,23,0,4,9,11,12,13,18,20,3,
                       2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3,
                       2,4,5,1,8,10,11,7,14,16,17,13,20,22,23,19};
    double ProjX[48] = {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};

    int firstY[32] = {3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22, // A->C dir
                      2,4,5,1,8,10,11,7,14,16,17,13,20,22,23,19}; // B->C dir
    int secondY[32] = {2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3,
                       1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23};
    double Py = sqrt(3)/2.;
    double ProjY[32] = {Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,
                        Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py};

    for (int i=0; i<48; ++i)
    {
        Jx.Make_J(firstX[i],secondX[i],rand[firstX[i]]*rand[secondX[i]]*ProjX[i]);
    }

    for (int i=0; i<32; ++i)
    {
        Jy.Make_J(firstY[i],secondY[i],rand[firstY[i]]*rand[secondY[i]]*ProjY[i]);
    }
}

void KagomeLattice18(H_TBq& H, const double * rand)
{
    int first[36] = {0,1,8,9,15,17,3,4,11,6,12,14,
        3,2,8,7,12,13,0,5,11,10,15,16,1,2,4,5,6,7,9,10,13,14,16,17};
    int second[36] = {1,8,9,15,17,0,4,11,6,12,14,3,
        2,8,7,12,13,0,5,11,10,15,16,3,2,4,5,1,7,9,10,6,14,16,17,13};

    for (int i=0; i<36; ++i)
    {
        H.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }
}

void KagomeKinetic18(H_TBq& Kx, H_TBq& Ky, const double * rand)
{
    int firstX[36] = {0,1,8,9,15,17,3,4,11,6,12,14,
                      3,2,8,7,12,13,0,5,11,10,15,16,
                      1,2,4,5,6,7,9,10,13,14,16,17};
    int secondX[36] = {1,8,9,15,17,0,4,11,6,12,14,3,
                       2,8,7,12,13,0,5,11,10,15,16,3,
                       2,4,5,1,7,9,10,6,14,16,17,13};
    double ProjX[36] = {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                        0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                        0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25};

    int firstY[24] = {3,2,8,7,12,13,0,5,11,10,15,16,
                      2,4,5,1,7,9,10,6,14,16,17,13};
    int secondY[24] = {2,8,7,12,13,0,5,11,10,15,16,3,
                       1,2,4,5,6,7,9,10,13,14,16,17};
    double Py = 3/4.;
    double ProjY[24] = {Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,
                        Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py};

    for (int i=0; i<36; ++i)
    {
        Kx.Make_H(firstX[i],secondX[i],-rand[firstX[i]]*rand[secondX[i]]*ProjX[i]);
    }

    for (int i=0; i<24; ++i)
    {
        Ky.Make_H(firstY[i],secondY[i],-rand[firstY[i]]*rand[secondY[i]]*ProjY[i]);
    }
}

void KagomeCurrent18(H_TBq& Jx, H_TBq& Jy, const double * rand)
{
    int firstX[36] = {0,1,8,9,15,17,3,4,11,6,12,14,
                      3,2,8,7,12,13,0,5,11,10,15,16,
                      1,2,4,5,6,7,9,10,13,14,16,17};
    int secondX[36] = {1,8,9,15,17,0,4,11,6,12,14,3,
                       2,8,7,12,13,0,5,11,10,15,16,3,
                       2,4,5,1,7,9,10,6,14,16,17,13};
    double ProjX[36] = {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};

    int firstY[24] = {3,2,8,7,12,13,0,5,11,10,15,16,
                      2,4,5,1,7,9,10,6,14,16,17,13};
    int secondY[24] = {2,8,7,12,13,0,5,11,10,15,16,3,
                       1,2,4,5,6,7,9,10,13,14,16,17};
    double Py = sqrt(3)/2.;
    double ProjY[24] = {Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,
                        Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py,Py};

    for (int i=0; i<36; ++i)
    {
        Jx.Make_J(firstX[i],secondX[i],rand[firstX[i]]*rand[secondX[i]]*ProjX[i]);
    }

    for (int i=0; i<24; ++i)
    {
        Jy.Make_J(firstY[i],secondY[i],rand[firstY[i]]*rand[secondY[i]]*ProjY[i]);
    }
}

void KagomeLattice12(H_TBq& H, const double * rand)
{
    int first[24] = {0,1,3,4,6,7,9,10,0,2,6,8,3,5,9,11,2,1,8,7,5,4,11,10};
    int second[24] = {1,3,4,0,7,9,10,6,2,6,8,0,5,9,11,3,1,11,7,5,4,8,10,2};

    for (int i=0; i<24; ++i)
    {
        H.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }
}

void KagomeKinetic12(H_TBq& Kx, H_TBq& Ky, const double * rand)
{
    int firstX[24] = {0,1,3,4,6,7,9,10,
                      0,2,6,8,3,5,9,11,
                      2,1,8,7,5,4,11,10};
    int secondX[24] = {1,3,4,0,7,9,10,6,
                       2,6,8,0,5,9,11,3,
                       1,11,7,5,4,8,10,2};
    double ProjX[24] = {1.,1.,1.,1.,1.,1.,1.,1.,
                        0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,
                        0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25};

    int firstY[16] = {0,2,6,8,3,5,9,11,
                      1,11,7,5,4,8,10,2};
    int secondY[16] = {2,6,8,0,5,9,11,3,
                       2,1,8,7,5,4,11,10};
    double Py = 3/4.;
    double ProjY[16] = {Py,Py,Py,Py,Py,Py,Py,Py,
                        Py,Py,Py,Py,Py,Py,Py,Py}; 

    for (int i=0; i<24; ++i)
    {
        Kx.Make_H(firstX[i],secondX[i],-rand[firstX[i]]*rand[secondX[i]]*ProjX[i]);
    }

    for (int i=0; i<16; ++i)
    {
        Ky.Make_H(firstY[i],secondY[i],-rand[firstY[i]]*rand[secondY[i]]*ProjY[i]);
    }
}

void KagomeCurrent12(H_TBq& Jx, H_TBq& Jy, const double * rand)
{
    int firstX[24] = {0,1,3,4,6,7,9,10,
                      0,2,6,8,3,5,9,11,
                      2,1,8,7,5,4,11,10};
    int secondX[24] = {1,3,4,0,7,9,10,6,
                       2,6,8,0,5,9,11,3,
                       1,11,7,5,4,8,10,2};
    double ProjX[24] = {1.,1.,1.,1.,1.,1.,1.,1.,
                        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};

    int firstY[16] = {0,2,6,8,3,5,9,11,
                      1,11,7,5,4,8,10,2};
    int secondY[16] = {2,6,8,0,5,9,11,3,
                       2,1,8,7,5,4,11,10};
    double Py = sqrt(3)/2.;
    double ProjY[16] = {Py,Py,Py,Py,Py,Py,Py,Py,
                        Py,Py,Py,Py,Py,Py,Py,Py}; 

    for (int i=0; i<24; ++i)
    {
        Jx.Make_J(firstX[i],secondX[i],rand[firstX[i]]*rand[secondX[i]]*ProjX[i]);
    }

    for (int i=0; i<16; ++i)
    {
        Jy.Make_J(firstY[i],secondY[i],rand[firstY[i]]*rand[secondY[i]]*ProjY[i]);
    }
}





