#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>
#include <bitset>
#include <algorithm>
#include <thrust/complex.h>
#include "../../MBL_test/include/HubMatrix_GPU.cuh"
#include "../include/HubMatrix_TotalSpin.cuh"
#include "../../MBL_test/include/argparse.hpp"

//#define __OUTFILE__

const int N = 12;

void LiebLattice12(H_TBq& Hup, H_TBq& Hdn);
void LiebLattice18(H_TBq& Hup, H_TBq& Hdn);

void LiebLattice3(H_TBq& Hup, H_TBq& Hdn)
{
    double t = -1.0;

    Hup.Make_H(0,1,t);
    Hup.Make_H(1,0,t);
    Hup.Make_H(0,2,t);
    Hup.Make_H(2,0,t);

    Hdn.Make_H(0,1,t);
    Hdn.Make_H(1,0,t);
    Hdn.Make_H(0,2,t);
    Hdn.Make_H(2,0,t);
}

int main(int argc, char * argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction"));
    options.push_back(pair_t("Q", "# of electrons"));
    options.push_back(pair_t("X", "# of diluted sites"));
    options.push_back(pair_t("Nsam", "# of dilution samples"));
    options.push_back(pair_t("dev", "device number"));
    options.push_back(pair_t("path", "directory to load and save files"));
   	// env; default value
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("Nsam", "1"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    unsigned int BitSetdl; // bitset containing diluted-sites information
    const int X = parser.find<int>("X"),
        Nsam = parser.find<int>("Nsam"),
        dev = parser.find<int>("dev");
    const double U = parser.find<double>("U");
    const auto Q = parser.mfind<int>("Q");
    const int Qup = Q[0], Qdn = Q[1];
    // print info of the registered args
    parser.print(std::cout);

    cudaSetDevice(dev);

	std::random_device rn;
	std::mt19937 generator(rn());

    std::vector<int> randomSites(N);
    for (int i=0;i<N;++i)
        randomSites[i] = i;

    if (Qup+Qdn > 2*N)
    {
        std::cout << "# # of electrons over # of sites!" << std::endl;
        return 0;
    }
    
    using ModelType = Hubbard;
    using OpType = TotalSpin::Hubbard;

    H_TBq HqUP(N,Qup), HqDN(N,Qdn);
    LiebLattice12(HqUP,HqDN);

    OpType Smat(HqUP,HqDN);

    const unsigned long dimUP = HqUP.count, dimDN = HqDN.count;
    std::cout<<"# Reduced Hilbert-space dim : "<<dimUP*dimDN<<std::endl;
    thrust::device_vector<double> f(dimUP*dimDN),V(dimUP*dimDN,0.0);
    double E;

    const std::string filename = parser.find<>("path") + "/ConservedTotalSpin_Lieb-U" + parser.find<>("U") 
        + "Q" + parser.find<>("Q")
        + "X" + parser.find<>("X") + ".dat";
#ifdef __OUTFILE__
    std::ofstream outfile;
    if (!std::ifstream(filename).is_open())
    {
        outfile.open(filename);
        outfile << "#       INT_dilutes     BitSet_dilutes       E       (S^+S^- + S^-S^+)/2       S(S+1)    " << std::endl;
    }
    else
        outfile.open(filename,std::ios::app);
    outfile.precision(10);
#else
    std::cout << "#       INT_dilutes     BitSet_dilutes       E        (S^+S^- + S^-S^+)/2        S(S+1)     " << std::endl;
    std::cout.precision(10);
#endif
    for (int sam=0;sam<Nsam;++sam)
    {
        //std::shuffle(randomSites.begin(),randomSites.end(),generator);
        BitSetdl = 1755;
        //for (int i=0;i<(N-X);++i)
        //    BitSetdl += 1<<randomSites[i];

        ModelType Hmat(HqUP,HqDN,U,BitSetdl);
        Lanczos<double> lanc(dimUP,dimDN);
        lanc.random_vector(f);
        lanc.eigen(Hmat,f,E,V);
       
        Smat.Hubbard_mv(V,f);
        double Sxy = lanc.thrust_Dotprod(V,f); 
        double Sop2 = 0.25*(Qup-Qdn)*(Qup-Qdn)+Sxy;
#ifdef __OUTFILE__
        outfile << "\t" << BitSetdl << "\t" << std::bitset<N>(BitSetdl) << "\t" << E 
            << "\t" << Sxy << "\t" << Sop2 << std::endl;
#else
        std::cout << "\t" << BitSetdl << "\t" << std::bitset<N>(BitSetdl) << "\t" << E 
            << "\t" << Sxy << "\t" << Sop2 << std::endl;
#endif
    }
#ifdef __OUTFILE__
    outfile.close();
#endif
    return 0;
}

void LiebLattice12(H_TBq& Hup, H_TBq& Hdn)
{
    int first[16] = {0,1,3,4,6,7,9,10,0,2,6,8,3,5,9,11};
    int second[16] = {1,3,4,0,7,9,10,6,2,6,8,0,5,9,11,3};
    double params[16] = {-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
                         -1.0,-1.0,-1.0,1.0,-1.0,-1.0,-1.0,1.0};

    for (int i=0; i<16; ++i)
    {
        Hup.Make_H(first[i],second[i],params[i]);
        Hdn.Make_H(first[i],second[i],params[i]);
    }
}

void LiebLattice18(H_TBq& Hup, H_TBq& Hdn)
{
    int first[24] = {0,1,8,9,15,17,3,4,11,6,12,14,
        3,2,8,7,12,13,0,5,11,10,15,16};
    int second[24] = {1,8,9,15,17,0,4,11,6,12,14,3,
        2,8,7,12,13,0,5,11,10,15,16,3};

    for (int i=0; i<24; ++i)
    {
        Hup.Make_H(first[i],second[i],-1.0);
        Hdn.Make_H(first[i],second[i],-1.0);
    }
}

