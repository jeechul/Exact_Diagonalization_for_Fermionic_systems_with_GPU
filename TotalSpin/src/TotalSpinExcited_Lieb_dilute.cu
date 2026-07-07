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
#include "../../MBL_test/include/argparse.h"

//#define EIGEN_NO_CUDA
#include <Eigen/Core>
//#include <Spectra/SymEigsSolver.h>
#include "../include/SymEigsSolver.cuh"

//#define __OUTFILE__

const int N = 12;

void LiebLattice12(H_TBq& Hup, H_TBq& Hdn);
void LiebLattice18(H_TBq& Hup, H_TBq& Hdn);
void LiebLattice24(H_TBq& Hup, H_TBq& Hdn);

using ModelType = cuSpectra::Hubbard<double>;
using OpType = TotalSpin::Hubbard;
using SolverType = cuSpectra::SymEigsSolver<ModelType>;

int main(int argc, char * argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction"));
    options.push_back(pair_t("Q", "# of electrons"));
    options.push_back(pair_t("X", "# of diluted sites"));
    options.push_back(pair_t("Nsam", "# of disorder samples"));
    options.push_back(pair_t("dev", "device number"));
    options.push_back(pair_t("nev", "# of eigen pairs"));
    options.push_back(pair_t("path", "directory to load and save files"));
   	// env; default value
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("Nsam", "1"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    unsigned int BitSetdl; // bitset containing diluted-sites information
    const int Nsam = parser.find<int>("Nsam"),
        nev = parser.find<int>("nev"),
        dev = parser.find<int>("dev"),
        X = parser.find<int>("X");
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

    THRUST::Thrust_Lin<double> LinAlg;

    H_TBq HqUP(N,Qup), HqDN(N,Qdn);
    LiebLattice12(HqUP,HqDN);
   
    OpType Smat(HqUP,HqDN);

    const int dimUP = HqUP.count, dimDN = HqDN.count;
    std::cout<<"# Reduced Hilbert-space dim : "<<dimUP*dimDN<<std::endl;

    const std::string filename = parser.find<>("path") + "/TotalSpinExcited_Lieb-L" + std::to_string(N) + "U" + parser.find<>("U") 
        + "Q" + parser.find<>("Q")
        + "X" + parser.find<>("X") + ".dat";
#ifdef __OUTFILE__
    std::ofstream outfile;
    if (!std::ifstream(filename).is_open())
    {
        outfile.open(filename); 
        outfile << "#        INT_dilutes     BitSet_dilutes      seq{ E , (S^+S^- + S^-S^+)/2 , S(S+1) }   " << std::endl;
    }
    else
        outfile.open(filename,std::ios::app);
    outfile.precision(10);
#else 
    std::cout << "#       INT_dilutes     BitSet_dilutes       seq{ E , (S^+S^- + S^-S^+)/2 , S(S+1) }    " << std::endl;
    std::cout.precision(10);
#endif
    for (int sam=0; sam<Nsam; ++sam)
    {
        std::shuffle(randomSites.begin(),randomSites.end(),generator);
        BitSetdl = 0;
        for (int i=0;i<(N-X);++i)
            BitSetdl += 1<<randomSites[i];

        ModelType Hmat(HqUP,HqDN,U,BitSetdl);

        SolverType eigs(Hmat,nev,4*(nev+1)); 
        eigs.init();
        eigs.compute(Spectra::SortRule::SmallestAlge,1000,1e-10,Spectra::SortRule::SmallestAlge);

        Eigen::VectorXd eval;
        Eigen::MatrixXd evec;
        if (eigs.info() == Spectra::CompInfo::Successful) {
            eval = eigs.eigenvalues();
            evec = eigs.eigenvectors();
            // std::cout << "#    Eigenvalues : " << eval.transpose() << std::endl;
        }

        double Sxy[nev], Sop2[nev];
        for (int i=0; i<nev; ++i)
        {
            thrust::device_vector<double> V(dimUP*dimDN), f(dimUP*dimDN);
            Eigen::VectorXd vec = evec.col(i);
            thrust::copy(vec.data(),vec.data()+dimUP*dimDN,V.begin());

            Smat.Hubbard_mv(V,f);
            Sxy[i] = LinAlg.thrust_Dotprod(V,f); 
            Sop2[i] = 0.25*(Qup-Qdn)*(Qup-Qdn)+Sxy[i];
        }
#ifdef __OUTFILE__
        outfile << "\t" << BitSetdl << "\t" << std::bitset<N>(BitSetdl); 
        for (int i=0; i<nev; ++i) 
            outfile << "\t" << eval[i] << "\t" << Sxy[i] << "\t" << Sop2[i];
        outfile << std::endl;
        std::cout << "\r# ---- " << std::setw(4) << sam+1 << "/" << std::setw(4) << Nsam << std::flush;
#else
        std::cout << "\t" << BitSetdl << "\t" << std::bitset<N>(BitSetdl); 
        for (int i=0; i<nev; ++i)
            std::cout << "\t" << eval[i] << "\t" << Sxy[i] << "\t" << Sop2[i];
        std::cout << std::endl;
#endif
    }
#ifdef __OUTFILE__
    outfile.close();
    std::cout << std::endl;
#endif
    return 0;
}

void LiebLattice12(H_TBq& Hup, H_TBq& Hdn)
{
    int first[16] = {0,1,3,4,6,7,9,10,0,2,6,8,3,5,9,11};
    int second[16] = {1,3,4,0,7,9,10,6,2,6,8,0,5,9,11,3};

    for (int i=0; i<16; ++i)
    {
        double param = -1.0;
        //if (i == 3 || i == 7) param = 1.0;
        Hup.Make_H(first[i],second[i],param);
        Hdn.Make_H(first[i],second[i],param);
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

void LiebLattice24(H_TBq& Hup, H_TBq& Hdn)
{
    int first[32] = {0,1,6,8,15,16,21,23,3,4,9,11,12,13,18,20,
        3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22};
    int second[32] = {1,6,8,15,16,21,23,0,4,9,11,12,13,18,20,3,
        2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3};

    for (int i=0; i<32; ++i)
    {
        Hup.Make_H(first[i],second[i],-1.0);
        Hdn.Make_H(first[i],second[i],-1.0);
    }
}


