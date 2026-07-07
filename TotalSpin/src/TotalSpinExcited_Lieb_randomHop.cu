#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>
#include <algorithm>
#include <thrust/complex.h>
#include "../../MBL_test/include/HubMatrix_GPU.cuh"
#include "../include/HubMatrix_TotalSpin.cuh"
#include "../../MBL_test/include/argparse.h"

//#define EIGEN_NO_CUDA
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
//#include "../include/SymEigsSolver.cuh"

//#define __OUTFILE__

const int N = 12;

void LiebLattice12(H_TBq& Hup, H_TBq& Hdn, const double * rand);
void LiebLattice18(H_TBq& Hup, H_TBq& Hdn, const double * rand);
void LiebLattice24(H_TBq& Hup, H_TBq& Hdn, const double * rand);

using ModelType = cuSpectra::Hubbard<double>;
using OpType = TotalSpin::Hubbard;
using SolverType = Spectra::SymEigsSolver<ModelType>;

int main(int argc, char * argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction"));
    options.push_back(pair_t("Q", "# of electrons"));
    options.push_back(pair_t("X", "disorder strength (0<= X <1)"));
    options.push_back(pair_t("Nsam", "# of disorder samples"));
    options.push_back(pair_t("dev", "device number"));
    options.push_back(pair_t("nev", "# of eigen pairs"));
    options.push_back(pair_t("path", "directory to load and save files"));
    options.push_back(pair_t("xpath", "directory to load disorders file"));
   	// env; default value
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("Nsam", "1"));
    defaults.push_back(pair_t("path", "."));
    defaults.push_back(pair_t("xpath", "None"));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    const int Nsam = parser.find<int>("Nsam"),
        nev = parser.find<int>("nev"),
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

    THRUST::Thrust_Lin<double> LinAlg;

    H_TBq HqUP(N,Qup), HqDN(N,Qdn);

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
        outfile << "#       seq{ E , (S^+S^- + S^-S^+)/2 , S(S+1) }   " << std::endl;
    }
    else
        outfile.open(filename,std::ios::app);
    outfile.precision(10);
#else 
    std::cout << "#       seq{ E , (S^+S^- + S^-S^+)/2 , S(S+1) }    " << std::endl;
    std::cout.precision(10);
#endif
    std::string randNumfilename;
    std::ifstream randNumfile;
    bool randNumfileFlag = false;
    if (parser.find<>("xpath").compare("None")!=0)
    {
        randNumfilename = parser.find<>("xpath") + "/randNum_Lieb-L" + std::to_string(N) + ".dat";
        randNumfileFlag = true;
        if (!std::ifstream(randNumfilename).is_open())
            std::cout << "There is not the file!" << std::endl;
        else 
            randNumfile.open(randNumfilename);
    }

    for (int sam=0; sam<Nsam; ++sam)
    {
        if (randNumfileFlag)
        {
            std::string randNumLine;
            std::getline(randNumfile,randNumLine);
            std::istringstream iss(randNumLine);

            if (randNumLine[0] != '#')
            {
                for (int i=0;i<N;++i)
                {
                    double rand;
                    iss >> rand;
                    randomSites[i] = 1.0+X*rand;
                }
            }
            else {
                sam--;
                continue;
            }
        } else {
            for (int i=0;i<N;++i)
                randomSites[i] = 1.0+X*dist(generator);
        }

        LiebLattice12(HqUP,HqDN,&randomSites[0]);
        ModelType Hmat(HqUP,HqDN,U);
        OpType Smat(HqUP,HqDN);

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
        for (int i=0; i<nev; ++i) 
            outfile << "\t" << eval[i] << "\t" << Sxy[i] << "\t" << Sop2[i];
        outfile << std::endl;
        std::cout << "\r# ---- " << std::setw(4) << sam+1 << "/" << std::setw(4) << Nsam << std::flush;
#else
        for (int i=0; i<nev; ++i)
            std::cout << "\t" << eval[i] << "\t" << Sxy[i] << "\t" << Sop2[i];
        std::cout << std::endl;
#endif
        HqUP.clear();
        HqDN.clear();
    }
#ifdef __OUTFILE__
    outfile.close();
    std::cout << std::endl;
#endif
    return 0;
}

void LiebLattice12(H_TBq& Hup, H_TBq& Hdn, const double * rand)
{
    int first[16] = {0,1,3,4,6,7,9,10,0,2,6,8,3,5,9,11};
    int second[16] = {1,3,4,0,7,9,10,6,2,6,8,0,5,9,11,3};

    for (int i=0; i<16; ++i)
    {
        double param = -1.0;
        if (i == 3 || i == 7) param = 1.0;
        Hup.Make_H(first[i],second[i],param*rand[first[i]]*rand[second[i]]);
        Hdn.Make_H(first[i],second[i],param*rand[first[i]]*rand[second[i]]);
    }
}

void LiebLattice18(H_TBq& Hup, H_TBq& Hdn, const double * rand)
{
    int first[24] = {0,1,8,9,15,17,3,4,11,6,12,14,
        3,2,8,7,12,13,0,5,11,10,15,16};
    int second[24] = {1,8,9,15,17,0,4,11,6,12,14,3,
        2,8,7,12,13,0,5,11,10,15,16,3};

    for (int i=0; i<24; ++i)
    {
        Hup.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
        Hdn.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }
}

void LiebLattice24(H_TBq& Hup, H_TBq& Hdn, const double * rand)
{
    int first[32] = {0,1,6,8,15,16,21,23,3,4,9,11,12,13,18,20,
        3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22};
    int second[32] = {1,6,8,15,16,21,23,0,4,9,11,12,13,18,20,3,
        2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3};

    for (int i=0; i<32; ++i)
    {
        Hup.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
        Hdn.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }
}


