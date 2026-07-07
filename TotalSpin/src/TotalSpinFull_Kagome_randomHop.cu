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

//#define __OUTFILE__

const int N = 12;

void KagomeLattice12(H_TBq& Hup, H_TBq& Hdn, const double * rand);
void KagomeLattice18(H_TBq& Hup, H_TBq& Hdn, const double * rand);
void KagomeLattice24(H_TBq& Hup, H_TBq& Hdn, const double * rand);

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

    using ModelType = Hubbard;
    using OpType = TotalSpin::Hubbard;
    THRUST::Thrust_Lin<double> LinAlg;

    H_TBq HqUP(N,Qup), HqDN(N,Qdn);

    const int dimUP = HqUP.count, dimDN = HqDN.count;
    std::cout<<"# Reduced Hilbert-space dim : "<<dimUP*dimDN<<std::endl;
    thrust::device_vector<double> PsiT(dimUP*dimDN*dimUP*dimDN,0), E(dimUP*dimDN,0);

    const std::string filename = parser.find<>("path") + "/TotalSpinExcited_Kagome-L" + std::to_string(N) + "U" + parser.find<>("U") 
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
        randNumfilename = parser.find<>("xpath") + "/randNum_Kagome-L" + std::to_string(N) + ".dat";
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
            for (int i=0;i<N;++i)
            {
                double rand;
                iss >> rand;
                randomSites[i] = 1.0+X*rand;
            }
        } else {
            for (int i=0;i<N;++i)
                randomSites[i] = 1.0+X*dist(generator);
        }

        KagomeLattice12(HqUP,HqDN,&randomSites[0]);
        ModelType Hmat(HqUP,HqDN,U);
        OpType Smat(HqUP,HqDN);

        Hmat.Write(PsiT);
        cusolverDsyevd(dimUP*dimDN,thrust::raw_pointer_cast(PsiT.data()),thrust::raw_pointer_cast(E.data()));
        thrust::host_vector<double> eval = E;

        double Sxy[nev], Sop2[nev];
        for (int i=0; i<nev; ++i)
        {
            thrust::device_vector<double> V(dimUP*dimDN), f(dimUP*dimDN);
            thrust::copy(PsiT.begin()+i*dimUP*dimDN,PsiT.begin()+(i+1)*dimUP*dimDN,V.begin());

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
        PsiT.assign(dimUP*dimDN*dimUP*dimDN,0.0);
    }
#ifdef __OUTFILE__
    outfile.close();
    std::cout << std::endl;
#endif
    return 0;
}

void KagomeLattice12(H_TBq& Hup, H_TBq& Hdn, const double * rand)
{
    int first[24] = {0,1,3,4,6,7,9,10,0,2,6,8,3,5,9,11,2,1,8,7,5,4,11,10};
    int second[24] = {1,3,4,0,7,9,10,6,2,6,8,0,5,9,11,3,1,11,7,5,4,8,10,2};

    for (int i=0; i<24; ++i)
    {
        Hup.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
        Hdn.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }

    for (int i=0; i<12; ++i)
    {
        Hup.Make_N(i,-2.0*rand[i]*rand[i]);
        Hdn.Make_N(i,-2.0*rand[i]*rand[i]);
    }
}

void KagomeLattice18(H_TBq& Hup, H_TBq& Hdn, const double * rand)
{
    int first[36] = {0,1,8,9,15,17,3,4,11,6,12,14,
        3,2,8,7,12,13,0,5,11,10,15,16,1,2,4,5,6,7,9,10,13,14,16,17};
    int second[36] = {1,8,9,15,17,0,4,11,6,12,14,3,
        2,8,7,12,13,0,5,11,10,15,16,3,2,4,5,1,7,9,10,6,14,16,17,13};

    for (int i=0; i<36; ++i)
    {
        Hup.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
        Hdn.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }

    for (int i=0; i<18; ++i)
    {
        Hup.Make_N(i,-2.0*rand[i]*rand[i]);
        Hdn.Make_N(i,-2.0*rand[i]*rand[i]);
    }
}

void KagomeLattice24(H_TBq& Hup, H_TBq& Hdn, const double * rand)
{
    int first[48] = {0,1,6,8,15,16,21,23,3,4,9,11,12,13,18,20,
        3,2,6,7,0,5,9,10,15,14,18,19,12,17,21,22,
        1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23};
    int second[48] = {1,6,8,15,16,21,23,0,4,9,11,12,13,18,20,3,
        2,6,7,12,5,9,10,15,14,18,19,0,17,21,22,3,
        2,4,5,1,8,10,11,7,14,16,17,13,20,22,23,19};

    for (int i=0; i<48; ++i)
    {
        Hup.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
        Hdn.Make_H(first[i],second[i],-rand[first[i]]*rand[second[i]]);
    }

    for (int i=0; i<24; ++i)
    {
        Hup.Make_N(i,-2.0*rand[i]*rand[i]);
        Hdn.Make_N(i,-2.0*rand[i]*rand[i]);
    }
}


