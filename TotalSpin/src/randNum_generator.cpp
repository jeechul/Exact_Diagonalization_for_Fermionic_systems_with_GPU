#include <iostream>
#include <cmath>
#include <fstream>
#include <random>
#include <algorithm>
#include "../../MBL_test/include/argparse.h"

#define __OUTFILE__

int main(int argc, char * argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("L", "Size of system"));
    options.push_back(pair_t("Nsam", "# of disorder samples"));
    options.push_back(pair_t("path", "directory to load and save files"));
   	// env; default value
    defaults.push_back(pair_t("Nsam", "1"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    const int Nsam = parser.find<int>("Nsam"),
        L = parser.find<int>("L");

  	std::random_device rn;
	std::mt19937 generator(rn());
    std::uniform_real_distribution<double> dist(-1.0,1.0);

    const std::string filename = parser.find<>("path") + "/randNum_Kagome-L" + std::to_string(L) + ".dat";
#ifdef __OUTFILE__
    std::ofstream outfile;
    if (!std::ifstream(filename).is_open())
    {
        outfile.open(filename); 
        outfile << "#       X_1        X_2     ....    " << std::endl;
    }
    else
        outfile.open(filename,std::ios::app);
    outfile.precision(10);
#else 
    std::cout << "#       X_1        X_2      ....     " << std::endl;
    std::cout.precision(10);
#endif
    for (int sam=0; sam<Nsam; ++sam)
    {
        for (int i=0; i<L; ++i)
        {
            outfile << dist(generator) << "\t";
        }
        outfile << std::endl;
    }

    return 0;
}
