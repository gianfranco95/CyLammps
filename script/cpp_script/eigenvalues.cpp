# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <cstring>
# include "fstream"
# include "string"
# include <armadillo>

using namespace std;
int main ()
{
    int Nrow = 69120000; //1728000000;
    double **HM = new double*[Nrow];
    for (int i = 0; i < Nrow; ++i){
        HM[i] = new double[3];
    }
    int NSrow = 14400; // 72000;
    //std::string DirMatFile = "/home/gianfranco/film_trimeri/box_200.12.9_stat/deform/run33/";
    std::string DirMatFile = "/home/gianfranco/film_trimeri/box_small/deform/run1/";
    std::string Defo = "0%";
    std::string MatFile = DirMatFile + "DyMat_" + Defo + ".dat";
    std::ifstream HMData(MatFile);

    using namespace arma;
    arma::mat SH = randu<mat>(NSrow,NSrow);
    for (int i = 0; i < Nrow; i++){
        for (int j = 0; j < 3; j++){
        HMData >> HM[i][j];
        }
    }

    int Ns = 4800; //24000;
    int beta = 3;
    int alpha = 3;
    for ( int i_s = 0; i_s < Ns; i_s++)
    {
        for ( int i_alpha = 0; i_alpha < alpha; i_alpha++)
        {
            for ( int j_s = 0; j_s < Ns; j_s++)
            {
                for ( int j_beta = 0; j_beta < beta; j_beta++)
                {
                    SH(alpha*i_s+i_alpha,beta*j_s+j_beta) = HM[NSrow*i_s+Ns*i_alpha+j_s][j_beta];
                }
            }
        }
    }
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec,SH,"dc");
    std::string EigenFile = DirMatFile + "Eigen_" + Defo + ".dat";
    std::string EigenVectorFile = DirMatFile + "EigenVector_" + Defo + ".dat";
    std::ofstream SaveEigen, SaveEigenVector;
    SaveEigen.open(EigenFile);
    SaveEigenVector.open(EigenVectorFile);
    for ( int i = 0; i < NSrow; i++ )
    {
        SaveEigen << eigval(i);
        SaveEigen << "\n";
        for ( int j = 0; j < NSrow; j++ ) {
          SaveEigenVector << eigvec(i,j);
          SaveEigenVector << "\t";
        }
        SaveEigenVector << "\n";
    }
    SaveEigen.close();
    SaveEigenVector.close();


    return 0;
}
