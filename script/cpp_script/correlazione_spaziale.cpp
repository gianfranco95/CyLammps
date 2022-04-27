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

//VARIABILI GLOBALI ***********************************************************************************
int N = 24000;
int L0 = 200;
int m=1;        //numero per cui dividere defo al variare di L0
//arma::vec defo = arma::linspace(1,100,100);
arma::vec defo ={1,10,20,30,40,50};
arma::vec CONF = L0*defo*200 ;

int Ndefo = defo.size();
int Binx=1;
double temp;

//DEFINIZIONE FUNZIONI  ************************************************************************************ 
void skip(std::istream & is, size_t n , char delim) 
{ 
   size_t i = 0; 
   while ( i++ < n) 
      is.ignore(1000, delim);  
// ignores up to 1000 chars but stops ignoring after delim 
} 
//...............................................................................
arma::mat read_lammps(std::string file){
  std::ifstream data(file);
  arma::mat Coord(N,2);

  //skip 9 righe
  skip(data,9,'\n');

  for (int i=0;i<N;i++){
    data >> temp;
    data >> temp;

    data >> Coord(i,0);
    data >> temp;
    data >> Coord(i,1);
    skip(data,1,'\n');
  }
  return Coord;
}
//...............................................................................

arma::uvec strato_idx(arma::mat coord)
{
  arma::uvec IDX = arma::uvec(N,arma::fill::zeros);
  //A sarebbe X, B sarebbe Z
  arma::vec A = coord.col(0);
  arma::vec B = coord.col(1);
  double xmin = A.min();
  double xmax = A.max();
  int Nx = int(floor((xmax-xmin)/Binx));
  arma::vec maxZ = arma::vec(Nx+1,arma::fill::zeros);

  for(int j=0;j<N;j++){
    int c = int( floor( (A(j)-xmin)/Binx ));
    if (B(j)>maxZ(c)){
      maxZ(c) = B(j);
    }
  }

  for(int j=0;j<N;j++){
    int c = int( floor( (A(j)-xmin)/Binx ));
    
    //INDICE PER STABILIRE A QUALE STRATO APPARTENGA LA PARTICELLA, PARTENDO DA 0 PER LA SUPERFICIE, STRATI SPESSI 2 DIAMETRI
    IDX(j) = int(floor( (maxZ(c)-B(j))/2 )) ; 
  }
    
  return IDX ;
}
//.........................................................................................
//.........................................................................................
//.........................................................................................
//.........................................................................................


int main(int argc, char** argv)
{
  std::string run=argv[1];
  std::string Dir = "run"+ run + "/";
  defo = defo/(m*100) ;

  arma::mat Coord0 = read_lammps(Dir + "lmp_data/Conf_0.dat");
  arma::vec X0 = Coord0.col(0);
  arma::vec Z0 = Coord0.col(1);
  Coord0.reset();

  //for loop al variare della deformazione***********************************
  for (int d=0; d<Ndefo; d++){
    double Lx = L0*(1-defo(d));

    std::string conf_file = Dir + "lmp_data/Conf_" + to_string(int(CONF(d))) + ".dat";
    arma::mat Coord = read_lammps(conf_file);

    arma::uvec idx = strato_idx(Coord);

    arma::vec X=Coord.col(0);
    arma::vec displX= X - (X0*Lx)/L0;
    arma::vec displZ= Coord.col(1) - Z0 ;
    Coord.reset();

    //loop sui 4 strati
    arma::mat Cr = arma::mat(int(Lx),9);  //4 entrate per dz e dx al variare degli strati, piu' prima colonna coordinata x
    Cr.col(0) = arma::linspace(0,int(Lx)-1,int(Lx)) ;

    for (int s =0;s<4;s++){
      arma::vec Cr_ZZstrato = arma::vec(int(Lx),arma::fill::zeros);
      arma::vec Cr_XXstrato = arma::vec(int(Lx),arma::fill::zeros);

      arma::uvec jj =  find(idx==s);
      arma::vec xx = X(jj);
      arma::vec dx = displX(jj);
      arma::vec dz = displZ(jj);
      int Nz = xx.size();
      
      for(int j=0;j<Nz;j++){
        arma::vec Cr_iXX = arma::vec(int(Lx));
        arma::vec Cr_iZZ = arma::vec(int(Lx));

        arma::vec dist = arma::ceil(arma::abs(xx - xx(j))) ;
        arma::vec crXX = dx*dx(j) ;
        arma::vec crZZ = dz*dz(j) ;
        
        for(int r =0;r< int(Lx);r++){
          arma::uvec idxR = find(dist==r);
          if( arma::all(dist!=r)){
            Cr_iXX(r) = 0 ;
            Cr_iZZ(r) = 0 ;
          }
          else{
            Cr_iXX(r) = arma::mean( crXX(idxR) );
            Cr_iZZ(r) = arma::mean( crZZ(idxR) );
          }
        }

        Cr_XXstrato = Cr_XXstrato + Cr_iXX ;
        Cr_ZZstrato = Cr_ZZstrato + Cr_iZZ ;        
      }

      Cr_XXstrato = Cr_XXstrato/Nz ;
      Cr_ZZstrato = Cr_ZZstrato/Nz ;


      Cr.col(2*s+1) = Cr_XXstrato; 
      Cr.col(2*s+2) = Cr_ZZstrato; 

    }

    Cr.save(Dir + "correlazioni/corr_spatial_"+ to_string(int(defo(d)*100))+ "%.dat",arma::raw_ascii);
  }

return 0;
}