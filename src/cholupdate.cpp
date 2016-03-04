#include <RcppArmadillo.h>
using namespace Rcpp;

//' @title Cholesky rank-1 update (downdate)
//' @name cholupdate
//' @description Let RMat be the lower triangular Cholesky factor of PMat. This function calculates the lower-triangular Cholesky factor of P1 = P + nu^(0.5) * xVec \%*\% t(xVec). For downdate take nu < 0.
//' @param RMat square matrix, cholesky decomposition (lower) of matrix A
//' @param xVec numeric vector such that \code{length(xVec) = dim(RMat)[j]}, \code{j=1,2}
//' @param nu real, positive for update, negative for downdate. Mind the square root in description.
//' @details This follows the definition in van der Merwe PhD Thesis ``Sigma-Point Kalma Filters for Probabilistic Inference in Dynamic State-Space Models''
//' @export
// [[Rcpp::export]]
arma::mat cholupdate(arma::mat RMat, arma::vec xVec, double nu){
  
  double nuSign = (nu > 0) - (nu < 0);
  
  // Take fourth root of nu
  nu = pow(abs(nu),0.25);
  
  // Take size of problem
  int Nx = xVec.n_elem;
  
  // Transpose xVec if necessary
  if(xVec.n_cols > xVec.n_rows){
    arma::inplace_trans(xVec);
  }
  
  // scale by nu
  xVec *= nu;
  
  double r, c, s;
  r=0;
  
  // algo
  for(int kk = 0; kk < Nx; kk++){
    if(nuSign == 1.0){
      r = sqrt(pow(RMat(kk,kk),2.0) + pow(xVec(kk),2.0));  
    } else if(nuSign == -1.0){
      r = sqrt(pow(RMat(kk,kk),2.0) - pow(xVec(kk),2.0));
    }
    
    c = r / RMat(kk,kk);
    s = xVec(kk) / RMat(kk,kk);
    RMat(kk,kk) = r;
    if(kk < Nx-1){
      if(nuSign == 1.0){
        RMat.submat(kk+1,kk,Nx-1,kk) = (RMat.submat(kk+1,kk,Nx-1,kk) + s * xVec.subvec(kk+1,Nx-1)) / c;
      } else if(nuSign == -1.0){
        RMat.submat(kk+1,kk,Nx-1,kk) = (RMat.submat(kk+1,kk,Nx-1,kk) - s * xVec.subvec(kk+1,Nx-1)) / c;
      }
      xVec.subvec(kk+1,Nx-1) = c * xVec.subvec(kk+1,Nx-1) - s * RMat.submat(kk+1,kk,Nx-1,kk);
    }
  }
  
  return RMat;
}