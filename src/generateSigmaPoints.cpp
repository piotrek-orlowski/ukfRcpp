#include <algorithm>
#include <RcppArmadillo.h>
using namespace std;


// xMat has as L rows (number of states); PMatChol is LxL. 
//' @export
// [[Rcpp::export]]
arma::mat generateSigmaPoints(const arma::mat xMat, const double gam, const arma::mat PMatChol){
  
  int L = PMatChol.n_cols;
  int xNumCols = xMat.n_cols;
  
  // If  xMat has one column, bigXMat is L x 2L+1. If xMat is to be augmented (vdM page 109, 3.174), we need to accomodate 2L additional columns.
  arma::mat bigXMat(xMat.n_rows,2L*xMat.n_rows + 1L + 2L*L*std::max(0.0,xNumCols-2.0*L),arma::fill::zeros);
  
  // Assign xMat to first columns of bigXmat
  bigXMat.cols(0L,xMat.n_cols-1L) = xMat;
  
  // Loop and fill remaining columns of bigXMat. Adding: columns 2L+1+1 to 2L+1+L. Subtracting: 2L+1+L+1 to 2L+1+L+L
  for(int ksig=0; ksig < L; ksig++){
    bigXMat.col(xMat.n_cols + ksig) = xMat.col(0) + gam * PMatChol.col(ksig);
    bigXMat.col(xMat.n_cols + L + ksig) = xMat.col(0) - gam * PMatChol.col(ksig);
  }
  
  return bigXMat;
}