#include <algorithm>
#include <RcppArmadillo.h>
using namespace std;

//' @title UKF: sigma points and weights
//' @name ukfPtsWts
//' @description Functions for generating sigma-points from original state and a Cholesky decomposition of the state variance-covariance matrix, and for generating weights for calculating mean and covariance via the Unscented Transformation.
//' @param xMat Original state matrix: each column is a value of state variables. If \code{ncol(xMat)==1}, the matrix is simply expanded with \code{+/- gam * PMatChol[,k]}. If \code{xMat} has more than one column, it's rewritten and an expansion of the first column is concatenated on the right.
//' @param gam Numeric. \code{gamma} parameter for Unscented Transformations.
//' @param PMatChol Cholesky decomposition (preferably lower) of the variance-covariance matrix of \code{xMat[,1]}. If past states are propagated, then the top-left block of \code{PMatChol} contains the decomposition and other entries are \code{0}.
//' @param L Dimension of state expansion. Should be equal to the number of non-zero entries on \code{diag(PMatChol)}
//' @return Matrix of size \code{nrow(xMat) x (ncol(xMat) + 2*L)}.
//' @export
// [[Rcpp::export]]
arma::mat generateSigmaPoints(const arma::mat xMat, const double gam, const arma::mat PMatChol, const int L){
  
  int xNumCols = xMat.n_cols;
  
  // If  xMat has one column, bigXMat is L x 2L+1. If xMat is to be augmented (vdM page 109, 3.174), we need to accomodate 2L additional columns.
  arma::mat bigXMat(xMat.n_rows,2L*L + 1L + 2L*L*std::max(0.0,xNumCols-2.0*L),arma::fill::zeros);
  
  // Assign xMat to first columns of bigXmat
  bigXMat.cols(0L,xMat.n_cols-1L) = xMat;
  
  // Loop and fill remaining columns of bigXMat. Adding: columns 2L+1+1 to 2L+1+L. Subtracting: 2L+1+L+1 to 2L+1+L+L
  for(int ksig=0; ksig < L; ksig++){
    bigXMat.col(xMat.n_cols + ksig) = xMat.col(0) + gam * PMatChol.col(ksig);
    bigXMat.col(xMat.n_cols + L + ksig) = xMat.col(0) - gam * PMatChol.col(ksig);
  }
  
  return bigXMat;
}