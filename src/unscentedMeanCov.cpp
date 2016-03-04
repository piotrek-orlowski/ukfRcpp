#include <RcppArmadillo.h>
using namespace std;

//' @export
//' @useDynLib ukfRcpp
// [[Rcpp::export]]
arma::mat unscentedMean(const arma::mat xSigma, const arma::vec unscWts){
//   xSigma.t().print("unscMean, transpose of xSigma");
//   unscWts.t().print("unscMean, transpose of unscWts");
  arma::mat retMean(xSigma.n_rows,1,arma::fill::zeros);
  
  for(unsigned int xrow = 0; xrow < xSigma.n_rows; xrow++){
    retMean(xrow) = arma::as_scalar(arma::sum(xSigma.row(xrow) % unscWts.t()));
  }
  
  return retMean;
}

//' @export
// [[Rcpp::export]]
arma::mat unscentedCov(const arma::mat xSigma, const arma::vec unscWtsMean, const arma::vec unscWtsCov){
  
  // calculate mean
  arma::mat xMean = unscentedMean(xSigma, unscWtsMean);
  
  // initialise covariance matrix
  arma::mat retVar(xSigma.n_rows,xSigma.n_rows,arma::fill::zeros);

  // lOOp
  for(unsigned int lnum = 0; lnum < unscWtsCov.n_elem; lnum++){
    retVar += unscWtsCov(lnum) * (xSigma.col(lnum) - xMean) * (xSigma.col(lnum) - xMean).t();
  }
  
  return retVar;
  
}

//' @export
// [[Rcpp::export]]
arma::mat unscentedCrossCov(const arma::mat xSigma, const arma::mat ySigma, const arma::vec unscWtsMean, const arma::vec unscWtsCov){
  
  // calculate means
  arma::mat xMean = unscentedMean(xSigma, unscWtsMean);
  arma::mat yMean = unscentedMean(ySigma, unscWtsMean);
  
  // initialise covariance matrix
  arma::mat retVar(xSigma.n_rows,ySigma.n_rows,arma::fill::zeros);
  
  // lOOp
  for(unsigned int lnum = 0; lnum < unscWtsCov.n_elem; lnum++){
    retVar += unscWtsCov(lnum) * (xSigma.col(lnum) - xMean) * (ySigma.col(lnum) - yMean).t();
  }
  
  return retVar;
  
}
