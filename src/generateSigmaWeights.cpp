#include <RcppArmadillo.h>
using namespace std;

//' @export
// [[Rcpp::export]]
arma::mat generateSigmaWeights(const int L, const double alpha, const double beta){
  
  // Initialise weight matrix; first column are mean weights, second column are variance weights
  arma::mat sigmaWts(2*L+1,2,arma::fill::zeros);
  
  // lambda
  double lambda = std::pow(alpha,2.0) * L - L;
  
  // fill sigmaWts
  sigmaWts.fill(1.0/(2.0*(L + lambda)));
  
  // Zeroth weights are different
  sigmaWts(0,0) = lambda / (lambda + L);
  sigmaWts(0,1) = lambda / (lambda + L) + (1 - std::pow(alpha,2.0) + beta);
  
  return sigmaWts;
}