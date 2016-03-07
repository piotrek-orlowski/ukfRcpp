#ifndef UKFUTILS_H
#define UKFUTILS_H

#include <RcppArmadillo.h>
using namespace std;

arma::mat unscentedMean(const arma::mat xSigma, const arma::vec unscWts);

arma::mat unscentedCov(const arma::mat xSigma, const arma::vec unscWtsMean, const arma::vec unscWtsCov);

arma::mat unscentedCrossCov(const arma::mat xSigma, const arma::mat ySigma, const arma::vec unscWtsMean, const arma::vec unscWtsCov);

arma::mat generateSigmaPoints(const arma::mat, double gam, const arma::mat, int L);

arma::mat generateSigmaWeights(const int L, const double alpha, const double beta);

#endif