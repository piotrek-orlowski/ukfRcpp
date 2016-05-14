#ifndef UKFCLASS_H
#define UKFCLASS_H
#include <RcppArmadillo.h>
#include "unscentedMeanCov.h"
#include "ukfRcpp.h"
using namespace std;

// Unscented Kalman Filter with additive noise, van der Merwe pp. 108-110. Rather general implementation
class ukfClass {
public:
  // Constructor for cpp
  ukfClass(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, stateHandler predictState_, stateHandler evaluateState_, stateControl stateController_, Rcpp::List modelingParams_);
  // Constructor for R
  ukfClass(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, SEXP predictState_, SEXP evaluateState_, Rcpp::List modelingParams_);
  // methods:
  // return covariance matrices of filtered states
  arma::cube getCovCube();
  
  // return filtered states
  arma::mat getStateMat();
  
  // calculate and return log-likelihood
  arma::vec getLogL();
  
  arma::mat getPredMat();
  
  arma::mat getFitMat();
  
  // set filtering parameters
  void setUKFconstants(arma::vec);
  
  // get filtering parameters
  arma::vec getUKFconstants();
  
  // variables
  // data container
  arma::mat dataMat;
  
  // pointers to functions that handle (1) the conditional mean and variance matrix of the latent process, (2) the mapping from the latent process to the observables
  Rcpp::List (*predictState)(const arma::mat&, const Rcpp::List&, const int);
  Rcpp::List (*evaluateState)(const arma::mat&, const Rcpp::List&, const int);
  arma::mat (*stateController)(arma::mat);
  
  // function that runs a step of the filter
  void filterAdditiveNoise();
  void filterSqrtAdditiveNoise();
  void reinitialiseFilter();
  
private:
  // UKF parameters
  int L;
  double alpha;
  double beta;
  int iterationCounter;
  int sampleSize;
  
  // state containers
  arma::vec initProcessState;
  arma::vec nextProcessState;
  arma::mat initProcessCov;
  arma::mat nextProcessCov;
  arma::vec constInitProcessState;
  arma::mat constInitProcessCov;
  arma::mat procCovChol;
  
  // storage
  arma::cube stateCovCube;
  arma::mat stateMat;
  arma::vec logL;
  arma::mat predMat;
  arma::mat fitMat;
  
  // model parameters (these will be big matrices, let's keep them hidden)
  Rcpp::List transitionParams;
  Rcpp::List observationParams;
  
  // Method to run one step of filter. Stores filtered state and covariance matrix. Updates initProcess and nextProcess variables
  void filterStep();
  void filterSqrtStep();
};

#endif