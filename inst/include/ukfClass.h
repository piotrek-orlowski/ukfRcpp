#ifndef UKFCLASS_H
#define UKFCLASS_H
#include <RcppArmadillo.h>
#include "../../src/unscentedMeanCov.h"
#include "ukfRcpp.h"
using namespace std;

// Unscented Kalman Filter with additive noise, van der Merwe pp. 108-110. Rather general implementation
class ukfRcpp {
public:
  // Constructor for cpp
  ukfRcpp(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, stateHandler predictState_, stateHandler evaluateState_, Rcpp::List modelingParams_);
  // Constructor for R
  ukfRcpp(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, SEXP predictState_, SEXP evaluateState_, Rcpp::List modelingParams_);
  // methods:
  // return covariance matrices of filtered states
  arma::cube getCovCube();
  
  // return filtered states
  arma::mat getStateMat();
  
  // return log-likelihood
  arma::vec getLogL();
  
  // set filtering parameters
  void setUKFconstants(arma::vec);
  
  // get filtering parameters
  arma::vec getUKFconstants();
  
  // variables
  // data container
  arma::mat dataMat;
  
  // pointers to functions that handle (1) the conditional mean and variance matrix of the latent process, (2) the mapping from the latent process to the observables
  Rcpp::List (*predictState)(arma::mat, Rcpp::List);
  Rcpp::List (*evaluateState)(arma::mat, Rcpp::List);
  
  // function that runs a step of the filter
  void filterAdditiveNoise();
  
private:
  // UKF parameters
  int L;
  double alpha;
  double beta;
  int iterationCounter;
  int sampleSize;
  
  // state containers
  arma::vec constInitProcessState;
  arma::mat constInitProcessCov;
  arma::vec initProcessState;
  arma::vec nextProcessState;
  arma::mat initProcessCov;
  arma::mat nextProcessCov;
  
  // storage
  arma::cube stateCovCube;
  arma::mat stateMat;
  arma::vec logL;
  
  // model parameters (these will be big matrices, let's keep them hidden)
  Rcpp::List transitionParams;
  Rcpp::List observationParams;
  
  // Method to run one step of filter. Stores filtered state and covariance matrix. Updates initProcess and nextProcess variables
  void filterStep();
  void reinitialiseFilter();
};

#endif