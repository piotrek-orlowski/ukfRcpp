#include <RcppArmadillo.h>
#include "../inst/include/ukfClass.h"
using namespace std;

Rcpp::List testTransition(const arma::mat& stateVec, const Rcpp::List& transitionParams, const int iterCount){
  
  Rcpp::List returnList;
  
  arma::mat Amat;
  Amat << 0.9 << 0.0 << arma::endr << 0.0 << 0.6 << arma::endr;
  
  arma::mat noiseMat;
  noiseMat << 0.15 << -0.025 << arma::endr << -0.025 << 0.05 << arma::endr;
  
  arma::mat returnVec;
  
  returnVec = Amat * stateVec;
  
  returnList = Rcpp::List::create(Rcpp::Named("stateVec") = returnVec, Rcpp::Named("procNoiseMat") = noiseMat);
  
  return returnList;
}

Rcpp::List testObservation(const arma::mat& stateVec, const Rcpp::List& observationParams, const int iterCount){
  
  Rcpp::List returnList;
  
  arma::mat obsNoiseMat;
  obsNoiseMat << 0.075 << 0.0 << arma::endr << 0.0 << 0.01 << arma::endr;
  
  arma::mat returnMat;
  returnMat = arma::sign(stateVec) % arma::pow(arma::abs(stateVec),2.25);
  
  returnList = Rcpp::List::create(Rcpp::Named("yhat") = returnMat, Rcpp::Named("obsNoiseMat") = obsNoiseMat);
  
  return returnList;
  
}

arma::mat testCheck(arma::mat stateVec){
  return stateVec;
}

//' @title Filtering application example
//' @name filtTests
//' @description These functions are provided as (1) a simple testing ground for the filter, (2) as a demo of how to write the Transition and Observation handler functions.
//' @param Nlength Integer. Length of generated sample for filtering.
//' @return List with true and estimated states, and the state covariance matrices.
//' @details The states follow an AR(1) model with correlated errors. All paramters are hard-coded in this example but can be passed in despective \code{Rcpp::List} objects. The observation equations are signed powers of the states, observed with additional correlated error. \code{testUKFclass} runs the standard-form filter. \code{testSqrtUKFclass} runs the square-root form (better numerical stability).
//' @export
// [[Rcpp::export]]
Rcpp::List testUKFclass(int Nlength){
  
  // Generate objects for filter initialisation. You need a data matrix, initial
  // state, initial state covariance matrix, two handling functions, and a list
  // with parameters for the state handling functions.
  
  // This test-demo function first generates states from a linear autoregressive
  // model, then transforms the states non-linearly into observations and adds
  // observation noise. 
  
  Rcpp::RNGScope rng_scope_;
  
  arma::mat driverProcess(Nlength,2,arma::fill::randn);
  arma::mat obsNoise(Nlength,2,arma::fill::randn);
  arma::mat noiseMat;
  noiseMat << 0.15 << -0.025 << arma::endr << -0.025 << 0.05 << arma::endr;
  arma::mat obsNoiseMat;
  obsNoiseMat << 0.075 << 0.0 << arma::endr << 0.0 << 0.01 << arma::endr;
  
  driverProcess = driverProcess * arma::chol(noiseMat,"lower");
  obsNoise = obsNoise * arma::chol(obsNoiseMat,"lower");
  
  arma::mat realCovDriver = arma::cov(driverProcess);

  arma::mat realCovObsNoise = arma::cov(obsNoise);

  arma::vec initState(2,arma::fill::zeros);
  arma::mat initCov = noiseMat;
  
  arma::mat Amat;
  Amat << 0.9 << 0.0 << arma::endr << 0.0 << 0.6 << arma::endr;
  
  arma::mat trueStateMat(driverProcess.n_rows+1,2,arma::fill::zeros);
  for(unsigned int ii=1; ii <= driverProcess.n_rows; ii++){
    trueStateMat.row(ii) = (Amat * trueStateMat.row(ii-1).t() + driverProcess.row(ii-1).t()).t();
  }
  
  Rcpp::List fakeParameterList = Rcpp::List::create(Rcpp::Named("transition") = arma::zeros(1), Rcpp::Named("observation") = arma::zeros(1));
  
  Rcpp::List observations = testObservation(trueStateMat.rows(1,trueStateMat.n_rows-1), fakeParameterList, 1L);
  
  arma::mat observationsMat = Rcpp::as<arma::mat>(observations["yhat"]);
  observationsMat += obsNoise;
  
  Rcpp::List res;
  
  stateHandler transitionPtr = &testTransition;
  stateHandler observationPtr = &testObservation;
  stateControl controlPtr = &testCheck;
  
  // ukfClass myFirstFilter;
  
  ukfClass myFirstFilter(observationsMat, initState, initCov, transitionPtr, observationPtr, controlPtr, fakeParameterList);
  
  myFirstFilter.filterAdditiveNoise();

  res = Rcpp::List::create(Rcpp::Named("trueState") = trueStateMat, Rcpp::Named("observations") = observationsMat, Rcpp::Named("estimState") = myFirstFilter.getStateMat(), Rcpp::Named("stateCovCube") = myFirstFilter.getCovCube(), Rcpp::Named("logL") = myFirstFilter.getLogL()) ;
  
  return res;
}

//' @rdname filtTests
//' @export
// [[Rcpp::export]]
Rcpp::List testSqrtUKFclass(int Nlength){
  
  // Generate objects for filter initialisation. You need a data matrix, initial
  // state, initial state covariance matrix, two handling functions, and a list
  // with parameters for the state handling functions.
  
  // This test-demo function first generates states from a linear autoregressive
  // model, then transforms the states non-linearly into observations and adds
  // observation noise. 
  
  Rcpp::RNGScope rng_scope_;
  
  arma::mat driverProcess(Nlength,2,arma::fill::randn);
  arma::mat obsNoise(Nlength,2,arma::fill::randn);
  arma::mat noiseMat;
  noiseMat << 0.15 << -0.025 << arma::endr << -0.025 << 0.05 << arma::endr;
  arma::mat obsNoiseMat;
  obsNoiseMat << 0.075 << 0.0 << arma::endr << 0.0 << 0.01 << arma::endr;
  
  driverProcess = driverProcess * arma::chol(noiseMat,"lower");
  obsNoise = obsNoise * arma::chol(obsNoiseMat,"lower");
  
  arma::mat realCovDriver = arma::cov(driverProcess);
  
  arma::mat realCovObsNoise = arma::cov(obsNoise);
  
  arma::vec initState(2,arma::fill::zeros);
  arma::mat initCov = noiseMat;
  
  arma::mat Amat;
  Amat << 0.9 << 0.0 << arma::endr << 0.0 << 0.6 << arma::endr;
  
  arma::mat trueStateMat(driverProcess.n_rows+1,2,arma::fill::zeros);
  for(unsigned int ii=1; ii <= driverProcess.n_rows; ii++){
    trueStateMat.row(ii) = (Amat * trueStateMat.row(ii-1).t() + driverProcess.row(ii-1).t()).t();
  }
  
  Rcpp::List fakeParameterList = Rcpp::List::create(Rcpp::Named("transition") = arma::zeros(1), Rcpp::Named("observation") = arma::zeros(1));
  
  Rcpp::List observations = testObservation(trueStateMat.rows(1,trueStateMat.n_rows-1), fakeParameterList, 1L);
  
  arma::mat observationsMat = Rcpp::as<arma::mat>(observations["yhat"]);
  observationsMat += obsNoise;
  
  Rcpp::List res;
  
  stateHandler transitionPtr = &testTransition;
  stateHandler observationPtr = &testObservation;
  stateControl controlPtr = &testCheck;
  
  // ukfClass myFirstFilter;
  
  ukfClass myFirstFilter(observationsMat, initState, initCov, transitionPtr, observationPtr, controlPtr, fakeParameterList);
  
  myFirstFilter.filterSqrtAdditiveNoise();
  
  res = Rcpp::List::create(Rcpp::Named("trueState") = trueStateMat, Rcpp::Named("observations") = observationsMat, Rcpp::Named("estimState") = myFirstFilter.getStateMat(), Rcpp::Named("stateCovCube") = myFirstFilter.getCovCube(), Rcpp::Named("logL") = myFirstFilter.getLogL()) ;
  
  return res;
}