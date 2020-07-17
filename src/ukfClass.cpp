#include <RcppArmadillo.h>
#include <fstream>
#include "../inst/include/ukfClass.h"
#include "../inst/include/unscentedMeanCov.h"
#include "cholupdate.h"
using namespace std;

// Unscented Kalman Filter with additive noise, van der Merwe pp. 108-110, and the square-root form, pp. 115-116.

// public:
// Constructor for cpp
ukfClass::ukfClass(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, stateHandler predictState_, stateHandler evaluateState_, stateControl stateController_, Rcpp::List modelingParams_) : dataMat(dataMat_), initProcessState(initProcessState_), initProcessCov(initProcessCov_), constInitProcessState(initProcessState_), constInitProcessCov(initProcessCov_), procCovChol(initProcessCov_) {
  
  // State dynamics and observation handling functions
  predictState = predictState_;
  evaluateState = evaluateState_;
  stateController = stateController_;
  
  nextProcessState = initProcessState;
  nextProcessCov = initProcessCov;
  
  // Some state lags might be propagated. In that case the noise variance for those states is 0, and so is the coefficient on the diagonal of the initial proc covariance matrix
  procCovChol.fill(0.0);
  arma::uvec diagNotZeros = arma::find(initProcessCov.diag() != 0);
  
  procCovChol.submat(diagNotZeros, diagNotZeros) = arma::chol(initProcessCov.submat(diagNotZeros,diagNotZeros), "lower");
  
  transitionParams = modelingParams_["transition"];
  observationParams = modelingParams_["observation"];
  
  alpha = 1.0;
  L = diagNotZeros.n_elem;
  beta = 2.0;
  
  stateCovCube = arma::zeros<arma::cube>(initProcessState_.n_elem, initProcessState_.n_elem, dataMat_.n_rows+1L);
  stateMat = arma::zeros<arma::mat>(dataMat_.n_rows+1L, initProcessState_.n_elem);
  
  stateMat.row(0) = initProcessState.t();
  stateCovCube.slice(0) = initProcessCov;
  
  iterationCounter = 0;
  sampleSize = dataMat.n_rows;
  
  logL = arma::zeros<arma::vec>(dataMat.n_rows);
  predMat = arma::zeros<arma::mat>(dataMat.n_rows, dataMat.n_cols);
  fitMat = arma::zeros<arma::mat>(dataMat.n_rows, dataMat.n_cols);
}

// methods:
// return covariance matrices of filtered states
arma::cube ukfClass::getCovCube(){
  return stateCovCube;
}

// return filtered states
arma::mat ukfClass::getStateMat(){
  return stateMat;
}

// return log-likelihood
arma::vec ukfClass::getLogL(){
  return logL;
}

arma::mat ukfClass::getPredMat(){
  return predMat;
}

arma::mat ukfClass::getFitMat(){
  return fitMat;
}


// set filtering parameters
void ukfClass::setUKFconstants(arma::vec alphaBeta){
  alpha = alphaBeta(0);
  beta = alphaBeta(1);
}

// get filtering parameters
arma::vec ukfClass::getUKFconstants(){
  arma::vec constants(2);
  constants(0) = alpha;
  constants(1) = beta;
  return constants;
}

// This function calls the whole filter
void ukfClass::filterAdditiveNoise(){
  // Reset filter iteration counter to avoid errors.
  iterationCounter = 0;
  // Filtering loop
  for(int runNum = 0; runNum < sampleSize; runNum++){
    filterStep();
  }
  // logL calculation
  // Clean up: reinitialise filter starting values
  reinitialiseFilter();
}

void ukfClass::filterSqrtAdditiveNoise(){
  // Reset filter iteration counter to avoid errors.
  iterationCounter = 0;
  
  // Filtering loop
  for(int runNum = 0; runNum < sampleSize; runNum++){
    filterSqrtStep();
  }
  
  // logL calculation
  // Clean up: reinitialise filter starting values
  reinitialiseFilter();
}

// private: Method to run one step of filter. Stores filtered state and 
// covariance matrix. Updates initProcess and nextProcess variables. Increments
// iteration counter.
void ukfClass::filterStep(){
  // Scaling constant for unscented transformation
  double gamma = pow(pow(alpha,2.0)*L,0.5);
  
  // Generate a set of sigma points from the initial state
  procCovChol.fill(0.0);
  arma::uvec diagNotZeros = arma::find(initProcessCov.diag() != 0);
  
  procCovChol.submat(diagNotZeros, diagNotZeros) = arma::chol(initProcessCov.submat(diagNotZeros,diagNotZeros), "lower");
  arma::mat stateSigma = generateSigmaPoints(initProcessState, gamma, procCovChol, L);
  
  // Propagate the augmented state through the transition dynamics function
  Rcpp::List statePrediction = predictState(stateSigma, transitionParams, iterationCounter);
  
  // Recover augumented propagated state and state noise covariance matrix
  arma::mat nextStateSigma = Rcpp::as<arma::mat>(statePrediction["stateVec"]);
  arma::mat procNoiseMat = Rcpp::as<arma::mat>(statePrediction["procNoiseMat"]);
  
  // Generate sigma point weights for the original augmented state
  arma::mat sigmaWts = generateSigmaWeights(L, alpha, beta);
  
  // Calculate the approximation of the average state after non-linear propagation
  nextProcessState = unscentedMean(nextStateSigma, sigmaWts.col(0));
  
  // Calculate the approximation of the variance-covariance matrix of the 
  // state after non-linear propagation, first line only covers state
  // filtering uncertainty
  nextProcessCov = unscentedCov(nextStateSigma, sigmaWts.col(0), sigmaWts.col(1));
  // Add the state noise itself
  nextProcessCov += procNoiseMat;
  
  // Further extend the state space, see eq. 3.174 and comment in vdM
  diagNotZeros = arma::find(nextProcessCov.diag() != 0);
  procCovChol.fill(0.0);
  procCovChol.submat(diagNotZeros, diagNotZeros) = arma::chol(nextProcessCov.submat(diagNotZeros,diagNotZeros),"lower");
  
  arma::mat extendedNextStateSigma = generateSigmaPoints(nextStateSigma, gamma, procCovChol, L);
  
  // New unscented transformation weights for bigger state matrix
  arma::mat extendedSigmaWts = generateSigmaWeights(2*L, alpha, beta);
  
  // Calculate the observation mapping at predicted points
  Rcpp::List observationPredictionList = evaluateState(extendedNextStateSigma, observationParams, iterationCounter);
  
  // Recover predicted observations and their noise covariance matrix
  arma::mat observationPrediction = Rcpp::as<arma::mat>(observationPredictionList["yhat"]);
  arma::mat observationNoise = Rcpp::as<arma::mat>(observationPredictionList["obsNoiseMat"]);
  
  // Calculate mean and covariance of observed values via the unscented transformation
  arma::mat observationMean = unscentedMean(observationPrediction, extendedSigmaWts.col(0));
  
  arma::mat observationCov = unscentedCov(observationPrediction, extendedSigmaWts.col(0), extendedSigmaWts.col(1));
  
  // Add observation noise to covariance matrix
  observationCov += observationNoise;
  
  // Calculate covariance matrix between states and observations
  arma::mat stateObservationCov = unscentedCrossCov(extendedNextStateSigma, observationPrediction, extendedSigmaWts.col(0), extendedSigmaWts.col(1));
  
  // Kalman Gain
  arma::mat kalmanGain = stateObservationCov * arma::inv_sympd(observationCov);
  
  // Pick data point from dataset
  arma::mat dataPoint = dataMat.row(iterationCounter);
  
  // Kalman update
  nextProcessState += kalmanGain * (dataPoint.t() - observationMean);
  
  // Store state
  stateMat.row(iterationCounter+1L) = nextProcessState.t();
  
  // store obsMean
  predMat.row(iterationCounter) = observationMean.t();
  
  // Calculate fitted values
  Rcpp::List fitted = evaluateState(nextProcessState, observationParams, iterationCounter);
  fitMat.row(iterationCounter) = Rcpp::as<arma::mat>(fitted["yhat"]).t();
  
  // log_likelihood
  logL(iterationCounter) = -1.0*(dataMat.n_cols)/2.0 * log(2.0*arma::datum::pi) - 0.5 * log(arma::det(observationCov));
  logL(iterationCounter) -= 0.5*arma::as_scalar((dataPoint.t() - observationMean).t() * arma::inv_sympd(observationCov) * (dataPoint.t() - observationMean));
  
  // Update state covariance matrix
  nextProcessCov -= kalmanGain * observationCov * kalmanGain.t();
  
  // Store state covariance matrix
  stateCovCube.slice(iterationCounter+1L) = nextProcessCov;
  
  // Update state and variance containers for the loop.
  initProcessState = nextProcessState;
  initProcessCov = nextProcessCov;
  
  // Move iteration counter forward
  ++iterationCounter;
}

void ukfClass::filterSqrtStep(){
  
  // Estabilsh where to expect zero variances, i.e. no noise
  arma::uvec diagNotZeros = arma::find(initProcessCov.diag() != 0);
  // Scaling constant for unscented transformation
  double gamma = pow(pow(alpha,2.0)*L,0.5);
  // Generate sigma points
  arma::mat stateSigma = generateSigmaPoints(initProcessState, gamma, procCovChol, L);
  // Propagate the augmented state through the transition dynamics function
  Rcpp::List statePrediction = predictState(stateSigma, transitionParams, iterationCounter);
  
  // Recover augumented propagated state and state noise covariance matrix
  arma::mat nextStateSigma = Rcpp::as<arma::mat>(statePrediction["stateVec"]);
  arma::mat procNoiseMat = Rcpp::as<arma::mat>(statePrediction["procNoiseMat"]);
  arma::mat procNoiseSmall(diagNotZeros.n_elem, diagNotZeros.n_elem);
  
  bool decSuccess = arma::chol(procNoiseSmall,procNoiseMat.submat(diagNotZeros, diagNotZeros),"lower");
  try{
    if(!decSuccess){
      procNoiseMat.print("procNoiseMat");
      throw std::range_error("ukfRcpp::filterSqrtStep: process noise decomposition failed.");
    }
  } catch(std::exception &ex) {
    ::Rf_error("ukfRcpp::filterSqrtStep: process noise decomposition failed.");
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  procNoiseMat.submat(diagNotZeros, diagNotZeros) = procNoiseSmall;
  
  // Generate sigma point weights for the original augmented state
  arma::mat sigmaWts = generateSigmaWeights(L, alpha, beta);
  
  // Calculate the approximation of the average state after non-linear propagation
  nextProcessState = unscentedMean(nextStateSigma, sigmaWts.col(0));
  
  // First part of covariance update in square root form: QR-decomp, but only take the non-zero stuff.
  arma::mat qrInputBig = nextStateSigma.cols(1,nextStateSigma.n_cols-1);
  arma::mat qrInputSmall(diagNotZeros.n_elem, qrInputBig.n_cols+diagNotZeros.n_elem,arma::fill::zeros);
  qrInputSmall.cols(0,qrInputBig.n_cols-1) = qrInputBig.rows(diagNotZeros);
  qrInputSmall.cols(qrInputBig.n_cols, qrInputSmall.n_cols-1) = procNoiseMat.submat(diagNotZeros,diagNotZeros);
  for(unsigned int kcol=0; kcol < qrInputBig.n_cols; kcol++){
    qrInputSmall.col(kcol) -= nextProcessState.elem(diagNotZeros);
    qrInputSmall.col(kcol) *= sqrt(sigmaWts(1,1)); // you have to multiply all elements by the covariance weights with indices greater than one, but these weights are all the same
  }
  
  // do the QR part
  arma::mat qrQ, qrR;
  arma::mat procCovCholSmall = procCovChol.submat(diagNotZeros,diagNotZeros);
  arma::qr(qrQ,qrR,qrInputSmall.t());
  procCovCholSmall = qrR.submat(0,0,qrR.n_cols-1,qrR.n_cols-1);
  arma::inplace_trans(procCovCholSmall);
  
  // cholupdate
  arma::uvec zeroInd(1,arma::fill::zeros);
  procCovCholSmall = cholupdate(procCovCholSmall, nextStateSigma.submat(diagNotZeros,zeroInd) - nextProcessState.elem(diagNotZeros), sigmaWts(0,1));
  
  // write into big chol
  procCovChol.submat(diagNotZeros,diagNotZeros) = procCovCholSmall;
  
  // augment and extend state space
  arma::mat extendedNextStateSigma = generateSigmaPoints(nextStateSigma, gamma, procCovChol, L);
  
  // New unscented transformation weights for bigger state matrix
  arma::mat extendedSigmaWts = generateSigmaWeights(2*L, alpha, beta);
  
  // Calculate the observation mapping at predicted points
  Rcpp::List observationPredictionList = evaluateState(extendedNextStateSigma, observationParams, iterationCounter);
  // Recover predicted observations and their noise covariance matrix
  arma::mat observationPrediction = Rcpp::as<arma::mat>(observationPredictionList["yhat"]);
  arma::mat observationNoise = Rcpp::as<arma::mat>(observationPredictionList["obsNoiseMat"]);
  decSuccess = arma::chol(observationNoise, observationNoise, "lower");
  try{
    if(!decSuccess){
      observationNoise.print("observation noise");
      throw std::range_error("ukfRcpp::filterSqrtStep: Observation noise decomposition failed.");
    }
  } catch(std::exception &ex) {
    ::Rf_error("ukfRcpp::filterSqrtStep: Observation noise decomposition failed.");
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  // Calculate mean and covariance of observed values via the unscented transformation
  arma::mat observationMean = unscentedMean(observationPrediction, extendedSigmaWts.col(0));
  // observation covariance -- qr form
  arma::mat qrInputObs(observationPrediction.n_rows, observationPrediction.n_cols-1 + observationNoise.n_cols, arma::fill::zeros);
  qrInputObs.cols(0,observationPrediction.n_cols-2) = observationPrediction.cols(1,observationPrediction.n_cols-1);
  qrInputObs.cols(observationPrediction.n_cols-1,qrInputObs.n_cols-1) = observationNoise;
  for(unsigned int kcol=0; kcol < observationPrediction.n_cols-1; kcol++){
    qrInputObs.col(kcol) -= observationMean;
    qrInputObs.col(kcol) *= sqrt(extendedSigmaWts(1,1)); // you have to multiply all elements by the covariance weights with indices greater than one, but these weights are all the same
  }
  // do the QR part
  arma::mat qrQO, qrRO;
  arma::qr(qrQO,qrRO,qrInputObs.t());
  observationNoise = qrRO.submat(0,0,qrRO.n_cols-1, qrRO.n_cols-1);
  arma::inplace_trans(observationNoise);
  // cholupdate
  observationNoise = cholupdate(observationNoise, observationPrediction.col(0) - observationMean, extendedSigmaWts(0,1));
  // Calculate covariance matrix between states and observations
  arma::mat stateObservationCov = unscentedCrossCov(extendedNextStateSigma, observationPrediction, extendedSigmaWts.col(0), extendedSigmaWts.col(1));
  
  // Kalman gain arma::solve(Sy.t(),arma::solve(Sy,pxy.t())).t();
  arma::mat kalmanGainComponent;
  decSuccess = arma::solve(kalmanGainComponent, arma::trimatl(observationNoise), stateObservationCov.t() );
  try{
    if(!decSuccess){
      observationNoise.print("observation noise");
      throw std::range_error("ukfRcpp::filterSqrtStep: kalmanGainComponent failed.");
    }
  } catch(std::exception &ex) {
    ::Rf_error("ukfRcpp::filterSqrtStep: kalmanGainComponent failed.");
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  arma::mat kalmanGain;
  decSuccess = arma::solve(kalmanGain, arma::trimatu(observationNoise.t()), kalmanGainComponent);
  arma::inplace_trans(kalmanGain);
  try{
    if(!decSuccess){
      observationNoise.print("observation noise");
      throw std::range_error("ukfRcpp::filterSqrtStep: kalmanGain failed.");
    }
  } catch(std::exception &ex) {
    ::Rf_error("ukfRcpp::filterSqrtStep: kalmanGain failed.");
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  // Pick data point from dataset
  arma::mat dataPoint = dataMat.row(iterationCounter);
  
  // Kalman update
  nextProcessState += kalmanGain * (dataPoint.t() - observationMean);
  nextProcessState = stateController(nextProcessState);
  
  // log_likelihood
  logL(iterationCounter) = -0.5*(dataMat.n_cols) * log(2.0*arma::datum::pi) - arma::accu(arma::log(observationNoise.diag()));
  arma::mat observationNoiseInv;
  decSuccess = arma::pinv(observationNoiseInv,observationNoise);
  try{
    if(!decSuccess){
      observationNoise.print("observationNoise for pinv");
      throw std::range_error("ukfRcpp::filterSqrtStep: partial inverse of obs noise chol failed");
    }
  } catch(std::exception &ex) {
    ::Rf_error("ukfRcpp::filterSqrtStep: partial inverse of obs noise chol failed");
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  
  logL(iterationCounter) -= 0.5*arma::as_scalar((dataPoint.t() - observationMean).t() * observationNoiseInv.t() * observationNoiseInv * (dataPoint.t() - observationMean));
  
  // Store state
  stateMat.row(iterationCounter+1L) = nextProcessState.t();
  
  // store obsMean
  predMat.row(iterationCounter) = observationMean.t();
  
  // new process obs matrix
  arma::mat UMat = kalmanGain * observationNoise;
  
  procCovCholSmall = procCovChol.submat(diagNotZeros, diagNotZeros);
  
  for(unsigned int kcol = 0; kcol < UMat.n_cols; kcol++){
    arma::vec UMatLoc = UMat.col(kcol);
    procCovCholSmall = cholupdate(procCovCholSmall, UMatLoc.rows(diagNotZeros), -1.0);
  }
  
  procCovChol.submat(diagNotZeros, diagNotZeros) = procCovCholSmall;
  
  // Store state covariance matrix
  stateCovCube.slice(iterationCounter+1L) = procCovChol;
  
  // Calculate fitted values
  Rcpp::List fitted = evaluateState(nextProcessState, observationParams, iterationCounter);
  fitMat.row(iterationCounter) = Rcpp::as<arma::mat>(fitted["yhat"]).t();
  
  // Update state and variance containers for the loop.
  initProcessState = nextProcessState;
  // Move iteration counter forward
  ++iterationCounter;
}

void ukfClass::reinitialiseFilter(){
  initProcessState = constInitProcessState;
  initProcessCov = constInitProcessCov;
}