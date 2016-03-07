#include <RcppArmadillo.h>
#include "../inst/include/ukfClass.h"
#include "../inst/include/unscentedMeanCov.h"
#include "cholupdate.h"
using namespace std;

// Unscented Kalman Filter with additive noise, van der Merwe pp. 108-110. Rather general implementation.

// public:
// Constructor for cpp
  ukfClass::ukfClass(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, stateHandler predictState_, stateHandler evaluateState_, Rcpp::List modelingParams_) : dataMat(dataMat_), initProcessState(initProcessState_), initProcessCov(initProcessCov_), constInitProcessState(initProcessState_), constInitProcessCov(initProcessCov_) {
    predictState = predictState_;
    evaluateState = evaluateState_;
    
    nextProcessState = initProcessState;
    nextProcessCov = initProcessCov;
    
    // Some state lags might be propagated. In that case the noise variance for those states is 0, and so is the coefficient on the diagonal of the initial proc covariance matrix
    procCovChol.fill(0.0);
    arma::uvec diagNotZeros = arma::find(initProcessCov.diag() != 0);
    procCovChol.submat(diagNotZeros, diagNotZeros) = arma::chol(initProcessCov.submat(diagNotZeros,diagNotZeros), "lower");
    
    transitionParams = modelingParams_["transition"];
    observationParams = modelingParams_["observation"];
    
    alpha = 1.0;
    L = initProcessState.n_elem;
    beta = 2.0;
    
    stateCovCube = arma::zeros<arma::cube>(initProcessState_.n_elem, initProcessState_.n_elem, dataMat_.n_rows+1L);
    stateMat = arma::zeros<arma::mat>(dataMat_.n_rows+1L, initProcessState_.n_elem);
    
    stateMat.row(0) = initProcessState.t();
    stateCovCube.slice(0) = initProcessCov;
    
    iterationCounter = 0;
    sampleSize = dataMat.n_rows;
  }
  
// Constructor for R
  ukfClass::ukfClass(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, SEXP predictState_, SEXP evaluateState_, Rcpp::List modelingParams_) : dataMat(dataMat_), initProcessState(initProcessState_), initProcessCov(initProcessCov_), constInitProcessState(initProcessState_), constInitProcessCov(initProcessCov_) {
    
    Rcpp::XPtr<stateHandler> predictStateXptr(predictState_);
    predictState = *predictStateXptr;
    
    Rcpp::XPtr<stateHandler> evaluateStateXptr(evaluateState_);
    evaluateState = *evaluateStateXptr;
    
    // predictState = predictState_;
    // evaluateState = evaluateState_;
    
    nextProcessState = initProcessState;
    nextProcessCov = initProcessCov;
    
    // Some state lags might be propagated. In that case the noise variance for those states is 0, and so is the coefficient on the diagonal of the initial proc covariance matrix
    procCovChol.fill(0.0);
    arma::uvec diagNotZeros = arma::find(initProcessCov.diag() != 0);
    procCovChol.submat(diagNotZeros, diagNotZeros) = arma::chol(initProcessCov.submat(diagNotZeros,diagNotZeros), "lower");
    
    transitionParams = modelingParams_["transition"];
    observationParams = modelingParams_["observation"];
    
    alpha = 0.25;
    L = initProcessState.n_elem;
    beta = 2.0;
    
    stateCovCube = arma::zeros<arma::cube>(initProcessState_.n_elem, initProcessState_.n_elem, dataMat_.n_rows+1L);
    stateMat = arma::zeros<arma::mat>(dataMat_.n_rows+1L, initProcessState_.n_elem);
    
    stateMat.row(0) = initProcessState.t();
    stateCovCube.slice(0) = initProcessCov;
    sampleSize = dataMat.n_rows;
    iterationCounter = 0;
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
    
    // arma::mat stateSigma = generateSigmaPoints(initProcessState, gamma, arma::chol(initProcessCov, "lower"));
    arma::mat stateSigma = generateSigmaPoints(initProcessState, gamma, procCovChol);
    
    // Propagate the augmented state through the transition dynamics function
    Rcpp::List statePrediction = predictState(stateSigma, transitionParams);
    
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
    // arma::mat extendedNextStateSigma = generateSigmaPoints(nextStateSigma, gamma, arma::chol(nextProcessCov,"lower"));
    arma::mat extendedNextStateSigma = generateSigmaPoints(nextStateSigma, gamma, procCovChol);
    
    // New unscented transformation weights for bigger state matrix
    arma::mat extendedSigmaWts = generateSigmaWeights(2*L, alpha, beta);
    
    // Calculate the observation mapping at predicted points
    Rcpp::List observationPredictionList = evaluateState(extendedNextStateSigma, observationParams);
    
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
    arma::mat kalmanGain = stateObservationCov * arma::inv(observationCov);
    
    // Pick data point from dataset
    arma::mat dataPoint = dataMat.row(iterationCounter);
    
    // Kalman update
    nextProcessState += kalmanGain * (dataPoint.t() - observationMean);
    
    // Store state
    stateMat.row(iterationCounter+1L) = nextProcessState.t();
    
    // Update state covariance matrix
    nextProcessCov -= kalmanGain * observationCov * kalmanGain.t();
    
    // Store state covariance matrix
    stateCovCube.slice(iterationCounter+1L) = nextProcessCov;
    
    // Update state and variance containers for the loop.
    initProcessState = nextProcessState;
    initProcessCov = nextProcessCov;
    
    // Move iteration counter forward
    iterationCounter++;
  }
  
  void ukfClass::filterSqrtStep(){
    
    // Estabilsh where to expect zero variances, i.e. no noise
    arma::uvec diagNotZeros = arma::find(initProcessCov.diag() != 0);
    
    // Scaling constant for unscented transformation
    double gamma = pow(pow(alpha,2.0)*L,0.5);

    // Generate sigma points
    arma::mat stateSigma = generateSigmaPoints(initProcessState, gamma, procCovChol);
    
    // Propagate the augmented state through the transition dynamics function
    Rcpp::List statePrediction = predictState(stateSigma, transitionParams);
    
    // Recover augumented propagated state and state noise covariance matrix
    arma::mat nextStateSigma = Rcpp::as<arma::mat>(statePrediction["stateVec"]);
    arma::mat procNoiseMat = Rcpp::as<arma::mat>(statePrediction["procNoiseMat"]);
    procNoiseMat = arma::chol(procNoiseMat);
    
    // Generate sigma point weights for the original augmented state
    arma::mat sigmaWts = generateSigmaWeights(L, alpha, beta);
    
    // Calculate the approximation of the average state after non-linear propagation
    nextProcessState = unscentedMean(nextStateSigma, sigmaWts.col(0));
    
    // First part of covariance update in square root form: QR-decomp, but only take the non-zero stuff.
    arma::mat qrInputBig = nextStateSigma.cols(1,nextStateSigma.n_cols-1);
    arma::mat qrInputSmall(diagNotZeros.n_elem, qrInputBig.n_cols+diagNotZeros.n_elem,arma::fill::zeros);
    qrInputSmall.cols(0,qrInputBig.n_cols-1) = qrInputBig.rows(diagNotZeros);
    qrInputSmall.cols(qrInputBig.n_cols, qrInputSmall.n_cols-1) = procNoiseMat.submat(diagNotZeros,diagNotZeros);
    for(int kcol=0; kcol < qrInputBig.n_cols; kcol++){
      qrInputSmall.col(kcol) -= nextProcessState.elem(diagNotZeros);
      qrInputSmall.col(kcol) *= sqrt(sigmaWts(0,1)); // you have to multiply all elements by the covariance weights with indices greater than one, but these weights are all the same
    }

    // do the QR part
    arma::mat qrQ, qrR;
    arma::mat procCovCholSmall = procCovChol.submat(diagNotZeros,diagNotZeros);
    arma::qr(qrQ,qrR,qrInputSmall);
    procCovCholSmall = qrR.rows(0,diagNotZeros.n_elem);
    arma::inplace_trans(procCovCholSmall);
    
    // cholupdate
    arma::uvec zeroInd(1,arma::fill::zeros);
    procCovCholSmall = cholupdate(procCovCholSmall, nextStateSigma.submat(diagNotZeros,zeroInd) - nextProcessState.elem(diagNotZeros), sigmaWts(0,1));
    
    // write into big chol
    procCovChol.submat(diagNotZeros,diagNotZeros) = procCovCholSmall;
    
    // augment and extend state space
    arma::mat extendedNextStateSigma = generateSigmaPoints(nextStateSigma, gamma, procCovChol);
    
    // New unscented transformation weights for bigger state matrix
    arma::mat extendedSigmaWts = generateSigmaWeights(2*L, alpha, beta);
    
    // Calculate the observation mapping at predicted points
    Rcpp::List observationPredictionList = evaluateState(extendedNextStateSigma, observationParams);
    
    // Recover predicted observations and their noise covariance matrix
    arma::mat observationPrediction = Rcpp::as<arma::mat>(observationPredictionList["yhat"]);
    arma::mat observationNoise = Rcpp::as<arma::mat>(observationPredictionList["obsNoiseMat"]);
    observationNoise = arma::chol(observationNoise);
    
    // Calculate mean and covariance of observed values via the unscented transformation
    arma::mat observationMean = unscentedMean(observationPrediction, extendedSigmaWts.col(0));
    
    // observation covariance -- qr form
    arma::mat qrInputObs(observationPrediction.n_rows, observationPrediction.n_cols-1 + observationNoise.n_cols, arma::fill::zeros);
    qrInputObs.cols(0,observationPrediction.n_cols-2) = observationPrediction.cols(1,observationPrediction.n_cols-1);
    qrInputObs.cols(observationPrediction.n_cols-1,qrInputObs.n_cols-1) = observationNoise;
    for(int kcol=0; kcol < observationPrediction.n_cols-1; kcol++){
      qrInputObs.col(kcol) -= observationMean;
      qrInputObs.col(kcol) *= sqrt(extendedSigmaWts(1,1)); // you have to multiply all elements by the covariance weights with indices greater than one, but these weights are all the same
    }
    
    // do the QR part
    arma::mat qrQO;
    arma::qr(qrQO,observationNoise,qrInputObs);
    arma::inplace_trans(observationNoise);
    
    // cholupdate
    observationNoise = cholupdate(observationNoise, observationPrediction.col(0) - observationMean, extendedSigmaWts(0,1));
    
    // Calculate covariance matrix between states and observations
    arma::mat stateObservationCov = unscentedCrossCov(extendedNextStateSigma, observationPrediction, extendedSigmaWts.col(0), extendedSigmaWts.col(1));
    
    // Kalman gain arma::solve(Sy.t(),arma::solve(Sy,pxy.t())).t();
    arma::mat kalmanGain = arma::solve(observationNoise.t(), arma::solve(observationNoise, stateObservationCov.t())).t();
    // arma::mat kalmanGain = arma::solve(arma::solve(observationNoise.t(),stateObservationCov),observationNoise);
    
    // Pick data point from dataset
    arma::mat dataPoint = dataMat.row(iterationCounter);
    
    // Kalman update
    nextProcessState += kalmanGain * (dataPoint.t() - observationMean);
    
    // Store state
    stateMat.row(iterationCounter+1L) = nextProcessState.t();
    
    // new process obs matrix
    arma::mat UMat = kalmanGain * observationNoise;
    procCovCholSmall = procCovChol.submat(diagNotZeros, diagNotZeros);
    for(int kcol = 0; kcol < UMat.n_cols; kcol++){
      procCovCholSmall = cholupdate(procCovCholSmall, UMat.col(kcol), -1.0);
    }
    procCovChol.submat(diagNotZeros, diagNotZeros) = procCovCholSmall;
    
    // Store state covariance matrix
    stateCovCube.slice(iterationCounter+1L) = procCovChol;
    
    // Update state and variance containers for the loop.
    initProcessState = nextProcessState;
    
    // Move iteration counter forward
    iterationCounter++;
  }
  
  void ukfClass::reinitialiseFilter(){
    initProcessState = constInitProcessState;
    initProcessCov = constInitProcessCov;
  }

RCPP_MODULE(ukf){
  
  Rcpp::class_<ukfClass>("ukf")
  
  // Expose R constructor
  .constructor<arma::mat, arma::vec, arma::mat, SEXP, SEXP, Rcpp::List>()
  
  // Expose methods
  .property("stateCovCube", &ukfClass::getCovCube, "Retrieve state covariance cube.")
  .property("stateMat", &ukfClass::getStateMat, "Retrieve filtered states.")
  .property("logL", &ukfClass::getLogL, "Retrieve log-likelihood vector.")
  .property("ukfConst", &ukfClass::getUKFconstants, &ukfClass::setUKFconstants, "UKF parameters alpha (first) and beta (second) argument/element of returned vector.")
  .method("filterAdditiveNoise", &ukfClass::filterAdditiveNoise, "Run filter on the sample.")
  .method("reinitialiseFilter", &ukfClass::reinitialiseFilter, "Reinitialise all containers and counters to at-construction state.")
  
  // Expose fields
  .field( "dataMat", &ukfClass::dataMat)
  ;
}