#include <RcppArmadillo.h>
#include "../inst/include/ukfClass.h"
#include "../inst/include/unscentedMeanCov.h"
using namespace std;

// Unscented Kalman Filter with additive noise, van der Merwe pp. 108-110. Rather general implementation.

// public:
// Constructor for cpp
  ukfRcpp::ukfRcpp(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, stateHandler predictState_, stateHandler evaluateState_, Rcpp::List modelingParams_) : dataMat(dataMat_), initProcessState(initProcessState_), initProcessCov(initProcessCov_), constInitProcessState(initProcessState_), constInitProcessCov(initProcessCov_) {
    predictState = predictState_;
    evaluateState = evaluateState_;
    
    nextProcessState = initProcessState;
    nextProcessCov = initProcessCov;
    
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
  ukfRcpp::ukfRcpp(arma::mat dataMat_, arma::vec initProcessState_, arma::mat initProcessCov_, SEXP predictState_, SEXP evaluateState_, Rcpp::List modelingParams_) : dataMat(dataMat_), initProcessState(initProcessState_), initProcessCov(initProcessCov_), constInitProcessState(initProcessState_), constInitProcessCov(initProcessCov_) {
    
    Rcpp::XPtr<stateHandler> predictStateXptr(predictState_);
    predictState = *predictStateXptr;
    
    Rcpp::XPtr<stateHandler> evaluateStateXptr(evaluateState_);
    evaluateState = *evaluateStateXptr;
    
    // predictState = predictState_;
    // evaluateState = evaluateState_;
    
    nextProcessState = initProcessState;
    nextProcessCov = initProcessCov;
    
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
  arma::cube ukfRcpp::getCovCube(){
    return stateCovCube;
  }
  
// return filtered states
  arma::mat ukfRcpp::getStateMat(){
    return stateMat;
  }
  
// return log-likelihood
  arma::vec ukfRcpp::getLogL(){
    return logL;
  }
  
// set filtering parameters
  void ukfRcpp::setUKFconstants(arma::vec alphaBeta){
    alpha = alphaBeta(0);
    beta = alphaBeta(1);
  }
  
// get filtering parameters
  arma::vec ukfRcpp::getUKFconstants(){
    arma::vec constants(2);
    constants(0) = alpha;
    constants(1) = beta;
    return constants;
  }
    
// This function calls the whole filter
  void ukfRcpp::filterAdditiveNoise(){
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
  
// private: Method to run one step of filter. Stores filtered state and 
// covariance matrix. Updates initProcess and nextProcess variables. Increments
// iteration counter.
  void ukfRcpp::filterStep(){
    // Scaling constant for unscented transformation
    double gamma = pow(pow(alpha,2.0)*L,0.5);
   
    // Generate a set of sigma points from the initial state
    arma::mat procCovChol(initProcessCov);
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
  void ukfRcpp::reinitialiseFilter(){
    initProcessState = constInitProcessState;
    initProcessCov = constInitProcessCov;
  }

RCPP_MODULE(ukf){
  
  Rcpp::class_<ukfRcpp>("ukf")
  
  // Expose R constructor
  .constructor<arma::mat, arma::vec, arma::mat, SEXP, SEXP, Rcpp::List>()
  
  // Expose methods
  .property("stateCovCube", &ukfRcpp::getCovCube, "Retrieve state covariance cube.")
  .property("stateMat", &ukfRcpp::getStateMat, "Retrieve filtered states.")
  .property("logL", &ukfRcpp::getLogL, "Retrieve log-likelihood vector.")
  .property("ukfConst", &ukfRcpp::getUKFconstants, &ukfRcpp::setUKFconstants, "UKF parameters alpha (first) and beta (second) argument/element of returned vector.")
  .method("filterAdditiveNoise", &ukfRcpp::filterAdditiveNoise, "Run filter on the sample.")
  .method("reinitialiseFilter", &ukfRcpp::reinitialiseFilter, "Reinitialise all containers and counters to at-construction state.")
  
  // Expose fields
  .field( "dataMat", &ukfRcpp::dataMat)
  ;
}