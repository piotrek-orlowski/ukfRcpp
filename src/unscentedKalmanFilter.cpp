#include "../inst/include/ukfClass.h"
using namespace Rcpp;

//' @name unscentedKalmanFilter
//' @title Unscented Kalman Filter
//' @description This function accepts pointers to compiled c++ evaluator 
//'   functions for the observation equations and the transition equations, plus
//'   data and parameters, and evaluates the UKF.
//' @param dataMat matrix containing observed data, by-column
//' @param initState initial state vector values
//' @param initProcCov initial variance-covariance matrix of the state vector
//' @param modelParams \code{list} of model parameter objects, compatible with 
//'   your \code{predictState} and \code{evaluateState} functions, and with the 
//'   specification of the \code{ukfClass}
//' @param predict_Xptr \code{XPtr} to \code{predictState} function which 
//'   handles your one-step head prediction of the state variable, given current
//'   values.
//' @param evaluate_XPtr \code{XPtr} to \code{evaluateState} function which 
//'   calculates the values of the measurement equation given values of the 
//'   state.
//' @param control_XPtr \code{XPtr} to \code{stateController} function which 
//'   handles pathological state cases after the filtering step (e.g. when using
//'   a Gaussian filter on strictly positive variables, which turn negative 
//'   after the filtering step, it is useful to set them to a small value)
//' @return \code{List} with the following fields: \code{estimState} \code{(T+1)
//'   x N} matrix of filtered states, with the initial state vector in the first
//'   row; \code{stateCovCube} 3-dimensional array of posterior state 
//'   variance-covariance matrices, \code{N x N x T}; \code{logL} vector of 
//'   Gaussian likelihood values, see e.g. "Time Series Analysis" by J.D. 
//'   Hamilton, \code{predMat} \code{T x M} matrix of predictions of 
//'   observations at time \code{t} given observations at time \code{t-1}, 
//'   \code{fitMat} \code{T x M} matrix of observation equations evaluated at
//'   filtered state values.
//' @export

// [[Rcpp::export]]
List unscentedKalmanFilter(const arma::mat dataMat, const arma::vec initState, const arma::mat initProcCov, const List modelParams, SEXP predict_XPtr, SEXP evaluate_XPtr, SEXP control_XPtr){
  
  Rcpp::XPtr<stateHandler> predict_XPtr_(predict_XPtr);
  Rcpp::XPtr<stateHandler> evaluate_XPtr_(evaluate_XPtr);
  Rcpp::XPtr<stateControl> control_XPtr_(control_XPtr);
  
  // Set state handlers  
  stateHandler transitionPtr = *predict_XPtr_;
  stateHandler observationPtr = *evaluate_XPtr_;
  stateControl controlPtr = *control_XPtr_;
  
  // Instantiate filter instance
  ukfClass filterInstance(dataMat, initState, initProcCov, transitionPtr, observationPtr, controlPtr, modelParams);
  
  // Run filter
  filterInstance.filterSqrtAdditiveNoise();
  
  // Set return variable
  List res = List::create(
    Named("estimState") = filterInstance.getStateMat()
    , Named("stateCovCube") = filterInstance.getCovCube()
    , Named("logL") = filterInstance.getLogL()
    , Named("predMat") = filterInstance.getPredMat()
    , Named("fitMat") = filterInstance.getFitMat()
  );
  
  return res;
}