% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{unscentedKalmanFilter}
\alias{unscentedKalmanFilter}
\title{Unscented Kalman Filter}
\arguments{
\item{dataMat}{matrix containing observed data, by-column}

\item{initState}{initial state vector values}

\item{initProcCov}{initial variance-covariance matrix of the state vector}

\item{modelParams}{\code{list} of model parameter objects, compatible with 
your \code{predictState} and \code{evaluateState} functions, and with the 
specification of the \code{ukfClass}}

\item{predict_Xptr}{\code{XPtr} to \code{predictState} function which 
handles your one-step head prediction of the state variable, given current
values.}

\item{evaluate_XPtr}{\code{XPtr} to \code{evaluateState} function which 
calculates the values of the measurement equation given values of the 
state.}

\item{control_XPtr}{\code{XPtr} to \code{stateController} function which 
handles pathological state cases after the filtering step (e.g. when using
a Gaussian filter on strictly positive variables, which turn negative 
after the filtering step, it is useful to set them to a small value)}
}
\value{
\code{List} with the following fields: \code{estimState} \code{(T+1)
  x N} matrix of filtered states, with the initial state vector in the first
  row; \code{stateCovCube} 3-dimensional array of posterior state 
  variance-covariance matrices, \code{N x N x T}; \code{logL} vector of 
  Gaussian likelihood values, see e.g. "Time Series Analysis" by J.D. 
  Hamilton, \code{predMat} \code{T x M} matrix of predictions of 
  observations at time \code{t} given observations at time \code{t-1}, 
  \code{fitMat} \code{T x M} matrix of observation equations evaluated at
  filtered state values.
}
\description{
This function accepts pointers to compiled c++ evaluator 
  functions for the observation equations and the transition equations, plus
  data and parameters, and evaluates the UKF.
}
