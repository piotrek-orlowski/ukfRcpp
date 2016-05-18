# ukfRcpp

The Unscented Kalman Filter (see http://www.cslu.ogi.edu/publications/ps/merwe04.pdf for a good overview of the algorithms) is a great tool for non-linear filtering tasks.

This package implements the UKF with *additive errors* algorithm, both in standard and square-root form; see pages 108 and 115 in the document linked above.

# Installation

The package has been tested to compile on Linux and Windows machines. To install the package, use the `devtools` package:
```
library(devtools)
install_github(repo = "piotrek-orlowski/ukfRcpp")
```
# When to use it?

The package is designed for using in scenarios when it is complicated to calculate the transition/observation parameters given a set of parameters of the underlying mathematical model. For example, in Stochastic Volatility models in finance applied to option pricing, the underlying model parameters are used to calculate: (1) the coefficients for evaluating the moments of the conditional state transition density, and (2) the coefficients for evaluating option prices given the current state. The use scenario is thus to write a program handling coefficient calculation and use them as parameters in the possibly non-linear transformations, plugged into the UKF via `predictState` and `evaluateState` functions described below.

# How to use it?

When using a Kalman Filter, the user has to specify matrices which determine the transition and observation densities. In the case of the UKF, the user has to specify functions which characterise the non-linear transition and non-linear observation transformations.

The objective, when writing this package, was to provide a *fast* general filtering interface and separate the *mathematical system modeling* step from the *probabilistic filtering* step. This is why the package is designed as a library to be linked against, so that users can provide Rcpp-based functions for handling the transition and observation equations.

A user of the package has to provide two `C++` functions, a `predictState` and an `evaluateState`. Both these functions have to accept the following arguments: `arma::mat&` (state matrix), `Rcpp::List&` (list with arbitrary objects passed as parameters of the model) and `int` where the observation number can be passed if the parameters of the model are time-varying. The handler functions should return `Rcpp::List` objects with function-specific field names.

The `predictState` function is responsible for forming `E[X_t | X_{t-1}]` for a range of state values. In the `Rcpp::List` object it has to return fields `stateVec` (state prediction) and `procNoiseMat` (driving process noise variance-covariance matric).

The `evaluateState` function is responible for evaluating the observation equations `Y_t = g[X_t]` for a range of state values. In the `Rcpp::List` object it has to return fields `yHat` containing the predicted observations, and `obsNoiseMat` containing the variance-covariance matrix of observation noise.

If the system under consideration has `N` latent states, the functions have to be able to handle `S` state values. The `arma::mat` argument to `predictState` and `evaluateState` should be of size `N x S`. The `stateVec` field from `predictState` has to be of the same size.

The package handles propagating past states: it requires setting the corresponding rows/columns of the initial state variance/covariance matrix to 0, and keeping it this way.

The functions are passed to the constructor of `ukfClass` as pointers.

In general, a function using the `ukfRcpp` package will be of the following form:
```
#include <RcppArmadillo.h>
#include "../inst/include/ukfClass.h"
#include "testHandlers.h" // where the user codes predictState and evaluateState
using namespace std;
Rcpp::List my_sqrtFilter(const arma::mat dataMat, const arma:vec initState, const arma::mat initProcCov, const Rcpp::List modelParams){
  
  stateHandler transitionPtr = &predictState;
  stateHandler observationPtr = &evaluateState;
  stateControl controlPtr = &testCheck;
  
  // Initialize ukfClass object
  ukfClass myFirstFilter(dataMat, initState, initProcCov, transitionPtr, observationPtr, controlPtr, modelParams);
  
  // Filter!
  myFirstFilter.filterSqrtAdditiveNoise();
  
  Rcpp::List res;
  // Create return list object
  res = Rcpp::List::create(Rcpp::Named("estimState") = myFirstFilter.getStateMat(), Rcpp::Named("stateCovCube") = myFirstFilter.getCovCube(), Rcpp::Named("logL") = myFirstFilter.getLogL(), Rcpp::Named("predMat") = myFirstFilter.getPredMat() ;
  
  return res;
}
```

The file https://github.com/piotrek-orlowski/ukfRcpp/blob/master/src/testFilter.cpp provides an example. This file compiles alongside the package and exposes to the user the `testUKFclass` function. The function simulates a simple two-state VAR, generates noisy non-linear observations, initializes and runs the filter, and finally returns true and filtered states, and noisy observations alongside a Gaussian likelihood.

# Using as a shared library

If you use ukfRcpp for multiple projects, the most convenient setup is one where you link your `R` packages dynamically. In order to do that, your `R` packages `src/Makevars` or `src/Makevars.win` files have to contain explicit linking information. If you're running a 64-bit Windows box, and assuming that your `R` packages are installed to `c:/path/to/Rlibs`, this amounts to providing the following flags in the `src/Makevars.win` file:

```
PKG_LIBS = $(shell $(R_HOME)/bin/Rscript.exe -e "Rcpp:::LdFlags()") $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS) -Lc:/path/to/Rlibs/ukfRcpp/libs/x64 -lukfRcpp
PKG_CXXFLAGS = -Ic:/path/to/Rlibs/ukfRcpp/include
```

If you use RStudio, providing these flags will also allow RStudio to help with code completion from the `ukfRcpp` code.

# Author

The package is being developed by Piotr Orłowski
