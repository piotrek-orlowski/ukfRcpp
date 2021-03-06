#ifndef FPTR_H
#define FPTR_H

#include<RcppArmadillo.h>

typedef Rcpp::List (*stateHandler)(const arma::mat&, const Rcpp::List&, const int);
typedef arma::mat (*stateControl)(arma::mat);

#endif