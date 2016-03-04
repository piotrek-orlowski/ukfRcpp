#ifndef CHOLUPDATE_H
#define CHOLUPDATE_H
#include <RcppArmadillo.h>

arma::mat cholupdate(arma::mat RMat, arma::vec xVec, double nu);

#endif