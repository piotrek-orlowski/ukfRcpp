set.seed(16237247)
xMat <- matrix(1,nrow=3)
L <- 3
alpha <- 0.9
beta <- 2
gam <- sqrt(L + alpha^2*L-L)
PMat <- diag(c(0.1,0.2,0.1))
PMat[1,2] <- PMat[2,1] <- -0.05
PMat[2,3] <- PMat[3,2] <- 0.025
PChol <- t(chol(PMat))

bigX <- generateSigmaPoints(xMat = xMat, gam = gam, PMatChol = PChol)
biggerX <- generateSigmaPoints(xMat = bigX, gam = gam, PMatChol = PChol)

xSample <- matrix(rnorm(n = 3*1e5, mean = 0, sd = 1), ncol = 3)
xSample <- xSample %*% chol(PMat)
xSample <- xSample + 1

locFoo <- function(x) x^2

xTransf <- locFoo(xSample)

bigXtransf <- locFoo(bigX)

unscWts <- generateSigmaWeights(L = L, alpha = alpha, beta = beta)

trueMean <- apply(xTransf,2,mean)
trueCov <- cov(xTransf)

unscXMean <- unscentedMean(xSigma = bigXtransf, unscWts = unscWts[,1])
unscXCov <- unscentedCov(xSigma = bigXtransf, unscWtsMean = unscWts[,1], unscWtsCov = unscWts[,2])

#### ---- UKF CPP TEST ----

library(ukfRcpp)
MSE <- numeric(length(seq(0.01,1,by=0.01)))
counter <- 1

aa <- testUKFclass(alpha = 0.5)
