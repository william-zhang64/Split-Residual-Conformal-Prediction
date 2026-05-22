### Implementation of DtACI method from https://arxiv.org/abs/2208.08401

vecZeroMax <- Vectorize(function(x){max(x,0)})
vecZeroMin <- Vectorize(function(x){min(x,0)})

### Definition of the pinball loss function
pinball <- function(u,alpha){
  alpha*u - vecZeroMin(u)
}

### Input values are the sequence beta_t, the target level, and the sequence of candidate gammas.
### Return value is a list containing the vectors alpha_t, err_t(alpha_t), err_t(alpha), 
### gamma_t, alphabar_t, err_t(alphabar_t), gammabar_t. Here we use the notation err_t(x)
### to refer to the errors computed using input x.
conformalAdaptStable <- function(betas, alpha, gammas, sigma = 1/1000, eta = 2.72, 
                                 alphaInit = alpha, etaAdapt = FALSE, etaLookback = 500, ahead = 1) {
  T <- length(betas)
  k <- length(gammas)
  
  # Initialize sequences
  alphaSeq <- rep(alphaInit, T)
  errSeqAdapt <- rep(0, T)
  errSeqFixed <- rep(0, T)
  gammaSeq <- rep(0, T)
  meanAlphaSeq <- rep(0, T)
  meanErrSeq <- rep(0, T)
  meanGammas <- rep(0, T)
  lossSeq <- rep(0, T)
  
  # Expert initialization
  expertAlphas <- rep(alphaInit, k)
  expertWs <- rep(1, k)
  expertProbs <- rep(1 / k, k)
  expertCumulativeLosses <- rep(0, k)
  curExpert <- sample(1:k, 1)
  
  for (t in 1:T) {
    t_pred <- t - ahead + 1  # when the prediction was made

    # Always record current predictions/states
    alphat <- expertAlphas[curExpert]
    alphaSeq[t] <- alphat
    errSeqFixed[t] <- as.numeric(alpha > betas[t])
    gammaSeq[t] <- gammas[curExpert]
    meanAlphaSeq[t] <- sum(expertProbs * expertAlphas)
    meanErrSeq[t] <- as.numeric(meanAlphaSeq[t] > betas[t])
    meanGammas[t] <- sum(expertProbs * gammas)

    # Only update after we get feedback for a prediction made at t_pred
    if (t_pred >= 1) {
      # Did the prediction made at t_pred cover the actual outcome at t?
      error_t <- as.numeric(alphaSeq[t_pred] > betas[t])

      # Compute pinball losses for each expert
      expertLosses <- pinball(betas[t] - expertAlphas, alpha)
      lossSeq[t] <- sum(expertLosses * expertProbs)

      # Update expert alphas
      expertAlphas <- expertAlphas + gammas * (alpha - error_t)

      # Clip alphas to [0, 1]
      expertAlphas <- pmin(pmax(expertAlphas, 0), 1)

      # Update expert weights and sample new expert
      if (etaAdapt && t > etaLookback) {
        eta <- sqrt((log(2 * k * etaLookback) + 1) / sum(lossSeq[(t - etaLookback):(t - 1)]^2))
      }

      if (eta < Inf) {
        expertBarWs <- expertWs * exp(-eta * expertLosses)
        expertNextWs <- (1 - sigma) * expertBarWs / sum(expertBarWs) + sigma / k
        expertProbs <- expertNextWs / sum(expertNextWs)
        curExpert <- sample(1:k, 1, prob = expertProbs)
        expertWs <- expertNextWs
      } else {
        expertCumulativeLosses <- expertCumulativeLosses + expertLosses
        curExpert <- which.min(expertCumulativeLosses)
      }
    }
  }

  return(list(
    alphaSeq = alphaSeq,
    errSeqAdapt = errSeqAdapt,
    errSeqFixed = errSeqFixed,
    gammaSeq = gammaSeq,
    meanAlphaSeq = meanAlphaSeq,
    meanErrSeq = meanErrSeq,
    meanGammas = meanGammas
  ))
}



