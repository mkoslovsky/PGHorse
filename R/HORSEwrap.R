########################################################################################################################################
#' This is an R wrapper for the C++ code PGHorse.cpp that implements Bayesian variable selection for random effects as well as varying-coefficients
#'  using horseshow prior for logistic regression models with repeated measures data using Polya-Gamma augmentation for
#'  efficient sampling. 
#'
#'
#' Author: Matt Koslovsky 2020
########################################################################################################################################

# Simulate data 
data_sim <- function(
  N = 100,
  n_i = rep(20, 100),
  P = 10,
  D = 10,
  cor = 0.0,
  beta_bar = NULL,
  non_linear = NULL,
  kappa = NULL,
  Gamma = NULL,
  zeta = NULL, 
  seed = 121113
){
  set.seed( seed )
  
  # Get libraries to simulate data
  library( mvtnorm )
  library( MCMCpack )
  
  # Adjust the number of observations per person if necessary
  if( length( n_i ) == 1 ){ n_i <- rep( n_i, N ) }
  
  # Make X - Correlation between covariates is cor
  sigma2 <- diag( P )
  
  for( i in 1:P ){
    for( j in 1:P ){
      if( i != j ){
        sigma2[ i , j ] = cor^abs(i - j)
      }
    }
  }
  
  # Simulate a covariate for each subject
  X <- rmvnorm( N, rep( 0, nrow( sigma2 ) ), sigma2 )
  
  # Replicate some and make the others wander
  X_ij <- numeric()
  cols <- sample( seq(1,P), floor( P/2 ) )
  for( i in 1:N ){
    # Replicate by number of observations
    x_ij <- matrix( rep( X[ i, ], n_i[ i ] ), ncol = P, byrow = TRUE )
    
    # Give 1/2 of the columns random noise
    x_ij[, cols ] <- x_ij[, cols ] + matrix( rnorm( length( cols )*n_i[ i ] ), ncol = length( cols ), byrow = TRUE )
    
    # Append observations
    X_ij <- rbind( X_ij, x_ij )
  }
  
  # Add intercept term to X_ij
  X_ij <- scale( X_ij )
  
  # Copy fixed effects for random effects
  X_ij[,1] <- 1
  Z_ij <- X_ij
  
  # Make U - Assumes that the input into spline function is the same
  U <- numeric()
  for( j in 1:length( n_i ) ){
    U <- c( U, sort( runif( n_i[ j ], 0, 1 )))
  }
  U <- matrix( rep( U, ncol( X_ij) ), ncol = ncol( X_ij) )
  
  # Matrix of barX. Rows indexed by ij. Columns indexed by 2P ( x1u, x1, x2u, x2,...xPu, xP )
  Xbar <- numeric()
  for( p in 1:ncol( U ) ){
    tmp <- X_ij[ , p]*U[ , p ]
    Xbar <- cbind( Xbar, tmp, X_ij[ , p ] )
  }
  
  ###### Simulate data ######
  # Make home for eta
  eta <- rep( 0 , sum( n_i ) )
  
  ### Fixed effects ###
  # main effects and linear interactions 
  beta_bar. <- if( is.null( beta_bar ) ){ c( 1, -1, sample( c( 0,1.5,-1.5,2,-2), ( 2*P - 2 ), TRUE, c(.6,.1,.1,.1,.1) ) ) }else{ beta_bar } 
  
  # non-linear interactions
  non_linear. <- if( is.null( non_linear ) ){ c( 1, sample( 2:P, 2 ) ) }else{ non_linear } 
  
  for( p in non_linear. ){
    if( p == 1 ){
      eta <- eta + ( pi*sin( 3*pi*U[,p] ) )*X_ij[ , p ]
    }
    if( p == 2 ){
      eta <- eta + ( pi*cos( 2*pi*U[,p] ) )*X_ij[ , p ]
    }
    if( p == 3 ){
      eta <- eta - ( pi*U[,p]*cos( 5*pi*U[,p] ) )*X_ij[ , p ]
    }
    if( p > 3 ){
      eta <- eta - ( pi*sin( 5*pi*U[,p] ) )*X_ij[ , p ]
    }
  }
  
  eta <- eta + Xbar%*%beta_bar.
  
  ### Random effects ###
  # Matrix of random effects
  kappa. <- if( is.null( kappa ) ){ diag( c( 1, sample( c( 0,1.5,2), ( D - 1 ), TRUE, c(.6,.2,.2) ) ) ) }else{ kappa } 
  Gamma.<- if( is.null( Gamma ) ){ diag( D ) }else{ Gamma } 
  zeta. <- if( is.null( zeta ) ){ matrix( rnorm( D*N ), nrow = N, ncol = D ) }else{ zeta } 
  
  subject <- rep( seq( 0, ( length( n_i ) - 1 ) ), n_i )
  for( i in 1:sum(n_i) ){
    sub <- subject[ i ] + 1
    eta[ i ] <- eta[ i ] + Z_ij[ i, ]%*%kappa.%*%Gamma.%*%zeta.[sub, ]
  }
  
  prob <- exp( eta )/(1 + exp( eta) ) 
  Y <- rbinom( length(prob) , 1 , prob )
  
  data_sim <- list( "Y" = Y, "n_i" = n_i, "X" = X_ij[, -1], "U" = U, "Z" = Z_ij[,-1], "beta_bar_true" = beta_bar., "kappa_true" = kappa., "Gamma_true" = Gamma., "zeta_true" = zeta. )
  return( data_sim ) 
  
}

# Provides results for Sensitivity, Specificity, and MCC for simulated data 
# Make sure that the indicies are aligned 
select_perf <- function( selected, truth ){
  
  if( any( ! selected %in% c( 0, 1 ) ) ) {
    stop("Bad input: selected should be zero or one")
  }
  if( any( ! truth %in% c( 0, 1 ) ) ) {
    stop("Bad input: truth should be zero or one")
  }
  select <- which( selected == 1 )
  not_selected <- which( selected == 0 )
  included <- which( truth == 1 )
  excluded <- which( truth == 0 )
  
  TP <- sum( select %in% included )
  TN <- sum( not_selected %in% excluded )
  FP <- sum( select %in% excluded )
  FN <- sum( not_selected %in% included )
  sensitivity <- TP/( FN + TP )
  specificity <- TN/( FP + TN ) 
  mcc <- ( TP*TN - FP*FN )/(sqrt( TP + FP )*sqrt(TP + FN )*sqrt(TN + FP )*sqrt(TN + FN) )
  
  return( list( sens = sensitivity, spec = specificity, mcc = mcc ) ) 
}


# Determines inclusion for covariates based on X% credible intervals for x and z
selection_horse <- function( PGHorse_obj = PGHorse_obj, CI_threshold_x = 0.05, threshold_z = 0.1, burnin = 0){
  # arg checking
  if( CI_threshold_x > 1 | CI_threshold_x < 0){
    stop("Bad input: CI_threshold_x should be a probability")
  }
  if(   threshold_z < 0){
    stop("Bad input: threshold_z should be positive")
  }
  
  samples <- ncol(PGHorse_obj$mcmc$beta)
  if( burnin > samples ){
    stop("Bad input: burnin should be less than the number of iterations")
  } 
  samples_red <- samples - burnin
  left <- floor( CI_threshold_x * samples_red/2 )
  right <- ceiling((1 - CI_threshold_x/2) * samples_red )
  
  betaSort <- apply( PGHorse_obj$mcmc$beta[, (burnin + 1):samples], 1, sort,  decreasing = F)
  fix_left <- betaSort[ left, ]
  fix_right <- betaSort[ right  , ]
  fix_signals <- as.numeric(1 - ((fix_left <= 0) & (fix_right >= 0)))
  fix_signals[c(1,2,3)] <- 1
  
  kappa <- apply( PGHorse_obj$mcmc$K[, (samples_red + 1):samples], 1, mean)
  ran_signals <- ( kappa > threshold_z)*1
  ran_signals[1] <- 1
  
  return(list( betas = fix_signals, kappas = ran_signals ) )
}



# Wrapper function for main PGBVS with horseshoe priors
PGHorse <- function ( 
  iterations = 5000,
  thin = 10,
  Y = NULL,
  n_i = NULL, 
  W = NULL, 
  X = NULL, 
  U = NULL,  
  Z = NULL,  
  beta = NULL,  
  sigma_beta = NULL,  
  tau_beta = NULL,  
  vartheta_beta = NULL,  
  A_beta= NULL,    
  xi = NULL,
  mu = NULL,  
  K = NULL, 
  sigma_kappa = NULL, 
  tau_kappa = NULL, 
  vartheta_kappa = NULL, 
  A_kappa = NULL, 
  Gamma = NULL,
  zeta = NULL, 
  V_gamma = NULL ,
  gamma_0 = NULL ,
  m_0 = 0,
  v_0 = 5,
  tau = 5,
  seed = 1212, 
  warmstart = FALSE ){
  
  set.seed( seed )
  
  # Defense
  if( iterations%%1 != 0 | iterations <= 0){
    stop("Bad input: iterations should be a positive integer")
  }
  
  if( thin%%1 != 0 | thin < 0 ){
    stop("Bad input: thin should be a positive integer")
  }
 
  
  if( is.null( X ) ){
    stop("Missing input: Please provide a matrix of covariates")
  }
  
  if( is.null( Y ) ){
    stop("Missing input: Please provide a vector of outcomes")
  }
  
  if( is.null( n_i ) ){
    stop("Missing input: Please provide a vector of subject observations in order of Y")
  }
  
  if( is.null( U ) ){
    stop("Missing input: Please provide a vector/matrix for varying effects")
  }
  
  if( ( ncol( as.matrix( U ) ) != 1 ) & ( ncol( as.matrix( U ) ) != ncol( as.matrix( X ) ) + 1 ) ){
    stop("Bad input: Please provide a vector/matrix for varying effects with 1 or P columns")
  }
  
  if( ! is.null( beta ) & warmstart ){
    stop("Bad input: Please provide either warmstart or initial values for beta, not both")
  }
  
  # Call dependent libraries 
  library( spikeSlabGAM )
  
  # Pre-processing of input 
  # Vector that indicates which observations come from a subject. Elements are ij starts at 0
  subject <- rep( seq( 0, ( length( n_i ) - 1 ) ), n_i )
  # Input vector of dimensions for each subject in the data. Element is equal to the starting indicies for corresponding n. Last term is the number of observations
  subject_dems <- c( 0, cumsum( n_i ) )
  
  # Add intercept to X
  X <- cbind( 1, X )
  
  # If Z missing, set it to X
  if( is.null( Z ) ){
    Z <- X
  }else{
    Z <- cbind( 1, Z )
  }
  
  # Adjust P and D 
  P <- ncol( X )
  D <- ncol( Z )
  
  # Ustar - Matrix of spline functions. Rows indexed by ij. Columns indexed by sum r_p over p. ( Should be a list if r_p != r_p' )
  # Ustar_dems - Input vector of dimensions for each spline function. Element is equal to starting indicies for corresponding p. Last term is number of columns. Ustar_dems[ 0 ] = 0 and length = P + 1
  # Allows for only one input for U 
  if( ncol( as.matrix( U ) ) == 1 ){
    Ustar <- numeric()
    Ustar_dems <- c( 0 )
    tmp <- sm( U )
    for( p in 1:P ){
      Ustar <- cbind( Ustar, tmp )
      Ustar_dems <- c( Ustar_dems, ( Ustar_dems[ p ] + ncol( tmp ) ) )
    }
    U <- matrix( rep( U, P ), ncol = P , nrow = length( Y ) )
  }else{
    Ustar <- numeric()
    Ustar_dems <- c( 0 )
    for( p in 1:ncol( U ) ){
      tmp <- sm( U[ , p ] )
      Ustar <- cbind( Ustar, tmp )
      Ustar_dems <- c( Ustar_dems, ( Ustar_dems[ p ] + ncol( tmp ) ) )
    }
  }
  
  # Matrix of barX. Rows indexed by ij. Columns indexed by 2P ( x1u, x1, x2u, x2,...xPu, xP )
  Xbar <- numeric()
  for( p in 1:P ){
    tmp <- X[ , p]*U[ , p ]
    Xbar <- cbind( Xbar, tmp, X[ , p ] )
  }
  
  # Set remaining priors if not given 
  if( is.null( V_gamma ) ){
    V_gamma <- diag( D*( D - 1 )/2 )  
  }
  
  if( is.null( gamma_0 ) ){
    gamma_0 <- rep( 0 , D*( D - 1 )/2 )  
  }
  
  # Allocate output memory
  N <- length( n_i )
  samples <- floor( iterations/thin )     
  W. <-  matrix( 0, nrow = sum( n_i ), ncol = samples )               # - Matrix of MCMC samples for auxillary variables. Rows indexed ij. Columns indexed by MCMC sample
  beta. <- matrix( 0, nrow = 3*P, ncol = samples )                    # - Matrix of MCMC samples for beta. Rows indexed by beta_temp. Columns indexed by MCMC sample.
  sigma_beta. <- matrix( 0, nrow = 3*P, ncol = samples )              # - Matrix of MCMC samples for sigma_beta.
  tau_beta. <- rep( 0 , samples )                                     # - Vector of MCMC samples for tau_beta.
  vartheta_beta. <-  matrix( 0, nrow = 3*P, ncol = samples )          # - Matrix of MCMC samples for vartheta_beta.
  A_beta. <- rep( 0 , samples )                                       # - Vector of MCMC samples for A_beta.
  xi. <-  matrix( 0, nrow = ncol( Ustar ), ncol = samples )           # - Matrix of MCMC samples for parameter expansion for beta. Rows indexed by xi_temp. Columns indexed by MCMC sample.
  mu. <-  matrix( 0, nrow = ncol( Ustar ), ncol = samples )           # - Matrix of MCMC samples of means for each xi. Rows indexed by mu_tempp. Columns indexed by MCMC sample.
  K. <-  matrix( 0, nrow = D, ncol = samples )                        # - Matrix of MCMC samples for K. Rows indexed by K. Columns indexed by MCMC sample.
  sigma_kappa. <-  matrix( 0, nrow = D, ncol = samples )              # - Matrix of MCMC samples for sigma_kappa
  tau_kappa. <- rep( 0 , samples )                                    # - Vector of MCMC samples for tau_kappa
  vartheta_kappa. <-  matrix( 0, nrow = D, ncol = samples )           # - Matrix of MCMC samples for vartheta_kappa
  A_kappa. <- rep( 0 , samples )                                      # - Vector of MCMC samples for A_kappa
  Gamma. <- array( 0, dim = c( D, D, samples ) )                      # - Array of MCMC samples for Gamma. x and y are indexed by Gamma_temp. z indexed by MCMC sample
  zeta. <-  array( 0, dim = c( N, D, samples ) )                      # - Array of MCMC samples for zeta.  x and y indexed by zeta_temp. z indexed by MCMC sample
 
  # Adjust initial values given input 
  xi.[ ,1 ] <- ifelse( is.null( xi ), 1, xi )
  mu.[ , 1] <- if( is.null( mu ) ){ ifelse( rbinom( ncol( Ustar ), 1, 0.5 ) == 1, 1, -1 )}else{ mu }
  Gamma.[ , , 1] <- if( is.null( Gamma ) ){ diag( D ) }else{ Gamma }
  zeta.[ , , 1] <- if( is.null( zeta ) ){ rnorm( N*D ) }else{ zeta }
  beta.[ , 1] <- if( is.null( beta ) ){ c( 0, 1.1, 1.2, rep( c( 0, 0, 0 ), ( P - 1 ) ) )  }else{ beta }
  
  # Adjust if warmstart 
  if( warmstart ){
    library(glmnet)
    cv_fit <- cv.glmnet( Xbar, Y, alpha = 1, family = "binomial" )
    fit <- glmnet( Xbar , Y, alpha = 1, family = "binomial", lambda = cv_fit$lambda.1se )
    beta_init <- as.vector( fit$beta ) 
    beta_init[ 2 ] <- fit$a0
    beta_full_init <- rep( 0, P*3 )
    index <- 1
    for( i in 1:length( beta_full_init ) ){
      if( i%%3 != 1 ){
        beta_full_init[ i ] <- beta_init[ index ]
        index <- index + 1 
      }
    }
    beta.[ , 1] <- beta_full_init
  }
  
  sigma_beta.[ , 1]  <- if( is.null( sigma_beta ) ){ rep( 1, P*3 )  }else{ sigma_beta }
  tau_beta.[ 1 ] <- if( is.null( tau_beta ) ){ 1  }else{ tau_beta }
  vartheta_beta.[ , 1] <- if( is.null( vartheta_beta) ){ rep( 1, P*3 ) }else{ vartheta_beta }
  A_beta.[ 1 ]  <- if( is.null( A_beta ) ){ 1 }else{ A_beta }
  
 
  K.[ , 1] <- if( is.null( K ) ){ c( 1 , rep( 0, D - 1 ) ) }else{ K }
 
  sigma_kappa.[ , 1]  <- if( is.null( sigma_kappa ) ){ rep( 1, D )  }else{ sigma_kappa }
  tau_kappa.[ 1 ] <- if( is.null( tau_kappa ) ){  1 }else{ tau_kappa }
  vartheta_kappa.[ , 1] <- if( is.null( vartheta_kappa ) ){ rep( 1, D ) }else{ vartheta_kappa }
  A_kappa.[ 1 ]  <- if( is.null( A_kappa ) ){ 1  }else{ A_kappa }
 
  
  ###### RUN MODEL ######
  # State time 
  ptm <- proc.time()
  
  output <-  horsePGcpp( iterations, thin, Y, W. , subject, subject_dems, Ustar, Ustar_dems,
              Xbar, Z, beta. , sigma_beta. , tau_beta. , vartheta_beta. , A_beta. ,
              xi. , mu. , K., sigma_kappa. , tau_kappa. , vartheta_kappa. ,
              A_kappa. , Gamma. , zeta. , V_gamma, gamma_0, m_0, v_0, tau ) 
  
  # Stop the clock
  total_time <- proc.time() - ptm
  
  names( output ) <- c( 'W', 'beta', 'sigma_beta', 'tau_beta', 'vartheta_beta', 'A_beta', 'xi', 'mu', 'sigma_kappa', 'tau_kappa', 'vartheta_kappa', 'A_kappa', 'K',  'Gamma', 'zeta' )
  
  return( list( mcmc = output, total_time = total_time, Ustar = Ustar, Ustar_dems = Ustar_dems, Y = Y, X = X, Z = Z, U = U ) )
  }
  