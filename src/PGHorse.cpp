/*  Implements Bayesian variables selection for random effects as well as varying-coefficients
 *  for logistic regression with repeated measures data using Polya Gamma Augmentation for 
 *  efficient sampling. 
 *  
 *  This version uses horseshoe priors for subject- and population-level effects
 *  
 *  Author Matt Koslovsky 2019
 *  
 *   References and Thanks to: 
 *
 *   Jesse Bennett Windle
 *   Forecasting High-Dimensional, Time-Varying Variance-Covariance Matrices
 *   with High-Frequency Data and Sampling Polya-Gamma Random Variates for
 *   Posterior Distributions Derived from Logistic Likelihoods  
 *   PhD Thesis, 2013   
 *
 *   Damien, P. & Walker, S. G. Sampling Truncated Normal, Beta, and Gamma Densities 
 *   Journal of Computational and Graphical Statistics, 2001, 10, 206-215
 *
 *   Chung, Y.: Simulation of truncated gamma variables 
 *   Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
 *
 *   Makalic, E. & Schmidt, D. F. High-Dimensional Bayesian Regularised Regression with the BayesReg Package 
 *   arXiv:1611.06649 [stat.CO], 2016 https://arxiv.org/pdf/1611.06649.pdf 
 */

//#include <omp.h>

#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
// [[Rcpp::plugins("cpp11")]]


// Mathematical constants
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

namespace help{

// Generate exponential distribution random variates
double exprnd(double mu)
{
  return -mu * (double)std::log(1.0 - (double)R::runif(0.0,1.0));
}

// Function a_n(x) defined in equations (12) and (13) of
// Bayesian inference for logistic models using Polya-Gamma latent variables
// Nicholas G. Polson, James G. Scott, Jesse Windle
// arXiv:1205.0310
//
// Also found in the PhD thesis of Windle (2013) in equations
// (2.14) and (2.15), page 24
double aterm(int n, double x, double t)
{
  double f = 0;
  if(x <= t) {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) + 1.5*(MATH_LOG_2_PI- (double)std::log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
  }
  else {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }    
  return (double)exp(f);
}

// Generate inverse gaussian random variates
double randinvg(double mu)
{
  // sampling
  double u = R::rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - (double)std::sqrt(4.0*mu*V + mu*mu * V*V) );
  
  if(R::runif(0.0,1.0) > mu /(mu+out)) {    
    out = mu*mu / out; 
  }    
  return out;
}

// Sample truncated gamma random variates
// Ref: Chung, Y.: Simulation of truncated gamma variables 
// Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
double truncgamma()
{
  double c = MATH_PI_2;
  double X, gX;
  
  bool done = false;
  while(!done)
  {
    X = help::exprnd(1.0) * 2.0 + c;
    gX = MATH_SQRT_PI_2 / (double)std::sqrt(X);
    
    if(R::runif(0.0,1.0) <= gX) {
      done = true;
    }
  }
  
  return X;  
}

// Sample truncated inverse Gaussian random variates
// Algorithm 4 in the Windle (2013) PhD thesis, page 129
double tinvgauss(double z, double t)
{
  double X, u;
  double mu = 1.0/z;
  
  // Pick sampler
  if(mu > t) {
    // Sampler based on truncated gamma 
    // Algorithm 3 in the Windle (2013) PhD thesis, page 128
    while(1) {
      u = R::runif(0.0, 1.0);
      X = 1.0 / help::truncgamma();
      
      if ((double)std::log(u) < (-z*z*0.5*X)) {
        break;
      }
    }
  }  
  else {
    // Rejection sampler
    X = t + 1.0;
    while(X >= t) {
      X = help::randinvg(mu);
    }
  }    
  return X;
}


// Sample PG(1,z)
// Based on Algorithm 6 in PhD thesis of Jesse Bennett Windle, 2013
// URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
double samplepg(double z)
{
  //  PG(b, z) = 0.25 * J*(b, z/2)
  z = (double)std::fabs((double)z) * 0.5;
  
  // Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
  double t = MATH_2_PI;
  
  // Compute p, q and the ratio q / (q + p)
  // (derived from scratch; derivation is not in the original paper)
  double K = z*z/2.0 + MATH_PI2/8.0;
  double logA = (double)std::log(4.0) - MATH_LOG_PI - z;
  double logK = (double)std::log(K);
  double Kt = K * t;
  double w = (double)std::sqrt(MATH_PI_2);
  
  double logf1 = logA + R::pnorm(w*(t*z - 1),0.0,1.0,1,1) + logK + Kt;
  double logf2 = logA + 2*z + R::pnorm(-w*(t*z+1),0.0,1.0,1,1) + logK + Kt;
  double p_over_q = (double)std::exp(logf1) + (double)std::exp(logf2);
  double ratio = 1.0 / (1.0 + p_over_q); 
  
  double u, X;
  
  // Main sampling loop; page 130 of the Windle PhD thesis
  while(1) 
  {
    // Step 1: Sample X ? g(x|z)
    u = R::runif(0.0,1.0);
    if(u < ratio) {
      // truncated exponential
      X = t + help::exprnd(1.0)/K;
    }
    else {
      // truncated Inverse Gaussian
      X = help::tinvgauss(z, t);
    }
    
    // Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U ? Sn(X|z) for an odd n or U > Sn(X|z) for an even n
    int i = 1;
    double Sn = help::aterm(0, X, t);
    double U = R::runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;
    
    while(1) 
    {
      Sn = Sn + asgn * help::aterm(i, X, t);
      
      // Accept if n is odd
      if(!even && (U <= Sn)) {
        X = X * 0.25;
        return X;
      }
      
      // Return to step 1 if n is even
      if(even && (U > Sn)) {
        break;
      }
      
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
  return X;
}

// Simulate MVT normal data
arma::mat mvrnormArma( int n, arma::vec mu, arma::mat sigma ) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn( n, ncols );
  return arma::repmat( mu, 1, n ).t()  + Y * arma::chol( sigma );
}

// Calculate mvt normal density (Not using it because it is slower than my code)
arma::vec dmvnrm_arma(arma::mat x,  
                      arma::rowvec mean,  
                      arma::mat sigma, 
                      bool logd = false) { 
  int n = x.n_rows;
  int xdim = x.n_cols;
  const double log2pi = std::log(2.0 * M_PI);
  arma::vec out(n);
  arma::mat rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  double rootisum = arma::sum(log(rooti.diag()));
  double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  
  for (int i=0; i < n; i++) {
    arma::vec z = rooti * arma::trans( x.row(i) - mean) ;    
    out(i)      = constants - 0.5 * arma::sum(z%z) + rootisum;     
  }  
  
  if (logd == false) {
    out = exp(out);
  }
  return(out);
}


// Make sX from Ustar, xi, Xbar, mu, Ustar_dems 
arma::mat make_sX( arma::mat Ustar, arma::vec xi, arma::mat Xbar, arma::vec Ustar_dems ){
  int P = Ustar_dems.size() - 1;
  int obs = Ustar.n_rows;
  
  arma::mat sX( obs, 3*P );
  sX.zeros();
  
  for( int p = 0; p < P; ++p ){
    arma::vec sum_p( obs );
    sum_p.zeros();
    
    // Get the range of columns corresponding to covariate p 
    arma::mat UstarXi = Ustar.cols( Ustar_dems[ p ], Ustar_dems[ p + 1 ] - 1) ;
    
    for( int ij = 0; ij < obs ; ++ij  ){
      arma::mat UstarXi_ind = UstarXi.row( ij ) * xi.subvec( Ustar_dems[ p ], Ustar_dems[ p + 1 ] - 1 ); 
      sum_p[ ij ] +=  UstarXi_ind[ 0 ];
    }
    
    // Order the covariates 
    sX.col( 3*p ) = sum_p % Xbar.col( 2*p + 1 );
    sX.col( 3*p + 1 ) = Xbar.col( 2*p );
    sX.col( 3*p + 2 ) = Xbar.col( 2*p + 1 ); 
    
  }
  return sX; 
}

// Make starX from B*, Ustar, x
arma::mat make_Xstar( arma::mat Ustar, arma::vec beta_temp, arma::mat Xbar, arma::vec Ustar_dems ){
  int P = Ustar_dems.size() - 1;
  int obs = Ustar.n_rows;
  int Rp = Ustar_dems[ Ustar_dems.size() - 1 ];
  
  arma::mat starX( obs, Rp );
  starX.zeros();
  
  for( int p = 0; p < P; ++p ){
    for( int rp = Ustar_dems[ p ]; rp < Ustar_dems[ p + 1 ]; ++rp ){
      starX.col( rp ) = beta_temp[ 3*p ] * Ustar.col( rp ) % Xbar.col( 2*p + 1 );
    }
  }  
  return starX; 
}

// Make Zhat from Z, K, zeta
arma::mat make_Zhat( arma::mat Z, arma::vec K_temp, arma::mat zeta_temp) {
  int D = Z.n_cols;
  int obs = Z.n_rows;
  
  arma::mat Zhat( obs, D*(D-1)/2 );
  Zhat.zeros();
  
  int count = 0;
  for( int m = 0; m < D - 1; ++m ){
    for( int l = m + 1; l < D; ++l ){
      Zhat.col( count ) = K_temp[ l ] * Z.col( l ) % zeta_temp.col( m );
      count += 1;
    }
  }
  return Zhat; 
}
// Make Zhat from Z, K, zeta and lambda
// This only makes a Zhat if its corresponding random effect components are included in the model 
arma::mat make_Zhat_lambda( arma::mat Z, arma::vec K_temp, arma::mat zeta_temp, arma::vec lambda_temp, arma::vec subject ) {
  int D = Z.n_cols;
  int D_lambda = sum( lambda_temp );
  int obs = Z.n_rows;
  
  arma::mat Zhat( obs, D_lambda*(D_lambda-1)/2 );
  Zhat.zeros();
  
  int count = 0;
  for( int m = 0; m < D - 1; ++m ){
    
    for( int l = m + 1; l < D; ++l ){
      if( lambda_temp[ l ] != 0 & lambda_temp[ m ] != 0 ){
        for( int j = 0; j < obs; ++j ){
          int sub = subject[ j ];
          Zhat( j, count ) = K_temp[ l ] * Z( j, l ) * zeta_temp( sub, m );
        }
        //Zhat.col( count ) = K_temp[ l ] * Z.col( l ) % zeta_temp.col( m );
        count += 1;
        
      }
    }
  }
  return Zhat; 
}

// Make Zstar from Z, zeta, Gamma
arma::mat make_Zstar( arma::mat Z, arma::mat zeta_temp, arma::mat Gamma_temp, arma::vec subject ) {
  int D = Z.n_cols;
  int obs = Z.n_rows;
  arma::mat Zstar( obs, D );
  Zstar.zeros();
  
  for( int i = 0; i < obs ; ++i ){
    int sub = subject[ i ];
    for( int m = 0; m < D ; ++m ){
      double sum_gamma_zeta = 0;
      for( int l = 0; l < m  ; ++l ){
        sum_gamma_zeta += Gamma_temp( m, l ) * zeta_temp( sub, l ); 
      }
      Zstar( i, m ) = Z( i, m ) * ( zeta_temp( sub, m ) + sum_gamma_zeta );
    }
  }
  
  
  // Return output 
  return Zstar; 
}


// Sample from an integrer vector 
int sample_cpp( IntegerVector x ){
  // Calling sample()
  Function f( "sample" );
  IntegerVector sampled = f( x, Named( "size" ) = 1 );
  return sampled[ 0 ];
}

// Function :: Log-likelihood: h_ij = k_ij/w_ij and h ~ N( psi, Omega), k_ij = y_ij - 1/2
double log_like_pg( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject)
{
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  arma::mat sX( obs, S );
  sX.zeros();
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  double log_like = 0;
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make psi and calculate the log-likelihood contribution 
  for( int i = 0; i < obs; ++i ){
    int sub = subject[ i ];
    arma::mat psi_val = sX.row( i )*beta_temp + Z.row( i )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() ; 
    double Winv = 1/W[i];
    log_like += -0.50*log( 2*M_PI*Winv ) - 1/( 2*Winv )*pow( H[i] - psi_val[0], 2 );
  }
  
  // Return output
  return log_like;
}




// Function :: Calculate beta-binomial log-density (individual)
double log_beta_binomial_cpp( double indicate, double a, double b ){
  
  double post_a = indicate + a;
  double post_b = 1 - indicate + b;
  double log_indicator = lgamma( post_a ) + lgamma( post_b ) - lgamma( post_a + post_b ) - ( lgamma( a ) + lgamma( b ) - lgamma( a + b ) );
  
  // Return output
  return log_indicator ;
}

// Function :: Calculate normal log-density ( univariate )
double log_normal_cpp( double value, double mu, double sigma2 ){
  double log_normal = -0.50*log( 2*M_PI*sigma2 ) - 1/( 2*sigma2 )*pow( value - mu ,2 );
  
  // Return output
  return log_normal; 
}

// Function :: Calculate folded normal log-density ( univariate )
double log_fold_normal_cpp( double value, double mu, double sigma2 ){
  double log_fold_normal = log( pow( 2*M_PI*sigma2, -0.50 )*exp( - 1/( 2*sigma2 )*pow( value - mu ,2 )) + pow( 2*M_PI*sigma2, -0.50 )*exp( - 1/( 2*sigma2 )*pow( value + mu ,2 ) ) );
  
  // Return output
  return log_fold_normal; 
}

// Function :: Rescale beta_temp and xi_temp
List rescaled( arma::vec beta_temp, arma::vec xi_temp, arma::vec Ustar_dems){
  int P = Ustar_dems.size() - 1;
  
  for( int p = 0; p < P; ++p ){
    int Rp = Ustar_dems[ p + 1 ] - Ustar_dems[ p ]; 
    double summed = 0;
    
    for( int r = Ustar_dems[ p ]; r < Ustar_dems[ p + 1 ]; ++r){
      summed += std::abs( xi_temp[ r ] );
    }
    
    for( int r = Ustar_dems[ p ]; r < Ustar_dems[ p + 1 ]; ++r){
      xi_temp[ r ] = ( Rp/summed )*xi_temp[ r ];
    }
    
    beta_temp[ 3*p ] = ( summed/Rp )*beta_temp[ 3*p ];
  }
  
  List rescaled( 2 );
  rescaled[ 0 ] = beta_temp;
  rescaled[ 1 ] = xi_temp;
  return rescaled;
}

// Updates

// Update auxillary parameters W 
arma::vec update_W( arma::mat Ustar, arma::vec xi, arma::mat Xbar, arma::vec Ustar_dems, arma::vec beta_temp, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject ){
  int obs = subject.size();
  int S = beta_temp.size();
  int D = K_temp.size();
  
  // Make a home for W updates
  arma::vec updated_W( obs ); 
  updated_W.zeros();
  
  // Make sX
  arma::mat sX( obs, S); 
  sX.zeros();
  sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make K matrix
  arma::mat K_mat( D, D );
  K_mat.zeros();
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Update each W individually
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    arma::mat phi_j(1,1);
    phi_j = sX.row(j)*beta_temp + Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() ;
    updated_W[ j ] = help::samplepg( phi_j[ 0 ] );
  }
  
  return updated_W;
  
}

// Function :: Update beta horseshoe
arma::vec update_beta( arma::vec Y, arma::vec W, arma::vec subject, arma::vec beta_temp, arma::vec sigma_beta_temp, double tau_beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject_dems, arma::vec mu_temp, double tau){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  
  arma::mat beta_update( 1, S );
  beta_update.zeros();
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sXand Sigma_beta
  arma::mat sX( obs, S); 
  sX.zeros();
  sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  arma::mat Sigma_beta( S, S ); 
  Sigma_beta.zeros();
  
  for( int s = 0; s < S; ++s){
    Sigma_beta( s, s ) = 1/ ( tau_beta_temp * sigma_beta_temp[ s ] );
  }
  Sigma_beta( 0, 0 ) = 1/tau; 
  Sigma_beta( 1, 1 ) = 1/tau; 
  Sigma_beta( 2, 2 ) = 1/tau; 
  
  // Make V_xi and mu_xi (and inside)
  arma::mat V_beta( S, S );
  arma::mat mu_beta( S, 1 );
  arma::mat mu_beta_inside( S, 1 );
  V_beta.zeros();
  mu_beta.zeros();
  mu_beta_inside.zeros();
  
  // Update for each individuals zeta_temp
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    
    V_beta += W[ j ]*sX.row( j ).t()*sX.row( j );
    mu_beta_inside += W[ j ]*sX.row( j ).t()*( H[ j ] -  Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() );
  }
  
  V_beta += Sigma_beta; 
  V_beta = inv( V_beta );
  mu_beta = mu_beta_inside;
  
  mu_beta = V_beta*mu_beta;
  
  beta_update = help::mvrnormArma( 1, mu_beta, V_beta ); 
  
  return beta_update.t();
}  

// Function :: Update sigma_beta horseshoe
// Function :: Update tau_beta horseshoe
// Function :: Update vartheta_beta horseshoe
// Function :: Update A_beta horseshoe

// Function :: Update xi 
arma::vec update_xi( arma::vec Y, arma::vec W, arma::vec subject, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject_dems, arma::vec mu_temp){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int P = Ustar_dems.size() - 1;
  int Rp = Ustar_dems[ Ustar_dems.size() - 1 ];
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  
  arma::mat xi_update( 1, Rp );
  xi_update.zeros();
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make Identity
  arma::mat I( Rp, Rp );
  I.zeros();
  for( int r = 0; r < Rp; ++r){
    I( r, r ) = 1;
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make X*
  arma::mat Xstar( obs, Rp );
  Xstar.zeros();
  Xstar = help::make_Xstar( Ustar, beta_temp, Xbar, Ustar_dems );
  
  // Make V_xi and mu_xi (and inside)
  arma::mat V_xi( Rp, Rp );
  arma::mat mu_xi( Rp, 1 );
  arma::mat mu_xi_inside( Rp, 1 );
  V_xi.zeros();
  mu_xi.zeros();
  mu_xi_inside.zeros();
  
  // Make beta_bar
  arma::vec beta_bar( S - P );
  for( int p = 0; p < P; ++p ){
    beta_bar[ 2*p ] = beta_temp[ 3*p + 1 ];
    beta_bar[ 2*p + 1 ] = beta_temp[ 3*p + 2 ] ;
  }
  
  // Update for each individuals xi
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    V_xi += W[ j ]*Xstar.row( j ).t()*Xstar.row( j );
    mu_xi_inside += W[ j ]*Xstar.row( j ).t()*( H[ j ] -  Xbar.row(j)*beta_bar - Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() );
  }
  
  V_xi += I; 
  V_xi = inv( V_xi );
  mu_xi = mu_xi_inside + mu_temp;
  mu_xi = V_xi*mu_xi;
  xi_update = help::mvrnormArma( 1, mu_xi, V_xi ); 
  
  return xi_update.t();
}

// Function :: Update mu_rp 
arma::vec update_mu_rp( arma::vec xi_temp ){
  int Rp = xi_temp.size();
  arma::vec xi_update( Rp );
  
  for( int rp = 0 ; rp < Rp; ++rp){
    
    // Get probability based on xi_temp 
    double prob = 1/(1 + exp( -2*xi_temp[ rp ] ) );
    
    // +- 1 based on sampled mu_ind
    int mu_ind = rbinom( 1, 1, prob )[0];
    
    if( mu_ind == 1){
      xi_update[ rp ] = 1;
    }else{
      xi_update[ rp ] = -1;
    }
    
  }
  return xi_update;
}

// Function :: Update sigma_beta
arma::vec update_sigma_beta( arma::vec beta_temp, double tau_beta_temp, arma::vec vartheta_beta_temp  ){
  int S = beta_temp.size();
  
  arma::vec sigma_beta_update( S );
  sigma_beta_update.zeros();
  
  arma::mat b_post( 1, 1 );
  b_post.zeros();
  
  for( int s = 0; s < S; ++s ){
    double a_post = 1;
    b_post =  pow( beta_temp[ s ], 2 )/( 2 * tau_beta_temp) + 1/vartheta_beta_temp;
    double hold = rgamma(1, a_post, 1/b_post[0] )[0];
    sigma_beta_update[ s ] = 1/hold;
  }
  return sigma_beta_update;
}

// Function :: Update tau_beta
double update_tau_beta( arma::vec beta_temp, double A_beta_temp, arma::vec sigma_beta_temp  ){
  int S = beta_temp.size();
  
  double tau_beta_update = 0;
  
  double b_post = 0; 
  for( int s = 0; s < S; ++s ){
    b_post += pow( beta_temp[ s ], 2 )/( 2 * sigma_beta_temp[ s ] );
  }
  
  b_post += 1/A_beta_temp;
  double a_post = ( S + 1 )/2;
  
  tau_beta_update = 1/rgamma(1, a_post, 1/b_post )[0];
  return tau_beta_update;
}

// Function :: Update vartheta_beta
arma::vec update_vartheta_beta( arma::vec sigma_beta_temp ){
  int S = sigma_beta_temp.size();
  
  arma::vec vartheta_beta_update( S );
  vartheta_beta_update.zeros();
  
  for( int s = 0; s < S; ++s ){
    double a_post = 1;
    double b_post =  1 + 1/sigma_beta_temp[ s ];
    vartheta_beta_update( s ) = 1/rgamma(1, a_post, 1/b_post )[ 0 ];
  }
  return vartheta_beta_update;
}

// Function :: Update A_beta
double update_A_beta( double tau_beta_temp ){ 
  
  double A_beta_update = 0;
  
  double b_post = 1 + 1/tau_beta_temp;
  double a_post = 1;
  
  A_beta_update = 1/rgamma(1, a_post, 1/b_post )[0];
  return A_beta_update;
}

// Updates selected K and lambda Within step
arma::vec update_K( arma::vec Y, arma::vec subject, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp,  arma::mat Gamma_temp, arma::mat zeta_temp, double m_0, arma::vec sigma_kappa, double tau_kappa, double v0 ){
  int D = K_temp.size();
  int obs = W.size();
  int S = beta_temp.size();
  arma::vec H( obs );
  H.zeros();
  arma::mat sX( obs, S );
  sX.zeros();
  
  // Make home for K updates
  arma::vec K_update( D );
  K_update.zeros();
  
  // Get Z*
  arma::mat Zstar = help::make_Zstar( Z, zeta_temp, Gamma_temp, subject );
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  for( int d = 0; d < D; ++d ){
    
    double var_kd = 0;
    arma::mat mean_kd( 1, 1 );
    mean_kd.zeros();
    
    // Iterate over the subjects 
    for( int j = 0; j < obs; ++j ){
      mean_kd += W[ j ]*Zstar( j, d )*( H[ j ] - sX.row( j )*beta_temp - Zstar.row( j )*K_temp + Zstar( j, d )*K_temp[ d ] );
      var_kd += W[ j ]*pow( Zstar( j, d ), 2 ); 
    }
    
    if( d != 0 ){
      var_kd += 1/( sigma_kappa[ d ]*tau_kappa ) ;
      var_kd = 1/var_kd;
      mean_kd += m_0/( sigma_kappa[ d ]*tau_kappa );
      mean_kd = var_kd*mean_kd[ 0 ]; 
    }
    if( d == 0 ){
      var_kd += 1/v0 ;
      var_kd = 1/var_kd;
      mean_kd += m_0/v0;
      mean_kd = var_kd*mean_kd[ 0 ]; 
    }
    
    K_update[ d ] = std::abs( mean_kd[ 0 ] + sqrt( var_kd )*rnorm( 1 )[ 0 ] ) ;
  }
  
  return K_update;
}


// Function :: Update sigma_kappa
arma::vec update_sigma_kappa( arma::vec K_temp, double tau_kappa_temp, arma::vec vartheta_kappa_temp  ){
  int D = K_temp.size();
  
  arma::vec sigma_kappa_update( D );
  sigma_kappa_update.zeros();
  
  arma::mat b_post( 1, 1 );
  b_post.zeros();
  
  for( int d = 0; d < D; ++d ){
    double a_post = 1;
    b_post =  pow( K_temp[ d ], 2 )/( 2 * tau_kappa_temp ) + 1/vartheta_kappa_temp;
    sigma_kappa_update( d ) = 1/rgamma(1, a_post, 1/b_post[0] )[0];
  }
  
  return sigma_kappa_update;
}


// Function :: Update tau_kappa
double update_tau_kappa( arma::vec K_temp, double A_kappa_temp, arma::vec sigma_kappa_temp  ){
  int D = K_temp.size();
  
  double tau_kappa_update = 0;
  
  double b_post = 0; 
  for( int d = 0; d < D; ++d ){
    b_post += pow( K_temp[ d ], 2 )/( 2 * sigma_kappa_temp[ d ] );
  }
  
  b_post += 1/A_kappa_temp;
  double a_post = ( D + 1 )/2;
  
  tau_kappa_update = 1/rgamma(1, a_post, 1/b_post )[0];
  return tau_kappa_update;
}


// Function :: Update vartheta_kappa
arma::vec update_vartheta_kappa( arma::vec sigma_kappa_temp ){
  int D = sigma_kappa_temp.size();
  
  arma::vec vartheta_kappa_update( D );
  vartheta_kappa_update.zeros();
  
  for( int d = 0; d < D; ++d ){
    double a_post = 1;
    double b_post =  1 + 1/sigma_kappa_temp[ d ];
    vartheta_kappa_update( d ) = 1/rgamma(1, a_post, 1/b_post )[0];
  }
  return vartheta_kappa_update;
}

// Function :: Update A_kappa
double update_A_kappa( double tau_kappa_temp ){ 
  
  double A_kappa_update = 0;
  
  double b_post = 1 + 1/tau_kappa_temp;
  double a_post = 1;
  
  A_kappa_update = 1/rgamma(1, a_post, 1/b_post )[0];
  return A_kappa_update;
}


// Function :: Update Gamma
arma::mat update_Gamma( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat zeta_temp, arma::mat V_gamma, arma::vec gamma_0, arma::vec subject){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  arma::mat sX( obs, S );
  sX.zeros();
  int D = K_temp.size();
  
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::mat G_inv( D, D );
  G_inv.zeros();
  arma::vec H( obs );
  H.zeros();
  
  // Get number of included random effects (all in this case)
  arma::vec lambda_temp( D );
  lambda_temp.ones();
  
  int D_lambda = D;
  
  // Set number of gamma parameters
  int num_gammas = D_lambda*(D_lambda-1)/2;
  
  // Set mean and variance for update based on included K
  arma::mat Vhat_gamma( num_gammas, num_gammas );
  arma::vec gamma_hat( num_gammas );
  arma::mat gamma_hat_inside( num_gammas, 1 );
  
  Vhat_gamma.zeros();
  gamma_hat.zeros();
  gamma_hat_inside.zeros();
  
  // Set up structure of Gamma
  arma::mat Gamma_update( D, D );
  Gamma_update.zeros();
  for( int d = 0; d < D; ++d ){
    Gamma_update( d, d ) = 1;
  }
  
  // Resize V_gamma and invert
  // Works under the assumption that the gammas are independent a priori (ie just takes diagonal terms)
  arma::mat V_gamma_red_inv( D_lambda*( D_lambda - 1 )/2, D_lambda*( D_lambda - 1 )/2 );
  V_gamma_red_inv.zeros();
  
  arma::vec gamma_0_red( D_lambda*( D_lambda - 1 )/2 );
  gamma_0_red.zeros();
  
  int count = 0;
  int count_red = 0; 
  for( int m = 0; m < D - 1; ++m ){
    for( int l = m + 1; l < D; ++l ){
      if( lambda_temp[ l ] != 0 & lambda_temp[ m ] != 0 ){
        V_gamma_red_inv( count_red, count_red ) = 1/V_gamma( count, count );
        gamma_0_red[ count_red ] = gamma_0[ count ];
        count_red += 1;
      }
      count += 1; 
    }
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make Zhat_lambda
  arma::mat Zhat_lambda( obs, D_lambda*(D_lambda-1)/2 );
  Zhat_lambda.zeros();
  Zhat_lambda =  help::make_Zhat_lambda( Z, K_temp, zeta_temp, lambda_temp, subject );
  
  // Update gammas currently included in the model ( based on lambda_temp )
  // Sum over all of the observations ij 
  for( int j = 0; j < obs; ++j ){
    Vhat_gamma += W[ j ]*(Zhat_lambda.row( j ).t()*Zhat_lambda.row( j )); 
    gamma_hat_inside +=  W[ j ]*Zhat_lambda.row( j ).t()*( H[ j ] -  sX.row( j )*beta_temp );
  }
  
  Vhat_gamma += V_gamma_red_inv;
  Vhat_gamma = inv( Vhat_gamma );
  
  gamma_hat_inside += V_gamma_red_inv*gamma_0_red;
  gamma_hat =  Vhat_gamma*gamma_hat_inside ;
  
  arma::mat gamma_elements( D_lambda*( D_lambda - 1 )/2, 1 );
  gamma_elements.zeros();
  gamma_elements = help::mvrnormArma( 1, gamma_elements, Vhat_gamma );
  
  // Append updated elements to the Gamma matrix appropriately 
  count_red = 0; 
  for( int m = 0; m < D - 1; ++m ){
    for( int l = m + 1; l < D; ++l ){
      if( lambda_temp[ l ] != 0 & lambda_temp[ m ] != 0 ){
        Gamma_update( l, m ) = gamma_elements[ count_red ];
        count_red += 1;
      }
    }
  }
  return Gamma_update;
}


// Function :: Update zeta
arma::mat update_zeta( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::vec subject_dems){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int N = subject_dems.size() - 1;
  arma::mat sX( obs, S );
  sX.zeros();
  int D = K_temp.size();
  arma::mat V_zeta( D, D );
  arma::vec mu_zeta( D );
  mu_zeta.zeros();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::mat I( D, D );
  I.zeros();
  arma::vec H( obs );
  H.zeros();
  arma::mat mu_zeta_inside( 1, D );
  
  arma::mat zeta_update( N, D );
  zeta_update.zeros();
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make I
  for( int i = 0; i < D; ++i ){
    I( i, i ) = 1;
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Update for each individuals zeta_temp
  for( int n = 0; n < N; ++n ){
    
    // Pull the Z that are associated with subject i 
    arma::mat z_i = Z.rows( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    // Get the number of observations for subject i;
    int obs_i = z_i.n_rows;
    
    // Pull the w that are associated with subject i 
    arma::vec w_i = W.subvec( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    //Pull the h that are associated with subject i 
    arma::vec h_i = H.subvec( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    // Pull the sX that are associated with subject i
    arma::mat sX_i = sX.rows( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    // Make V_zeta and mu_zeta 
    mu_zeta_inside.zeros();
    V_zeta.zeros();
    
    for( int j = 0; j < obs_i; ++j){
      V_zeta += w_i[ j ]*Gamma_temp.t()*K_mat*z_i.row( j ).t()*z_i.row( j )*K_mat*Gamma_temp;
      mu_zeta_inside += w_i[ j ]*( h_i[ j ] -  sX_i.row( j )*beta_temp )*z_i.row( j )*K_mat*Gamma_temp;
    }
    
    V_zeta += I; 
    
    V_zeta = inv( V_zeta );
    
    mu_zeta = (mu_zeta_inside*V_zeta).t();
    
    zeta_update.row( n ) = help::mvrnormArma( 1, mu_zeta, V_zeta ); 
    
  }
  
  return zeta_update;
}



}  // For namespace 'help'

// Function :: MCMC algorithm
// [[Rcpp::export]]
List horsePGcpp(
    int iterations,             // Number of iterations
    int thin,                   // How often to thin to make the output less
    arma::vec Y,                // Y - Vector of outcomes. Indexed ij.
    arma::mat W,                // W - Matrix of MCMC samples for auxillary variables. Rows indexed ij. Columns indexed by MCMC sample
    arma::vec subject,          // subject - Vector that indicates which observations come from a subject. Elements are ij
    arma::vec subject_dems,     // subject_dems - Input vector of dimensions for each subject in the data. Element is equal to the starting indicies for corresponding n. Last term is the number of observations
    arma::mat Ustar,            // Ustar - Matrix of spline functions. Rows indexed by ij. Columns indexed by sum r_p over p. ( Should be a list if r_p != r_p' )
    arma::vec Ustar_dems,       // Ustar_dems - Input vector of dimensions for each spline function. Element is equal to starting indicies for corresponding p. Last term is number of columns. Ustar_dems[ 0 ] = 0 and length = P + 1
    arma::mat Xbar,             // Xbar - Matrix of barX. Rows indexed by ij. Columns indexed by 2P ( x1u, x1, x2u, x2,...xPu, xP )
    arma::mat Z,                // Z - Matrix of random covariates for each subject. Columns indexed by D. Rows indexed by ij
    arma::mat beta,             // beta - Matrix of MCMC samples for beta. Rows indexed by beta_temp. Columns indexed by MCMC sample.
    arma::mat sigma_beta,       // Matrix of MCMC samples for sigma_beta.
    arma::vec tau_beta,         // Vector of MCMC samples for tau_beta.
    arma::mat vartheta_beta,    // Matrix of MCMC samples for vartheta_beta.
    arma::vec A_beta,           // Vector of MCMC samples for A_beta.
    arma::mat xi,               // xi - Matrix of MCMC samples for parameter expansion for beta. Rows indexed by xi_temp. Columns indexed by MCMC sample.
    arma::mat mu,               // mu - Matrix of MCMC samples of means for each xi. Rows indexed by mu_tempp. Columns indexed by MCMC sample.
    arma::mat K,                // K - Matrix of MCMC samples for K. Rows indexed by K. Columns indexed by MCMC sample.
    arma::mat sigma_kappa,      // Matrix of MCMC samples for sigma_kappa
    arma::vec tau_kappa,        // Vector of MCMC samples for tau_kappa
    arma::mat vartheta_kappa,   // Matrix of MCMC samples for vartheta_kappa
    arma::vec A_kappa,          // Vector of MCMC samples for A_kappa
    arma::cube Gamma,           // Gamma - Array of MCMC samples for Gamma. x and y are indexed by Gamma_temp. z indexed by MCMC sample
    arma::cube zeta,            // zeta - Array of MCMC samples for zeta.  x and y indexed by zeta_temp. z indexed by MCMC sample
    arma::mat V_gamma,           // V_gamma - Matrix of Gamma variance-covariance priors for MVT normal
    arma::vec gamma_0,           // gamma_0 - Vector of Gamma mean priors for MVT normal
    double m_0,                  // m_0 - double for folded normal prior mean
    double v_0,                  // v_0 - double for folded normal prior variance
    double tau                  // tau - variance for beta intercept
){
  
  // Initiate memory for List updates
  List return_rescaled( 2 );
  
  // Set temporary data to enable thinning
  arma::vec W_temp = W.col( 0 );            // W_temp - Vector of current auxillary parameters. Indexed by ij.
  arma::vec beta_temp = beta.col( 0 );      // beta_temp - Vector of coefficients for fixed effects (includes terms forced into model). Elements indexed by 3P. B*_p,B^0_p,B_p0,...
  arma::vec sigma_beta_temp = sigma_beta.col( 0 );  // sigma_beta_temp - Vector of local variances. Indexed by T = 3P
  double tau_beta_temp = tau_beta[ 0 ];             // tau_beta_temp - Global variance for beta terms. 
  arma::vec vartheta_beta_temp = vartheta_beta.col( 0 ); // vartheta_beta_temp - Vector of local auxillary parameters for betas indexed by T = 3P
  double A_beta_temp = A_beta[ 0 ];         // A_beta_temp - Global auxillary term for beta                   
  arma::vec xi_temp = xi.col( 0 );          // xi_temp - Vector of parameter expansion for beta. Elements indexed by sum r_p over p.
  arma::vec mu_temp = mu.col( 0 );          // mu_temp - Vector of means for each xi. Elements indexed by sum r_p over p.
  arma::vec K_temp = K.col( 0 );            // K_temp - Vector of coefficients for random effects (includes terms forced into model). Elements indexed by D.
  arma::vec sigma_kappa_temp = sigma_kappa.col( 0 ); // sigma_kappa_temp - Vector of local variances. Indexed by D
  double tau_kappa_temp = tau_kappa[ 0 ];            // tau_kappa_temp - Global variance for kappa terms. 
  arma::vec vartheta_kappa_temp = vartheta_kappa.col( 0 );    // vartheta_kappa_temp - Vector of local auxillary parameters for kappas indexed by D 
  double A_kappa_temp =  A_kappa[ 0 ];      // A_kappa_temp - Global auxillary term for kappa       
  arma::mat Gamma_temp = Gamma.slice( 0 );  // Gamma_temp - Lower trianglular matrix for random effects. Columns and rows indexed by D.
  arma::mat zeta_temp = zeta.slice( 0 );    // zeta_temp - Matrix of random effects for each subject. Columns indexed by D. Rows indexed by i.
  
  // Looping over the number of iterations specified by user
  for( int iter = 0; iter < iterations; ++iter ){
    
    // Update W (stays)
    W_temp = help::update_W( Ustar, xi_temp, Xbar, Ustar_dems, beta_temp, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject );
    
    // Within beta (make new)
    beta_temp = help::update_beta( Y, W_temp, subject, beta_temp, sigma_beta_temp, tau_beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject_dems, mu_temp, tau );
    
    // Update sigma beta
    sigma_beta_temp = help::update_sigma_beta( beta_temp, tau_beta_temp, vartheta_beta_temp  );
    
    // Update tau beta
    tau_beta_temp =  help::update_tau_beta( beta_temp, A_beta_temp, sigma_beta_temp );
    
    // Update vartheta beta
    vartheta_beta_temp = help::update_vartheta_beta( sigma_beta_temp );
    
    // Update A beta
    A_beta_temp = help::update_A_beta( tau_beta_temp );             
     
    // Update xi (stays)
    xi_temp = help::update_xi( Y, W_temp, subject, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject_dems, mu_temp);
    
    // Rescale
    return_rescaled = help::rescaled( beta_temp, xi_temp, Ustar_dems);
    beta_temp = as<arma::vec>( return_rescaled[ 0 ] );
    xi_temp = as<arma::vec>( return_rescaled[ 1 ] );
    
    // Update mu
    mu_temp = help::update_mu_rp( xi_temp );
    
    // Update kappa 
    K_temp =  help::update_K( Y, subject, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, m_0, sigma_kappa_temp, tau_kappa_temp, v_0 );
    
    // Update sigma kappa 
    sigma_kappa_temp = help::update_sigma_kappa( K_temp, tau_kappa_temp, vartheta_kappa_temp );
    
    // Update tau kappa 
    tau_kappa_temp = help::update_tau_kappa( K_temp, A_kappa_temp, sigma_kappa_temp );
     
    // Update vartheta kappa
    vartheta_kappa_temp = help::update_vartheta_kappa( sigma_kappa_temp );
    
    // Update A kappa
    A_kappa_temp = help::update_A_kappa( tau_kappa_temp );
    
    //Update Gamma
    Gamma_temp = help::update_Gamma( Y, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, zeta_temp, V_gamma, gamma_0, subject);
    
    // Update zeta
    zeta_temp = help::update_zeta( Y, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, subject_dems );
    
    // Set the starting values for the next iteration
    if( ( iter + 1 ) % thin == 0 ){
      W.col( ( iter + 1 )/thin - 1 ) = W_temp;            // W_temp - Vector of current auxillary parameters. Indexed by ij.
      beta.col( ( iter + 1 )/thin - 1 ) = beta_temp;      // beta_temp - Vector of coefficients for fixed effects (includes terms forced into model). Elements indexed by 3P. B*_p,B^0_p,B_p0,...
      sigma_beta.col( ( iter + 1 )/thin - 1 ) = sigma_beta_temp;
      tau_beta[ ( iter + 1 )/thin - 1 ] = tau_beta_temp;
      vartheta_beta.col( ( iter + 1 )/thin - 1 ) = vartheta_beta_temp;
      A_beta[ ( iter + 1 )/thin - 1 ] = A_beta_temp;
      xi.col( ( iter + 1 )/thin - 1 ) = xi_temp;          // xi_temp - Vector of parameter expansion for beta. Elements indexed by sum r_p over p.
      mu.col( ( iter + 1 )/thin - 1 ) = mu_temp;          // mu_temp - Vector of means for each xi. Elements indexed by sum r_p over p.
      K.col( ( iter + 1 )/thin - 1 ) = K_temp;            // K_temp - Vector of coefficients for random effects (includes terms forced into model). Elements indexed by D.
      sigma_kappa.col( ( iter + 1 )/thin - 1 ) = sigma_kappa_temp;
      tau_kappa[ ( iter + 1 )/thin - 1 ] = tau_kappa_temp;
      vartheta_kappa.col( ( iter + 1 )/thin - 1 ) = vartheta_kappa_temp;
      A_kappa[ ( iter + 1 )/thin - 1 ] = A_kappa_temp;
      Gamma.slice( ( iter + 1 )/thin - 1 ) = Gamma_temp;  // Gamma_temp - Lower trianglular matrix for random effects. Columns and rows indexed by D.
      zeta.slice( ( iter + 1 )/thin - 1 ) = zeta_temp;    // zeta_temp - Matrix of random effects for each subject. Columns indexed by D. Rows indexed by i.
    }
    
    // Print out progress
    double printer = iter % 50;
    
    if( printer == 0 ){
      Rcpp::Rcout << "Iteration = " << iter << std::endl;
    }
  }
  
  // Return output
  List output( 15 );
  output[ 0 ] = W;
  output[ 1 ] = beta;
  output[ 2 ] = sigma_beta;
  output[ 3 ] = tau_beta;
  output[ 4 ] = vartheta_beta;
  output[ 5 ] = A_beta;
  output[ 6 ] = xi;
  output[ 7 ] = mu;
  output[ 8 ] = sigma_kappa;
  output[ 9 ] = tau_kappa;
  output[ 10 ] = vartheta_kappa;
  output[ 11 ] = A_kappa;
  output[ 12 ] = K;
  output[ 13 ] = Gamma;
  output[ 14 ] = zeta;
  
  return output ;
}

