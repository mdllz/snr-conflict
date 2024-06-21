functions{
  real shifted_lognormal_lpdf(matrix Dat, real int1, real eff, real sigma, real shift){
   vector[rows(Dat)] joint_density;
   for(j in 1:rows(Dat)){
    if(Dat[j,1]-shift > 0){
        joint_density[j] = lognormal_lpdf(Dat[j,1]-shift | int1+(eff*Dat[j,3]), sigma);
    } else {
    joint_density[j] = log(1e-10);
  }
  }
  return(sum(joint_density));
 }
 
 
  real shifted_lognormal2_lpdf(real rt, real mu, real sigma, real shift){
  if(rt-shift < 0){
    return log(1e-10);
  } else {
    return lognormal_lpdf(rt-shift | mu, sigma);
    }
  }
 
}
data{
  int<lower = 1> I;                     // participants
  int<lower = 1> N;                     // number of observations
  int<lower = 1, upper = I> subj[N];    // participant ids
  matrix[N,3] Y;
  int trialsum[I];   // participant id indicator
  int idx_start[I];

}
parameters{
  real alpha_mu;                    // population-level alpha
  real theta_mu;   
  real<lower=0> alpha_sigma; // population-level effect
  real<lower=0> theta_sigma; // population-level effect
  vector[I] alpha;                       
  vector[I] theta;
  real<lower=0> sigma[I];
  real<lower=0> sigma_sigma;
  real<lower=0> sigma_mu;
  real<lower=0> psi_mu;
  real<lower=0> psi_sigma;
  real<lower=0.1> psi[I];
}

model{
  alpha_mu ~ normal(-2,1);
  alpha_sigma ~ student_t(4,0,0.5);
  theta_mu ~ normal(0,1);               
  theta_sigma ~student_t(4,0,0.5);
  sigma_mu ~ normal(0,0.5);
  sigma_sigma ~ student_t(4,0,0.5);
  psi_mu ~ normal(.3, .2)T[0.1,0.5];
  psi_sigma ~ student_t(4,0,0.5);
  
  for(i in 1:I){
    sigma[i] ~ normal(sigma_mu, sigma_sigma);
    psi[i] ~ normal(psi_mu, psi_sigma);
    alpha[i] ~ normal(alpha_mu, alpha_sigma);
    theta[i] ~ normal(theta_mu, theta_sigma);
    // slice data matrix into individual participant matrices
    block(Y, idx_start[i], 1, trialsum[i], 3) ~ shifted_lognormal_lpdf(alpha[i], theta[i], sigma[i], psi[i]);
  }
}

generated quantities{
  vector[N] log_lik;
  vector[I] mu_c;
  vector[I] mu_ic;
  real y_pred[N];
  
  mu_c = alpha - (0.5*theta); // mu_c = alpha
  mu_ic =alpha + (0.5*theta);
  
  for(n in 1:N){
    log_lik[n] = shifted_lognormal2_lpdf(Y[n,1] | alpha[subj[n]] + (Y[n,3]*theta[subj[n]]), sigma[subj[n]], psi[subj[n]]);
    y_pred[n] = psi[subj[n]] + lognormal_rng(alpha[subj[n]] + Y[n,3]*theta[subj[n]], sigma[subj[n]]);
  }
  
}
