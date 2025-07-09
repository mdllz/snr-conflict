data{
  int<lower = 1> I;                     // participants
  int<lower = 1> N;                     // number of observations
  int<lower = 1, upper = I> subj[N];    // participant ids
  real y[N];
  real b[N];
}
parameters{
  real alpha_mu;                    // population-level alpha
  real theta_mu;   
  real<lower=0> alpha_sigma; // population-level effect
  real<lower=0> theta_sigma; // population-level effect
  vector[I] alpha;                       
  vector[I] theta_raw;
  real<lower=0> sigma[I];
  real<lower=0> sigma_sigma;
  real<lower=0> sigma_mu;
}

transformed parameters{
    vector[I] theta;
    
    for(i in 1:I){
      theta[i] = theta_mu + theta_raw[i]*theta_sigma;
    }
}

model{
  alpha_mu ~ normal(0.5,1);
  alpha_sigma ~ student_t(4,0,0.5);// prior on sigmaAlpha
  theta_mu ~ normal(0,1);                 // prior on muTheta
  theta_sigma ~student_t(4,0,0.5);
  sigma_mu ~ normal(0,0.5);
  sigma_sigma ~ student_t(4,0,0.5);
  
  for(i in 1:I){
    sigma[i] ~ normal(sigma_mu, sigma_sigma);
    alpha[i] ~ normal(alpha_mu, alpha_sigma);
    theta_raw[i] ~ std_normal();
  }
  
  for(n in 1:N){
    y[n] ~ normal(alpha[subj[n]] + b[n]*theta[subj[n]], sigma[subj[n]]);
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
    log_lik[n] = normal_lpdf(y[n] | alpha[subj[n]] + (b[n]*theta[subj[n]]), sigma[subj[n]]);
    y_pred[n] =  normal_rng(alpha[subj[n]] + b[n]*theta[subj[n]], sigma[subj[n]]);
  }
  
}
