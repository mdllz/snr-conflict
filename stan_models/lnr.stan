functions{ // data matrix, NDT, intercept matching drift, intercept mismatching drift, effect on matching drift, sigma matching, sigma mismatching
  real lognormal_race_lpdf(matrix Dat, real ndt, real int1, real int2, real eff, real sig1, real sig2){
    real log_pdf;
    real log_cdf_comp;
    vector[rows(Dat)] joint_density;
    real sum_joint_density;
     for(j in 1:rows(Dat)){
       // if RT > NDT
        if(Dat[j,1]-ndt > 0){ 
          if(Dat[j,2] == 1){ // correct response: conflict effect
           log_pdf = lognormal_lpdf(Dat[j,1] - ndt | (int1+(eff*Dat[j,3])), sig1);
           log_cdf_comp = lognormal_lccdf(Dat[j,1]-ndt | int2, sig2);
           joint_density[j] = log_pdf + log_cdf_comp;
           } else { // incorrect response: no conflict eff
           log_pdf = lognormal_lpdf(Dat[j,1] - ndt | int2, sig2);
           log_cdf_comp = lognormal_lccdf(Dat[j,1]-ndt | (int1+(eff*Dat[j,3])), sig1);
          joint_density[j] = log_pdf + log_cdf_comp;
         } 
       // if RT < NDT 
        } else { 
         joint_density[j] = log(1e-10);
       }
      if(joint_density[j] < log(1e-10)){ // extra protection against very small values
       joint_density[j] = log(1e-10);
     }
   }
  sum_joint_density = sum(joint_density);
 return sum_joint_density;
}

matrix lognormal_race_rng(matrix Dat, real int1, real int2, real eff, real sigma1, real sigma2, real psi){

  vector[2] y;
  matrix[rows(Dat),3] pred; // rt, accuracy, condition
  
  for(j in 1:rows(Dat)){
    y[1] = lognormal_rng((int1+(eff*Dat[j,3])), sigma1);
    y[2] = lognormal_rng(int2, sigma2);
    pred[j,1] = min(y) + psi; //min(y); //+ psi; // rt
    pred[j,2] = sort_indices_asc(y)[1]; // decision
    pred[j,3] = Dat[j,3];
  }
  return pred;
  }
}

data {
  int<lower=0> N; // nr of observations
  int J; // max number of trials
  int I; // nr of participants
  matrix[N,3] Y;
  int trialsum[I];   // participant id indicator
  int idx_start[I];
}

transformed data{
   matrix[J,3] filler_mat;
   
   // create task data filler matrix to deal with unequal trial numbers
   for(j in 1:J){
    filler_mat[j] = [-9999,-9999,-9999]; 
  }
}

parameters {
  real<lower = 0> sigma_1[I]; // noise parameters
  real<lower = 0> sigma_1_mu;
  real<lower = 0> sigma_2;
  real<lower = 0> sigma_1_sigma;
  vector<lower = 0.1>[I] psi; // ndt 
  real<lower = 0.1> psi_mu;
  real<lower = 0> psi_sigma;
  real alpha_1[I];
  real alpha_1_mu; 
  real<lower = 0> alpha_1_sigma;
  real alpha_2; 
  real theta[I];
  real theta_mu;
  real<lower = 0> theta_sigma;
}

model {
   // hyperpriors
   // intercepts
   alpha_2 ~ normal(-0.5,1);
   alpha_1_mu ~ normal(-2,1);
   alpha_1_sigma ~ student_t(4,0,0.5);
   // conflict effect
   theta_mu ~ normal(0,1);
   theta_sigma ~ student_t(4,0,0.5);
   // ndt
   psi_mu ~ normal(.3, .2)T[0.1,0.5];
   psi_sigma ~ student_t(4,0,0.5);
   // noise term
   sigma_1_mu ~ normal(0,0.5);//student_t(4,0,0.5);
   sigma_2 ~ student_t(4,0,0.5);
   sigma_1_sigma ~ student_t(4,0,0.5);

  // individual effects
  for(i in 1:I){
   sigma_1[i] ~ normal(sigma_1_mu, sigma_1_sigma);
   alpha_1[i] ~ normal(alpha_1_mu, alpha_1_sigma);
   theta[i] ~ normal(theta_mu, theta_sigma);
   psi[i] ~ normal(psi_mu, psi_sigma);
    // slice data matrix into individual participant matrices
   block(Y, idx_start[i], 1, trialsum[i], 3) ~ lognormal_race(psi[i], alpha_1[i], alpha_2, theta[i], sigma_1[i], sigma_2);
  }
}

generated quantities{
  matrix [J,3] Y_pred[I];
  real mu_ic[I];
  real mu_c[I];
   
  for(i in 1:I){
    mu_c[i] = alpha_1[i] - (0.5*theta[i]);
    mu_ic[i] = alpha_1[i] + (0.5*theta[i]);
  }
      
  for(i in 1:I){ 
   if(trialsum[i] == J){
       Y_pred[i] = lognormal_race_rng(block(Y, idx_start[i], 1, trialsum[i], 3), alpha_1[i], alpha_2, theta[i], sigma_1[i], sigma_2, psi[i]);
      }
   if(trialsum[i] < J){
    // row-bind predictions and subset of filler matrix such that each individual prediction matrix has J rows
      Y_pred[i] = append_row(lognormal_race_rng(block(Y, idx_start[i], 1, trialsum[i], 3), alpha_1[i], alpha_2, theta[i], sigma_1[i], sigma_2, psi[i]),
      block(filler_mat, 1, 1, J-trialsum[i], 3));
    }
  }
}

