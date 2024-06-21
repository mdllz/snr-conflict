functions{
real racing_diffusion_lpdf(matrix Dat, real ndt, real bound, real int1, real int2, real eff,  real sig1, real sig2){
    real log_pdf;
    real cdf;
    real log_cdf_comp;
    vector[rows(Dat)] joint_density;
    real sum_joint_density;
      for(j in 1:rows(Dat)){
        if(Dat[j,1]-ndt > 0){ // if rt-ndt is positive
          log_cdf_comp = 0;
          if(Dat[j,2] == 1){ // correct response: conflict effect
              log_pdf = log(bound) - log(sig1 * ((2*pi()*(Dat[j,1]-ndt)^3)^0.5)) + (-(bound - ( (int1+(eff*Dat[j,3]))   *(Dat[j,1]-ndt)))^2 / (2*(sig1^2)*(Dat[j,1]-ndt)));
              cdf = Phi( (-bound + (   int2  *(Dat[j,1]-ndt))) / (sig2*((Dat[j,1]-ndt)^0.5))) + (exp((2*   int2   *bound) / sig2^2) * Phi((-bound- (int2*(Dat[j,1]-ndt)))/(sig2*((Dat[j,1]-ndt)^0.5))));
              log_cdf_comp = log(1 - cdf) + log_cdf_comp;
            } else { // incorrect response: no conflict eff
log_pdf = log(bound) - log(sig2 * ((2*pi()*(Dat[j,1]-ndt)^3)^0.5)) + (-(bound - (int2*(Dat[j,1]-ndt)))^2 / (2*(sig2^2)*(Dat[j,1]-ndt)));
cdf = Phi((-bound+((int1+(eff*Dat[j,3]))*(Dat[j,1]-ndt))) / (sig1*((Dat[j,1]-ndt)^0.5))) + (exp((2*(int1+(eff*Dat[j,3]))*bound)/sig1^2)*Phi((-bound-((int1+(eff*Dat[j,3]))*(Dat[j,1]-ndt)))/(sig1*((Dat[j,1]-ndt)^0.5))));
log_cdf_comp = log(1 - cdf) + log_cdf_comp;
          }
          joint_density[j] = log_pdf + log_cdf_comp;
        } else { // if rt-ndt is not positive
          joint_density[j] = log(1e-10);
        }
      }
      sum_joint_density = sum(joint_density);
      return sum_joint_density;      
    }
    
real wald_rng(real bound, real drift, real sig) {
  
  real mu;
  real lambda;
  real nu;
  real y;
  real x;
  real z;
  
   mu = bound/drift;
   lambda = (bound/sig)^2;
  
  
    nu = normal_rng(0, 1);
    y = nu^2;
    x = mu + (((mu^2)*y)/(2*lambda)) - ((mu/(2*lambda))*sqrt((4*mu*lambda*y)+((mu^2) * (y^2))));
    z = uniform_rng(0, 1);
    
    if(z<= (mu/(mu+x))){
     return x;
    } else {
     return (mu^2)/x;
    }
  
  }
  
  matrix rdm_rng(matrix Dat, real int1, real int2, real eff, real bound, real sig1, real sig2, real psi){

  vector[2] y;
  matrix[rows(Dat),3] pred; // rt, accuracy, condition
  
  for(j in 1:rows(Dat)){
    y[1] = wald_rng(bound, (int1+(eff*Dat[j,3])), sig1);
    y[2] = wald_rng(bound, int2, sig2);
    pred[j,1] = min(y) + psi; //min(y); //+ psi; // rt
    pred[j,2] = sort_indices_asc(y)[1]; // decision
    pred[j,3] = Dat[j,3];
   }
  return pred;
  } 
}


data{
 int<lower=0> N; // nr of observations
 int J; // max number of trials
 int I; // nr of subjects
 int K; // nr of racers
 matrix[N,3] Y;
 int trialsum[I];   // participant id indicator
 int idx_start[I];
}

transformed data{
   matrix[J,3] filler_mat;
   
   // create task data filler matrix made up of dummy variables
   for(j in 1:J){
    filler_mat[j] = [-9999,-9999,-9999]; 
  }
}

parameters{
 real<lower = 0> beta[I]; // threshold
 real<lower = 0> beta_mu;
 real<lower = 0> beta_sigma;
 vector<lower = 0.1>[I] psi; // ndt 
 real<lower = 0> psi_mu;
 real<lower = 0> psi_sigma;
 real alpha_1_mu;
 real<lower = 0> alpha_1_sigma;
 real<lower = 0> alpha_2; // drift rate intercept mismatching accumulator
 real theta[I]; 
 real alpha_1[I];
 real theta_mu;
 real<lower = 0> theta_sigma;
}

model{
// hyperpriors
// intercepts
 alpha_2 ~ normal(0.6,2);
 alpha_1_mu ~ normal(3.7, 1);
 alpha_1_sigma ~ student_t(4,0,0.5);
 // conflict effect
 theta_mu ~ normal(0,1);
 theta_sigma ~ student_t(4,0,0.5);
 // ndt
 psi_mu ~ normal(.3, .2)T[0.1,0.5];
 psi_sigma ~ student_t(4,0,0.5);
 // threshold
 beta_mu ~ normal(0,2);
 beta_sigma ~ student_t(4,0,1);
 
 // individual effects
  for(i in 1:I){
    alpha_1[i] ~ normal(alpha_1_mu, alpha_1_sigma);
    theta[i] ~ normal(theta_mu, theta_sigma);
    psi[i] ~ normal(psi_mu, psi_sigma);
    beta[i] ~ normal(beta_mu, beta_sigma);

// likelihood
      block(Y, idx_start[i], 1, trialsum[i], 3) ~ racing_diffusion(psi[i], beta[i], alpha_1[i], alpha_2, theta[i], 1, 1);

  }

}

generated quantities{

   real drift_1_inc[I];
   real drift_1_c[I];
   matrix [J,3] Y_pred[I];
      
  for(i in 1:I){ 
      drift_1_inc[i] = alpha_1[i] - 0.5*theta[i];
      drift_1_c[i] = alpha_1[i] + 0.5*theta[i];
  
   if(trialsum[i] == J){
      
       Y_pred[i] = rdm_rng(block(Y, idx_start[i], 1, trialsum[i], 3), alpha_1[i], alpha_2, theta[i], beta[i], 1, 1, psi[i]);

      }
    if(trialsum[i] < J){
    // row-bind predictions and subset of filler matrix such that each individual prediction matrix has J rows
      Y_pred[i] = append_row(rdm_rng(block(Y, idx_start[i], 1, trialsum[i], 3), alpha_1[i], alpha_2, theta[i], beta[i], 1, 1, psi[i]),
                                     block(filler_mat, 1, 1, J-trialsum[i], 3));
    }
  }
  
}
