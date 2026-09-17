data{
    int<lower = 0> N; // number of observations
    int<lower = 0> J; // number of individuals
    array[N] int<lower=1, upper=J> I; // individual IDs
    vector[N] T; // temperature
    vector[N] R; // log mass-specific RMR
    vector[N] M; // mass or SVL
    vector[J] S; // site indicator
    vector[J] Y; // year indicator
}

parameters{
    // population-level fixed effects
    real a; // baseline intercept
    real <lower=0> b; // baseline slope truncated at 0
    real a_S; // effect of site on intercept
    real a_Y; // effect of year on intercept
    real a_M; // effect of body mass on intercept
    real b_S; // effect of site on slope
    real b_Y; // effect of year on slope
    real b_M; // effect of body mass on slope

    // individual-level random effects
    real <lower=0> Sigma; // residual error
    vector<lower=0>[2] Tau; // SD of individual intercepts and slopes
    cholesky_factor_corr[2] L_Omega; // Cholesky factor of correlation matrix
    matrix[2, J] Z; // standardized individual effects
}

transformed parameters {
   // define matrices to hold individual parameters and predicted means
    matrix[2, J] Theta; // matrix to temporarily hold alpha_j and beta_j
    matrix[2, J] Mu; // matrix of predicted mean intercepts and slopes

    // compute indiividual parameters
    for(j in 1:J) {
        // intercept of lizard j
        Mu[1, j] = a + (a_S * S[j]) + (a_Y * Y[j]) + (a_M * M[j]);
        // slope of lizard j
        Mu[2, j] = b + (b_S * S[j]) + (b_Y * Y[j]) + (b_M * M[j]);
    }

    // implement non-centered parameterization
    Theta = Mu + diag_pre_multiply(Tau, L_Omega) * Z;
}

model{
    // priors
    a ~ normal(0, 0.5); // population-level intercept
    b ~ normal(0.1, 0.5); // population-level slope (informative prior)
    a_S ~ normal(0, 0.5); // effect of site on intercept
    a_Y ~ normal(0, 0.5); // effect of year on intercept
    a_M ~ normal(0, 0.5); // effect of body mass on intercept
    b_S ~ normal(0, 0.5); // effect of site on slope
    b_Y ~ normal(0, 0.5); // effect of year on slope
    b_M ~ normal(0, 0.5); // effect of body mass on slope
    Sigma ~ normal(0, 0.5); // residual error
    Tau ~ normal(0, 0.5); // SD of individual intercepts and slopes
    L_Omega ~ lkj_corr_cholesky(2); // Cholesky factor of correlation matrix
    to_vector(Z) ~ normal(0, 1); // standardized individual effects

    // likelihood
    for(n in 1:N) {
        R[n] ~ normal(Theta[1, I[n]] + Theta[2, I[n]] * T[n], Sigma);
    }
}

generated quantities {
    // correlation matrix of intercept and slope
    corr_matrix[2] Omega = multiply_lower_tri_self_transpose(L_Omega);
    real Rho = Omega[1, 2]; // correlation between intercept and slope

    // estimated individual interecepts and slopes
    vector[J] a_I = Theta[1, ]'; // individual intercepts
    vector[J] b_I = Theta[2, ]'; // individual slopes

    // posterior predictions and log-likelihood for model comparison
    vector[N] R_pred; // posterior predictions
    vector[N] log_lik; // log-likelihood for each observation
    for(n in 1:N) {
        R_pred[n] = normal_rng(Theta[1, I[n]] + Theta[2, I[n]] * T[n], Sigma);
        log_lik[n] = normal_lpdf(R[n] | Theta[1, I[n]] + Theta[2, I[n]] * T[n], Sigma);
    }
}
