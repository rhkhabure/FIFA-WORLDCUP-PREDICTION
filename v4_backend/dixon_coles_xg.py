import numpy as np
import pandas as pd
from scipy.optimize import minimize

def continuous_dixon_coles_nll(params, df, n_teams, decay_rate=0.0018):
    """
    Computes the Negative Log-Likelihood of an xG-based Dixon-Coles model.
    Uses continuous xG values for attack/defense optimization and discrete 
    goals for the low-score interdependence parameter (rho).
    """
    # 1. Unpack parameters
    alphas = params[0:n_teams]
    betas = params[n_teams:2*n_teams]
    gamma = params[2*n_teams]  # Home Advantage
    rho = params[2*n_teams + 1] # Low-score joint probability correction
    
    # 2. Extract match vectors
    h_idx = df['home_idx'].values
    a_idx = df['away_idx'].values
    obs_xg_h = df['obs_xg_h'].values
    obs_xg_a = df['obs_xg_a'].values
    act_g_h = df['act_g_h'].values
    act_g_a = df['act_g_a'].values
    days_ago = df['days_ago'].values
    
    # 3. Calculate lambda and mu (clamped to prevent log(0) and overflow)
    lambdas = np.clip(alphas[h_idx] * betas[a_idx] * gamma, 1e-5, 15.0)
    mus = np.clip(alphas[a_idx] * betas[h_idx], 1e-5, 15.0)
    
    # 4. Continuous Poisson log-likelihood terms (Gamma function drops constant)
    ll_home = obs_xg_h * np.log(lambdas) - lambdas
    ll_away = obs_xg_a * np.log(mus) - mus
    
    # 5. Dixon-Coles Rho Adjustment based on integer outcomes
    rho_terms = np.ones(len(df))
    m00 = (act_g_h == 0) & (act_g_a == 0)
    m10 = (act_g_h == 1) & (act_g_a == 0)
    m01 = (act_g_h == 0) & (act_g_a == 1)
    m11 = (act_g_h == 1) & (act_g_a == 1)
    
    rho_terms[m00] = 1.0 - lambdas[m00] * mus[m00] * rho
    rho_terms[m10] = 1.0 + mus[m10] * rho
    rho_terms[m01] = 1.0 + lambdas[m01] * rho
    rho_terms[m11] = 1.0 - rho
    
    # Clamp rho corrections to keep them mathematically valid
    rho_terms = np.clip(rho_terms, 1e-5, 10.0)
    ll_rho = np.log(rho_terms)
    
    # 6. Apply Time Decay weighting (w_m = e^(-xi * t))
    # decay_rate = 0.0018 corresponds to roughly a 1-year half-life
    weights = np.exp(-decay_rate * days_ago)
    
    # Return Negative LL because standard solvers minimize
    return -np.sum((ll_home + ll_away + ll_rho) * weights)

def train_dixon_coles_prior(df_ledger, teams_list):
    """
    Executes the constrained SLSQP optimization sequence to fit parameters.
    """
    n_teams = len(teams_list)
    
    # Initial Parameter Guesses: Alphas=1.0, Betas=1.0, Gamma=1.15, Rho=0.0
    init_params = np.concatenate([np.ones(n_teams), np.ones(n_teams), [1.15, 0.0]])
    
    # Define rigid boundaries to prevent mathematical explosions during iterations
    alpha_bounds = [(0.1, 3.5)] * n_teams
    beta_bounds = [(0.1, 3.5)] * n_teams
    gamma_bounds = [(0.6, 2.0)]
    rho_bounds = [(-0.25, 0.25)]  # Keeps joint probability matrix positive definite
    bounds = alpha_bounds + beta_bounds + gamma_bounds + rho_bounds
    
    # The Critical Normalization Constraint: mean(alphas) == 1.0
    # Enforced via equality constraint mapping: mean(p) - 1.0 = 0
    def constraint_mean_alpha(params):
        return np.mean(params[0:n_teams]) - 1.0
        
    constraints = {'type': 'eq', 'fun': constraint_mean_alpha}
    
    # Execute SLSQP Optimization Loop
    print("Initiating Maximum Likelihood Estimation via SLSQP...")
    res = minimize(
        fun=continuous_dixon_coles_nll,
        x0=init_params,
        args=(df_ledger, n_teams),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 150, 'disp': True}
    )
    
    if not res.success:
        raise ValueError(f"Solver failed to converge: {res.message}")
        
    # Restructure outputs into clean dictionary objects
    optimized_params = res.x
    prior_dict = {}
    for idx, team in enumerate(teams_list):
        prior_dict[team] = {
            'alpha': optimized_params[idx],
            'beta': optimized_params[idx + n_teams]
        }
        
    meta_params = {
        'gamma_home_advantage': optimized_params[2 * n_teams],
        'rho_draw_correction': optimized_params[2 * n_teams + 1]
    }
    
    return prior_dict, meta_params