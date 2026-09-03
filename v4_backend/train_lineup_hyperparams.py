import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
from pathlib import Path

# Ensure local imports work regardless of execution directory
sys.path.append(str(Path(__file__).parent))
from bivariate_poisson import generate_match_probabilities

def lineup_brier_loss(hyperparams, df_train, prior_dict, static_gamma, rho):
    """
    Objective function to minimize the Brier Score of pre-match 1X2 market odds
    by tuning the lineup dampening scalars lambda_a and lambda_d.
    """
    lambda_a, lambda_d = hyperparams
    total_brier_score = 0.0
    n_matches = len(df_train)
    
    # 1. Unpack arrays from training ledger
    home_teams = df_train['home_team_name'].values
    away_teams = df_train['away_team_name'].values
    delta_att_h = df_train['delta_att_h'].values
    delta_def_h = df_train['delta_def_h'].values
    delta_att_a = df_train['delta_att_a'].values
    delta_def_a = df_train['delta_def_a'].values
    
    # One-hot encoded actual match outcome vectors [Home Win, Draw, Away Win]
    actual_outcomes = df_train[['act_win_h', 'act_draw', 'act_win_a']].values 
    
    for i in range(n_matches):
        h_team = home_teams[i]
        a_team = away_teams[i]
        
        # Pull core prior baselines from Module 1 Dictionary
        # Fallback to 1.0 if a team isn't found (e.g. newly promoted)
        alpha_h = prior_dict.get(h_team, {}).get('alpha', 1.0)
        beta_h = prior_dict.get(h_team, {}).get('beta', 1.0)
        alpha_a = prior_dict.get(a_team, {}).get('alpha', 1.0)
        beta_a = prior_dict.get(a_team, {}).get('beta', 1.0)
        
        # 2. Apply Lineup Likelihood Adjustments using current iteration scalars
        alpha_h_adj = alpha_h * (delta_att_h[i] ** lambda_a)
        beta_h_adj = beta_h * ((1.0 / max(delta_def_h[i], 0.1)) ** lambda_d)
        alpha_a_adj = alpha_a * (delta_att_a[i] ** lambda_a)
        beta_a_adj = beta_a * ((1.0 / max(delta_def_a[i], 0.1)) ** lambda_d)
        
        # 3. Generate 1X2 Probabilities using your structural matrix engine
        probs, _, _ = generate_match_probabilities(
            alpha_h_adj, beta_a_adj, alpha_a_adj, beta_h_adj, 
            static_gamma, rho
        )
        
        pred_vector = np.array([probs['home_win'], probs['draw'], probs['away_win']])
        act_vector = actual_outcomes[i]
        
        # 4. Accumulate Brier Loss: Sum of squared errors between probabilities and outcomes
        total_brier_score += np.sum((pred_vector - act_vector) ** 2)
        
    return total_brier_score / n_matches

def calibrate_lineup_scalars(df_train, prior_dict, static_gamma, rho):
    """Executes L-BFGS-B optimization to define optimal lambda parameters."""
    # Initial guesses: starting at standard moderate dampening baseline (0.50)
    init_guesses = [0.50, 0.50] 
    
    # Bounded to prevent inverse logic flips (must stay positive and realistic)
    bounds = [(0.05, 1.50), (0.05, 1.50)] 
    
    print("Initiating Secondary Optimization Loop for Lineup Calibrations...")
    res = minimize(
        fun=lineup_brier_loss,
        x0=init_guesses,
        args=(df_train, prior_dict, static_gamma, rho),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    if not res.success:
        print("⚠️ Calibrator failed to hit ideal global convergence. Using baseline assumptions.")
        return {'lambda_a': 0.65, 'lambda_d': 0.55}
        
    return {'lambda_a': res.x[0], 'lambda_d': res.x[1]}
