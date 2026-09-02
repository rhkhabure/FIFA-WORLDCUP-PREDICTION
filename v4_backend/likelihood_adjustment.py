import numpy as np

def calculate_roster_delta(matchday_lineup, ideal_roster):
    """
    Computes the roster strength ratio between the announced Matchday XI
    and the historical rolling Ideal XI.
    
    matchday_lineup: Dict containing lists of Sofascore ratings grouped by position
                    e.g., {'attack_mid': [7.2, 6.9, 7.5, 7.1], 'def_gk': [6.8, 7.0, 7.3, 6.5, 7.1]}
    ideal_roster: Dict containing the ideal highest-rated XI baseline for the squad
                 e.g., {'ideal_att_mid_score': 7.42, 'ideal_def_gk_score': 7.15}
    """
    # 1. Protect against missing player data by enforcing a baseline replacement floor
    # If a player has no score (e.g., debutant), we assign a default professional rating floor
    att_ratings = [rating if rating > 0 else 6.5 for rating in matchday_lineup.get('attack_mid', [])]
    def_ratings = [rating if rating > 0 else 6.4 for rating in matchday_lineup.get('def_gk', [])]
    
    # 2. Compute current matchday averages
    s_prime_att = np.mean(att_ratings) if att_ratings else 6.8
    s_prime_def = np.mean(def_ratings) if def_ratings else 6.7
    
    # 3. Calculate Deltas (Ratios against ideal roster strength)
    # A value of 0.96 implies the squad group is operating at 96% of its maximum threat capacity
    delta_att = s_prime_att / ideal_roster['ideal_att_mid_score']
    delta_def = s_prime_def / ideal_roster['ideal_def_gk_score']
    
    # Safeguard: Clamp deltas so data errors or hyper-anomalies can't warp parameters past reality
    delta_att = np.clip(delta_att, 0.80, 1.20)
    delta_def = np.clip(delta_def, 0.80, 1.20)
    
    return delta_att, delta_def

def apply_bayesian_lineup_adjustment(alpha_prior, beta_prior, delta_att, delta_def, hyperparams):
    """
    Applies the dampened roster delta updates to the Dixon-Coles parameters.
    
    hyperparams: Dict containing tuned scalars {'lambda_a': 0.65, 'lambda_d': 0.55} 
                 learned from historical injury datasets.
    """
    lambda_a = hyperparams['lambda_a']
    lambda_d = hyperparams['lambda_d']
    
    # Update Attack Strength: A lower delta reduces alpha (clamped to protect bounds)
    alpha_adjusted = alpha_prior * (delta_att ** lambda_a)
    alpha_adjusted = np.clip(alpha_adjusted, 0.1, 3.5)
    
    # Update Defense Strength: A lower delta increases beta (inflating conceded xG)
    beta_adjusted = beta_prior * ((1.0 / delta_def) ** lambda_d)
    beta_adjusted = np.clip(beta_adjusted, 0.1, 3.5)
    
    return alpha_adjusted, beta_adjusted
