import numpy as np
from scipy.stats import poisson

def determine_final_state(home_score, away_score):
    if home_score > away_score:
        return {'live_probabilities': {'1': 1.0, 'X': 0.0, '2': 0.0}, 'live_fair_odds': {'1': 1.0, 'X': 999.0, '2': 999.0}}
    elif home_score == away_score:
        return {'live_probabilities': {'1': 0.0, 'X': 1.0, '2': 0.0}, 'live_fair_odds': {'1': 999.0, 'X': 1.0, '2': 999.0}}
    else:
        return {'live_probabilities': {'1': 0.0, 'X': 0.0, '2': 1.0}, 'live_fair_odds': {'1': 999.0, 'X': 999.0, '2': 1.0}}

def generate_live_in_play_odds(current_minute, home_score, away_score, 
                               live_xg_h, live_xg_a, 
                               alpha_h_adj, beta_a_adj, alpha_a_adj, beta_h_adj, 
                               gamma, rho, max_goals=9):
    """
    Computes live in-play 1X2 probabilities for the remainder of the match,
    dynamically adjusting the Poisson parameters for time remaining and game-state bias.
    """
    # 1. Calculate remaining time factor
    time_remaining = max(0, 90 - current_minute)
    time_fraction = time_remaining / 90.0
    
    if time_remaining == 0:
        # Match over: Output definitive 1X2 based on current score
        return determine_final_state(home_score, away_score)

    # 2. Calculate pre-match baseline expectations up to this minute
    expected_xg_h_until_now = (alpha_h_adj * beta_a_adj * gamma) * (current_minute / 90.0)
    expected_xg_a_until_now = (alpha_a_adj * beta_h_adj) * (current_minute / 90.0)
    
    # 3. Game-State Scaling Factors (Tuned from historical trailing/leading metrics)
    # If a team is trailing, their tactical urgency increases baseline xG generation
    omega_h = 1.0
    omega_a = 1.0
    
    if home_score > away_score:      # Home team is leading
        omega_h = 0.75               # Low block reduction
        omega_a = 1.35               # Trailing urgency boost
    elif away_score > home_score:    # Away team is leading
        omega_h = 1.40               # Trailing urgency boost
        omega_a = 0.70               # Low block reduction

    # 4. Compute Live Performance Modifiers (Clamped to prevent severe single-match noise)
    # Compares actual live performance against the game-state scaled expectation
    perf_mod_h = np.clip(live_xg_h / max(expected_xg_h_until_now, 0.1), 0.5, 2.0)
    perf_mod_a = np.clip(live_xg_a / max(expected_xg_a_until_now, 0.1), 0.5, 2.0)

    # 5. Project parameters for the REMAINING minutes
    lambda_h_rem = (alpha_h_adj * beta_a_adj * gamma) * time_fraction * omega_h * perf_mod_h
    mu_a_rem = (alpha_a_adj * beta_h_adj) * time_fraction * omega_a * perf_mod_a
    
    # Ensure they don't drop to complete zero
    lambda_h_rem = max(lambda_h_rem, 1e-4)
    mu_a_rem = max(mu_a_rem, 1e-4)

    # 6. Generate the matrix for the REMAINDER goals scored from this point forward
    goals_range = np.arange(max_goals)
    prob_h_rem = poisson.pmf(goals_range, lambda_h_rem)
    prob_a_rem = poisson.pmf(goals_range, mu_a_rem)
    
    # Apply our airtight tail-end collection fix
    prob_h_rem[-1] = poisson.sf(max_goals - 2, lambda_h_rem)
    prob_a_rem[-1] = poisson.sf(max_goals - 2, mu_a_rem)
    
    remainder_matrix = np.outer(prob_h_rem, prob_a_rem)
    
    # Re-normalize remainder matrix
    remainder_matrix /= np.sum(remainder_matrix)
    
    # 7. Map remainder matrix to the current live game score
    # We iterate over the remaining goals matrix and project the final scorelines
    prob_home_win = 0.0
    prob_draw = 0.0
    prob_away_win = 0.0
    
    for h_rem in range(max_goals):
        for a_rem in range(max_goals):
            final_h = home_score + h_rem
            final_a = away_score + a_rem
            cell_prob = remainder_matrix[h_rem, a_rem]
            
            if final_h > final_a:
                prob_home_win += cell_prob
            elif final_h == final_a:
                prob_draw += cell_prob
            else:
                prob_away_win += cell_prob
                
    return {
        'live_probabilities': {'1': round(prob_home_win, 4), 'X': round(prob_draw, 4), '2': round(prob_away_win, 4)},
        'live_fair_odds': {'1': round(1/max(prob_home_win, 1e-4), 2), 'X': round(1/max(prob_draw, 1e-4), 2), '2': round(1/max(prob_away_win, 1e-4), 2)}
    }