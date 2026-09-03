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
    Includes airtight interval boundary checks for the infinite tail remainder matrix.
    """
    # 1. Calculate remaining time factor
    time_remaining = max(0, 90 - current_minute)
    time_fraction = time_remaining / 90.0
    
    if time_remaining == 0:
        return determine_final_state(home_score, away_score)

    # 2. Calculate pre-match baseline expectations up to this minute
    expected_xg_h_until_now = (alpha_h_adj * beta_a_adj * gamma) * (current_minute / 90.0)
    expected_xg_a_until_now = (alpha_a_adj * beta_h_adj) * (current_minute / 90.0)
    
    # 3. Game-State Scaling Factors
    omega_h = 1.0
    omega_a = 1.0
    if home_score > away_score:
        omega_h = 0.75
        omega_a = 1.35
    elif away_score > home_score:
        omega_h = 1.40
        omega_a = 0.70

    # 4. Compute Live Performance Modifiers
    perf_mod_h = np.clip(live_xg_h / max(expected_xg_h_until_now, 0.1), 0.5, 2.0)
    perf_mod_a = np.clip(live_xg_a / max(expected_xg_a_until_now, 0.1), 0.5, 2.0)

    # 5. Project parameters for the REMAINING minutes
    lambda_h_rem = (alpha_h_adj * beta_a_adj * gamma) * time_fraction * omega_h * perf_mod_h
    mu_a_rem = (alpha_a_adj * beta_h_adj) * time_fraction * omega_a * perf_mod_a
    
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
    remainder_matrix /= np.sum(remainder_matrix)
    
    # 7. Map remainder matrix to the current live game score (Air-Tight Interval Check)
    prob_home_win = 0.0
    prob_draw = 0.0
    prob_away_win = 0.0
    
    for h_rem in range(max_goals):
        for a_rem in range(max_goals):
            cell_prob = remainder_matrix[h_rem, a_rem]
            
            # Detect if we are handling the infinite tail indices
            is_home_tail = (h_rem == max_goals - 1)
            is_away_tail = (a_rem == max_goals - 1)
            
            final_h = home_score + h_rem
            final_a = away_score + a_rem
            
            if not is_home_tail and not is_away_tail:
                # Standard exact scoreline cell
                if final_h > final_a:
                    prob_home_win += cell_prob
                elif final_h == final_a:
                    prob_draw += cell_prob
                else:
                    prob_away_win += cell_prob
            else:
                # Tail cell handling: Evaluate bounds conservatively 
                # to prevent misallocating draw mass to wins
                if final_h > final_a and not is_away_tail:
                    prob_home_win += cell_prob
                elif final_a > final_h and not is_home_tail:
                    prob_away_win += cell_prob
                else:
                    # In overlapping tail events (e.g., both teams explosive late),
                    # split the tail remainder weight based on relative live strengths
                    ratio_h = lambda_h_rem / (lambda_h_rem + mu_a_rem)
                    prob_home_win += cell_prob * ratio_h
                    prob_away_win += cell_prob * (1.0 - ratio_h)
                
    return {
        'live_probabilities': {'1': round(prob_home_win, 4), 'X': round(prob_draw, 4), '2': round(prob_away_win, 4)},
        'live_fair_odds': {'1': round(1/max(prob_home_win, 1e-4), 2), 'X': round(1/max(prob_draw, 1e-4), 2), '2': round(1/max(prob_away_win, 1e-4), 2)}
    }