import numpy as np
from scipy.stats import poisson

def generate_match_probabilities(alpha_h_adj, beta_a_adj, alpha_a_adj, beta_h_adj, gamma, rho, max_goals=9):
    """
    Constructs the bivariate Poisson distribution matrix using adjusted parameters
    and extracts the final 1X2 market probabilities and fair decimal odds.
    """
    # 1. Calculate the final adjusted expected goals (lambda and mu) for this match
    lambda_h = alpha_h_adj * beta_a_adj * gamma
    mu_a = alpha_a_adj * beta_h_adj
    
    # 2. Generate independent Poisson probabilities for all scoreline combinations
    # Creating vectors of probabilities from 0 to max_goals-1
    goals_range = np.arange(max_goals)
    prob_home = poisson.pmf(goals_range, lambda_h)
    prob_away = poisson.pmf(goals_range, mu_a)
    
    # Airtight Tail Capture: Force index 8 to represent "8 or more goals"
    prob_home[-1] = poisson.sf(max_goals - 2, lambda_h)
    prob_away[-1] = poisson.sf(max_goals - 2, mu_a)
    
    # Outer product creates the base independent matrix: matrix[h, a] = P(H=h) * P(A=a)
    score_matrix = np.outer(prob_home, prob_away)
    
    # 3. Apply the Dixon-Coles Rho Adjustment to low-scoring cells
    # Modifying cells: (0,0), (1,0), (0,1), (1,1)
    if max_goals > 1:
        score_matrix[0, 0] *= (1.0 - lambda_h * mu_a * rho)
        score_matrix[1, 0] *= (1.0 + mu_a * rho)
        score_matrix[0, 1] *= (1.0 + lambda_h * rho)
        score_matrix[1, 1] *= (1.0 - rho)
        
    # Ensure any extreme rho adjustment didn't push values below zero, then re-normalize
    score_matrix = np.clip(score_matrix, 0.0, 1.0)
    score_matrix /= np.sum(score_matrix)
    
    # 4. Aggregate scorelines into standard 1X2 Main Market outcomes
    # Home Win: Lower triangle of the matrix (rows > columns)
    prob_home_win = np.sum(np.tril(score_matrix, -1))
    
    # Draw: Main diagonal of the matrix (rows == columns)
    prob_draw = np.sum(np.diag(score_matrix))
    
    # Away Win: Upper triangle of the matrix (rows < columns)
    prob_away_win = np.sum(np.triu(score_matrix, 1))
    
    # 5. Convert probabilities to raw Fair Decimal Odds (1 / probability)
    # Using a minor clamp to prevent division by zero for highly skewed matchups
    odds_home = 1.0 / max(prob_home_win, 1e-4)
    odds_draw = 1.0 / max(prob_draw, 1e-4)
    odds_away = 1.0 / max(prob_away_win, 1e-4)
    
    probabilities = {
        'home_win': prob_home_win,
        'draw': prob_draw,
        'away_win': prob_away_win
    }
    
    fair_odds = {
        '1': round(odds_home, 2),
        'X': round(odds_draw, 2),
        '2': round(odds_away, 2)
    }
    
    return probabilities, fair_odds, score_matrix