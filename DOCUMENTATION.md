# 📓 V4 Mathematical Documentation

This document explains the core statistical and mathematical framework powering the V4 Bayesian Tri-State engine.

---

## 1. The Continuous xG Maximum Likelihood Estimator (Prior)
Unlike traditional Dixon-Coles models that optimize against discrete goals ($0, 1, 2$), our model minimizes the Negative Log-Likelihood (-LL) of **Continuous Expected Goals (xG)**.

Because traditional goal models use $x!$ (factorial) in the denominator, they crash when fed continuous floats (e.g. $1.45!$). We mathematically drop the factorial denominator since it acts as a normalization constant that is independent of our $\alpha, \beta, \gamma$ parameters.

The resulting continuous log-likelihood function for a single match becomes:
$$ -LL = \lambda - \hat{x}\ln(\lambda) + \mu - \hat{y}\ln(\mu) - \ln(\tau_{\rho}) $$

*   $\hat{x}$: Historical Home xG
*   $\hat{y}$: Historical Away xG
*   $\lambda = \alpha_i \cdot \beta_j \cdot \gamma$
*   $\mu = \alpha_j \cdot \beta_i$
*   $\tau_{\rho}$: The Dixon-Coles low-score interdependence correction factor.

### The $\rho$ (Rho) Factor
We specifically train the $\rho$ factor on **actual integer goals**, not xG. This is because the inflation of 0-0 and 1-1 draws is a psychological and tactical phenomenon (teams protecting a draw) rather than a pure statistical artifact of xG chance.

### The Normalization Constraint
To prevent the Scipy SLSQP optimizer from scaling parameters to infinity (e.g. multiplying Attack by 10 and dividing Defense by 10 yields the exact same expected goals), we enforce a strict mean constraint:
$$ \frac{1}{N} \sum_{i=1}^{N} \alpha_i = 1.0 $$
This grounds an $\alpha$ of `1.0` as the exact statistical average attack in a given league.

### The Time-Decay Weighting
Every match's LL is multiplied by a time-decay weight $w_m$ before summation:
$$ w_m = e^{-\xi \cdot t_m} $$
Where $t_m$ is days since the match, ensuring recent form dictates current odds.

---

## 2. Lineup Likelihood Adjustments (Module 2)
Missing a star winger does not drop a team's win probability linearly. We use Bayesian adjustments to dampen lineup shocks.

1.  **Calculate Deltas:** $\Delta_{att} = \frac{\text{Matchday Attack XI Rating}}{\text{Ideal Attack XI Rating}}$
2.  **Dampen with Hyperparameters:**
    $$ \alpha'_{i} = \alpha_i \times (\Delta_{att})^{\lambda_a} $$
    $$ \beta'_{i} = \beta_i \times (1 / \Delta_{def})^{\lambda_d} $$
Where $\lambda_a$ and $\lambda_d$ are learned scalars (e.g., 0.65) that prevent the model from assuming an absent star drops team output to zero.

---

## 3. In-Game Posterior (Module 3)
A major flaw in live models is the **"Ghost xG Bias"**. If a team is leading 1-0 at 65', they will naturally drop into a low block, conceding territory. The trailing team will rack up live xG. A naive model sees this and assumes the trailing team is dominating and likely to score.

V4 defeats this by applying Game-State scaling factors ($\omega$).
*   If trailing: $\omega \approx 1.35$ (tactical urgency).
*   If leading: $\omega \approx 0.75$ (low block preservation).

We calculate the live performance modifier ($\delta_{\text{live}}$) by dividing the actual live xG by the **game-state adjusted expectation**, and use that final factor to project the $\lambda_{rem}$ (remainder lambda) into the Bivariate Poisson matrix for the final minutes of the match. 

### Infinite Tail Capture
When plotting the final $9 \times 9$ probability matrix, the final index (`[8]`) uses the Poisson Survival Function (`poisson.sf`) to capture the infinite tail (all probabilities of scoring 8 *or more* goals). This ensures the matrix contains exactly 100% of the probability mass before being converted into Fair Decimal Odds.