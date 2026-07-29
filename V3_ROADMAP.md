# V3 Universal Football Model — Roadmap

## 1. Core Model Upgrades (The "Universal" Model)
Move from a tournament-specific model to a global, multi-league architecture.
* **League Embeddings:** Feed `league_id` into the model so it learns the varying dynamics of different leagues (e.g., 1-0 in Serie A vs. 1-0 in the Premier League).
* **Starting XI Value (Solving the Lineup Problem):** Instead of individual player IDs, calculate the total market value (or FIFA rating) of the starting 11 players. This gives the model a dynamic, pre-match strength metric (`home_starting_xi_value`, `away_starting_xi_value`) to account for rotated squads and injuries.
* **Fixture Congestion:** Add `days_since_last_match` to account for fatigue.
* **Expanded Architecture:** Widen the neural network (e.g., 64 → 32 hidden layers) to absorb the complexity of club football.
* **Continuous Play:** Treat knockout matches as normal 90-minute games in the base model. Handle "To Advance" probabilities (Extra Time / Penalties) with a secondary logic module.

## 2. Data Strategy
* **Soft Launch Target:** The "Big 5" European Leagues (EPL, La Liga, Serie A, Bundesliga, Ligue 1).
* **Live API Upgrade:** Migrate to a professional sports data provider (e.g., API-Football via RapidAPI) to access live formations, lineups, referees, stadiums, and xG data.

## 3. UI & Frontend Vision (Breaking the "Streamlit Slop" Mold)
* **Dynamic Theming:** Pages adapt their color palette to the selected league or team (e.g., Sky Blue for Man City).
* **The Match View:** A visual football pitch (via custom SVG/HTML injection) plotting the team's formation (e.g., 4-3-3), showing jersey icons and player names.
* **Fallback Logic:** Display projected/last-known lineups if live lineups are not yet announced.
* **Team Profiles:** Dedicated team pages displaying match history, upcoming fixtures, the manager, and the roster.
* **Tournament Brackets:** An animated bracket simulator with step-by-step visual progression.
* **Architecture:** Stick to Streamlit for velocity, utilizing `unsafe_allow_html=True` to inject AI-generated custom CSS/HTML to achieve the premium look without requiring the developer to learn frontend languages.

## 4. Backend Logging
* Implement structured logging (SQLite/CSV) for all Monte Carlo bracket simulations to track how a team's tournament odds fluctuate week-by-week.
