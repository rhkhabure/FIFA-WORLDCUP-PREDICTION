# 📘 User Manual: Running the V4 Universal Model

This manual provides step-by-step instructions for running the complete V4 Bayesian Tri-State Model, from scraping raw historical data to serving the local web dashboard.

---

## Step 1: Environment Setup
Ensure you have Python 3.11+ installed. We highly recommend using a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn jinja2 pandas numpy scipy soccerdata requests python-dotenv
```

---

## Step 2: Build the Historical Database (Module 1 Data)
Before the model can make predictions, it needs historical context to train the Dixon-Coles attack and defense parameters.

1. Navigate to the root of the repository.
2. Execute the historical scraper:
   ```bash
   python v4_backend/historical_scraper.py
   ```
3. **What happens:** The `soccerdata` package will reach out to Understat, scrape the last 4 seasons of matches (with their Continuous Expected Goals) for the Big 5 Leagues, and save them perfectly formatted into a new file called `v4_historical_data.sqlite`.

---

## Step 3: Train the Priors (Module 1 Execution)
Now that the SQLite database exists, you must mathematically calculate the actual $\alpha$ (Attack) and $\beta$ (Defense) ratings for every team.

1. Open the execution notebook:
   ```bash
   cd v4_backend/notebooks
   jupyter notebook train_v4_priors.ipynb
   ```
2. **Action:** Click "Run All".
3. **What happens:** The notebook will loop through the `v4_historical_data.sqlite` ledger, apply the Exponential Time-Decay factor, and execute the SLSQP optimization solver. It will output the true attack parameters and save them globally in the repository as `v4_priors.json`.

*(You can also test the model's out-of-sample accuracy by running `v4_opening_weekend_eval.ipynb` immediately after).*

---

## Step 4: Run the Web Dashboard
With the mathematical engine trained and the backend data prepped, you are ready to visualize the probabilities in the beautiful FastAPI interface.

1. Navigate to the web application directory:
   ```bash
   cd v3_web
   ```
2. Launch the Uvicorn server:
   ```bash
   python -m uvicorn main:app --reload
   ```
3. Open your web browser and navigate to: **`http://localhost:8000`**

### Navigating the UI:
*   **The Sidebar:** Use the collapsible sidebar (☰) to switch between the 5 Major Leagues and select specific Team Profiles.
*   **The SVG Pitch:** On the `Live Match` and `League Hub` pages, the pitch renders dynamically. Click on any circular player token to view their unique Player Profile and Radar Chart!
*   **Tournament Simulator:** Click the "Tournament" tab to run the visual Monte Carlo knockout bracket.