import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# V4 Prior Optimization: Training the Dixon-Coles Models\n",
    "\n",
    "This notebook connects to the `v4_historical_data.sqlite` database you just populated with `soccerdata`.\n",
    "It splits the massive 7,171-match database by League, and feeds each league's history into our custom `continuous_dixon_coles_nll` Maximum Likelihood Estimator to calculate the optimal Attack ($\\alpha$) and Defense ($\\beta$) parameters for every team."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sqlite3\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json\n",
    "from pathlib import Path\n",
    "import sys\n",
    "\n",
    "# Ensure we can import from the backend directory regardless of cwd\n",
    "sys.path.append(str(Path.cwd().parent))\n",
    "sys.path.append(str(Path.cwd() / 'v4_backend'))\n",
    "sys.path.append(str(Path.cwd().parent.parent / 'v4_backend'))\n",
    "\n",
    "from dixon_coles_xg import train_dixon_coles_prior\n",
    "\n",
    "possible_db_paths = [\n",
    "    Path(\"../../v4_historical_data.sqlite\"), # If running from v4_backend/notebooks/\n",
    "    Path(\"v4_historical_data.sqlite\"),       # If running from root\n",
    "    Path(\"../v4_historical_data.sqlite\")     # If running from v4_backend/\n",
    "]\n",
    "\n",
    "DB_PATH = next((p for p in possible_db_paths if p.exists()), None)\n",
    "if not DB_PATH:\n",
    "    print(\"❌ Could not find v4_historical_data.sqlite in expected locations.\")\n",
    "else:\n",
    "    print(f\"✅ Found database at: {DB_PATH.resolve()}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load and Prep the Historical Ledger\n",
    "We need to calculate `days_ago` to apply the exponential time-decay weighting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    conn = sqlite3.connect(DB_PATH)\n",
    "    df = pd.read_sql_query(\"SELECT * FROM matches_xg\", conn)\n",
    "    conn.close()\n",
    "    print(f\"✅ Successfully loaded {len(df)} matches from SQLite database.\")\n",
    "except Exception as e:\n",
    "    print(f\"❌ Failed to load database: {e}\")\n",
    "    df = pd.DataFrame()\n",
    "\n",
    "if not df.empty:\n",
    "    # Convert date strings to datetime objects\n",
    "    df['date'] = pd.to_datetime(df['date'], errors='coerce')\n",
    "    \n",
    "    # Drop any rows where dates failed to parse\n",
    "    df = df.dropna(subset=['date'])\n",
    "    \n",
    "    # Calculate days_ago for the time decay function\n",
    "    # We use the most recent match in the dataset as \"today\"\n",
    "    max_date = df['date'].max()\n",
    "    df['days_ago'] = (max_date - df['date']).dt.days\n",
    "    \n",
    "    print(f\"Data spans from {df['date'].min().date()} to {max_date.date()}.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Optimize the Parameters per League\n",
    "We loop through the 5 Big Leagues, dynamically mapping the team names to numeric indices so the Scipy SLSQP optimizer can vector-map them efficiently."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "v4_priors = {}\n",
    "\n",
    "if not df.empty:\n",
    "    leagues = df['league'].unique()\n",
    "    \n",
    "    for lg in leagues:\n",
    "        print(f\"\\n{'='*50}\")\n",
    "        print(f\"📈 Optimizing Parameters for: {lg}\")\n",
    "        print(f\"{'='*50}\")\n",
    "        \n",
    "        # Filter matches for this specific league\n",
    "        df_lg = df[df['league'] == lg].copy()\n",
    "        \n",
    "        # Get unique teams in this league\n",
    "        teams_list = sorted(list(set(df_lg['home_team'].unique()) | set(df_lg['away_team'].unique())))\n",
    "        \n",
    "        # Map string team names to integer indices (0 to N)\n",
    "        team_to_idx = {team: idx for idx, team in enumerate(teams_list)}\n",
    "        df_lg['home_idx'] = df_lg['home_team'].map(team_to_idx)\n",
    "        df_lg['away_idx'] = df_lg['away_team'].map(team_to_idx)\n",
    "        \n",
    "        # Rename columns to match what `continuous_dixon_coles_nll` expects\n",
    "        df_lg = df_lg.rename(columns={\n",
    "            'home_xg': 'obs_xg_h',\n",
    "            'away_xg': 'obs_xg_a',\n",
    "            'home_goals': 'act_g_h',\n",
    "            'away_goals': 'act_g_a'\n",
    "        })\n",
    "        \n",
    "        try:\n",
    "            # Run the SLSQP optimizer!\n",
    "            prior_dict, meta_params = train_dixon_coles_prior(df_lg, teams_list)\n",
    "            \n",
    "            v4_priors[lg] = {\n",
    "                \"teams\": prior_dict,\n",
    "                \"meta\": meta_params\n",
    "            }\n",
    "            \n",
    "            print(f\"\\n✅ {lg} Optimization Complete!\")\n",
    "            print(f\"   Home Advantage (gamma): {meta_params['gamma_home_advantage']:.3f}\")\n",
    "            print(f\"   Draw Correction (rho): {meta_params['rho_draw_correction']:.3f}\")\n",
    "            \n",
    "            # Let's peek at the top 3 and bottom 3 attacking teams\n",
    "            sorted_attacks = sorted(prior_dict.items(), key=lambda x: x[1]['alpha'], reverse=True)\n",
    "            print(f\"\\n   Top 3 Attacks: {sorted_attacks[0][0]} ({sorted_attacks[0][1]['alpha']:.2f}), {sorted_attacks[1][0]} ({sorted_attacks[1][1]['alpha']:.2f}), {sorted_attacks[2][0]} ({sorted_attacks[2][1]['alpha']:.2f})\")\n",
    "            print(f\"   Bot 3 Attacks: {sorted_attacks[-1][0]} ({sorted_attacks[-1][1]['alpha']:.2f}), {sorted_attacks[-2][0]} ({sorted_attacks[-2][1]['alpha']:.2f}), {sorted_attacks[-3][0]} ({sorted_attacks[-3][1]['alpha']:.2f})\")\n",
    "            \n",
    "        except Exception as e:\n",
    "            print(f\"\\n❌ Optimization failed for {lg}: {e}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Save the V4 Priors\n",
    "We save the finalized mathematically-pure parameters to a JSON file so they can be injected into the UI dashboard or a real-time betting script."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if v4_priors:\n",
    "    out_path = Path(\"../v4_priors.json\")\n",
    "    with open(out_path, \"w\") as f:\n",
    "        json.dump(v4_priors, f, indent=2)\n",
    "    print(f\"\\n💾 All V4 Prior parameters successfully exported to {out_path}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('v4_backend/notebooks/train_v4_priors.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
