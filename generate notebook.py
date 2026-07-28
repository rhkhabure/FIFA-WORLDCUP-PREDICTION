import json
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# World Cup 2026 Post-Mortem Evaluation\n",
    "\n",
    "This notebook runs a suite of 7 tests on the `football_v2` live win probability model against the completed 2026 World Cup data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import json\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, classification_report\n",
    "\n",
    "# Add parent directory to import common\n",
    "sys.path.append(str(Path.cwd().parent))\n",
    "import common as c\n",
    "\n",
    "# Set up plotting\n",
    "plt.style.use('ggplot')\n",
    "pd.set_option('display.float_format', '{:.3f}'.format)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Loading & Setup\n",
    "Load the model and parse the finished matches from the local cache/raw files."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load model\n",
    "model, scaler, T = c.load_model()\n",
    "\n",
    "# Load teams and games\n",
    "# We'll use the raw data in notebooks/data/raw/wc26_games.json directly if worldcup26.ir is offline\n",
    "raw_games_path = Path('data/raw/wc26_games.json')\n",
    "raw_teams_path = Path('data/raw/wc26_teams.json')\n",
    "\n",
    "try:\n",
    "    with open(raw_games_path) as f:\n",
    "        games = json.load(f).get('games', [])\n",
    "    with open(raw_teams_path) as f:\n",
    "        teams = json.load(f).get('teams', [])\n",
    "except Exception as e:\n",
    "    print(f\"Warning: {e}. Falling back to fetch_wc26 if possible.\")\n",
    "    games = c.fetch_wc26('games')\n",
    "    teams = c.fetch_wc26('teams')\n",
    "\n",
    "team_lookup = c.build_team_lookup(teams)\n",
    "finished_games = [g for g in games if str(g.get(\"finished\", \"\")).upper() == \"TRUE\"]\n",
    "print(f\"Found {len(finished_games)} finished matches out of {len(games)} total.\")\n",
    "\n",
    "def determine_outcome(g):\n",
    "    hs, as_ = c.safe_int(g.get(\"home_score\")), c.safe_int(g.get(\"away_score\"))\n",
    "    if hs > as_: return \"home\"\n",
    "    if hs < as_: return \"away\"\n",
    "    \n",
    "    # Check penalties if knockout draw\n",
    "    hp = c.safe_int(g.get(\"home_penalty_score\", \"0\"))\n",
    "    ap = c.safe_int(g.get(\"away_penalty_score\", \"0\"))\n",
    "    if hp > ap: return \"home\"\n",
    "    if hp < ap: return \"away\"\n",
    "    return \"draw\"\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Generating Predictions\n",
    "We calculate the full timeline for each match using `c.build_match_timeline`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "timelines = {}\n",
    "for g in finished_games:\n",
    "    match_id = str(g[\"id\"])\n",
    "    timelines[match_id] = c.build_match_timeline(g, team_lookup, model, scaler, T)\n",
    "\n",
    "print(f\"Generated timelines for {len(timelines)} matches.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Pre-Game Predictions Extraction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pre_game_preds = []\n",
    "for g in finished_games:\n",
    "    match_id = str(g[\"id\"])\n",
    "    t = timelines[match_id]\n",
    "    # pre-game is the first element (minute 0)\n",
    "    kickoff = t[0]\n",
    "    \n",
    "    actual = determine_outcome(g)\n",
    "    pred_probs = {\"away\": kickoff[\"p_away\"], \"draw\": kickoff[\"p_draw\"], \"home\": kickoff[\"p_home\"]}\n",
    "    predicted_class = max(pred_probs, key=pred_probs.get)\n",
    "    \n",
    "    pre_game_preds.append({\n",
    "        \"match_id\": match_id,\n",
    "        \"stage\": g.get(\"type\"),\n",
    "        \"is_knockout\": g.get(\"type\") != \"group\",\n",
    "        \"p_home\": kickoff[\"p_home\"],\n",
    "        \"p_draw\": kickoff[\"p_draw\"],\n",
    "        \"p_away\": kickoff[\"p_away\"],\n",
    "        \"predicted\": predicted_class,\n",
    "        \"confidence\": pred_probs[predicted_class],\n",
    "        \"actual\": actual,\n",
    "        \"correct\": predicted_class == actual\n",
    "    })\n",
    "\n",
    "df_pre = pd.DataFrame(pre_game_preds)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Test 1: Group stage accuracy (3-way: win / draw / loss)\n",
    "For every group-stage match, pull the pre-game prediction and check whether the model's top pick matches the real result."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "group_df = df_pre[~df_pre['is_knockout']]\n",
    "group_acc = accuracy_score(group_df['actual'], group_df['predicted'])\n",
    "print(f\"Group Stage Kickoff Accuracy: {group_acc:.3f} ({group_df['correct'].sum()}/{len(group_df)})\")\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(group_df['actual'], group_df['predicted'], zero_division=0))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Test 2: Knockout stage accuracy (2-way: who advances)\n",
    "Same idea, but graded separately since knockouts suppress the draw probability."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "ko_df = df_pre[df_pre['is_knockout']]\n",
    "if len(ko_df) > 0:\n",
    "    ko_acc = accuracy_score(ko_df['actual'], ko_df['predicted'])\n",
    "    print(f\"Knockout Stage Kickoff Accuracy: {ko_acc:.3f} ({ko_df['correct'].sum()}/{len(ko_df)})\")\n",
    "else:\n",
    "    print(\"No knockout games found in the finished dataset yet.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Test 3: Confidence Calibration\n",
    "Bucket every prediction by how confident the model was and check if higher buckets were right more often."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_pre['conf_bucket'] = pd.cut(df_pre['confidence'], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])\n",
    "calib = df_pre.groupby('conf_bucket')['correct'].agg(['mean', 'count'])\n",
    "calib.columns = ['Accuracy', 'N']\n",
    "print(\"Calibration Table (Kickoff Predictions):\")\n",
    "display(calib)\n",
    "\n",
    "calib['Accuracy'].plot(kind='bar', title=\"Accuracy by Confidence Bucket\", ylabel=\"Accuracy\", xlabel=\"Confidence Range\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Test 4: The deeper \"how far will they go\" predictions\n",
    "Check `bracket_odds_history.csv` if available to assess multi-round forecast reliability."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "bracket_file = Path('../results/bracket_odds_history.csv')\n",
    "if bracket_file.exists():\n",
    "    df_bracket = pd.read_csv(bracket_file)\n",
    "    print(f\"Loaded {len(df_bracket)} rows from bracket_odds_history.csv\")\n",
    "    display(df_bracket.head())\n",
    "    # We would need the final tournament standings to evaluate this properly.\n",
    "    print(\"Note: Further analysis requires final tournament placements to match against these probabilities.\")\n",
    "else:\n",
    "    print(f\"File {bracket_file} not found. Skipping Test 4.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Test 5: Temporal Performance (Log-Loss by Match Minute)\n",
    "Evaluates how the model's accuracy and confidence evolve as the match progresses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_preds = []\n",
    "for g in finished_games:\n",
    "    actual = determine_outcome(g)\n",
    "    for row in timelines[str(g[\"id\"])]:\n",
    "        all_preds.append({\n",
    "            'minute': row['minute'],\n",
    "            'p_home': row['p_home'],\n",
    "            'p_draw': row['p_draw'],\n",
    "            'p_away': row['p_away'],\n",
    "            'actual': actual,\n",
    "            'actual_idx': 2 if actual == 'home' else (1 if actual == 'draw' else 0)\n",
    "        })\n",
    "\n",
    "df_timeline = pd.DataFrame(all_preds)\n",
    "\n",
    "def calculate_loss_by_minute(df):\n",
    "    minutes = sorted(df['minute'].unique())\n",
    "    losses = []\n",
    "    for m in minutes:\n",
    "        sub = df[df['minute'] == m]\n",
    "        if len(sub) == 0: continue\n",
    "        probs = sub[['p_away', 'p_draw', 'p_home']].values\n",
    "        loss = log_loss(sub['actual_idx'], probs, labels=[0,1,2])\n",
    "        losses.append({'minute': m, 'log_loss': loss})\n",
    "    return pd.DataFrame(losses)\n",
    "\n",
    "loss_df = calculate_loss_by_minute(df_timeline)\n",
    "loss_df.plot(x='minute', y='log_loss', title='Log Loss over Match Time', ylabel='Log Loss')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Test 6: Class-Specific Performance (Precision/Recall for Draws)\n",
    "Confirming if the V2 model successfully handled draws over the whole tournament timeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_timeline['predicted_idx'] = df_timeline[['p_away', 'p_draw', 'p_home']].values.argmax(axis=1)\n",
    "print(\"Global Classification Report (all minutes combined):\")\n",
    "print(classification_report(df_timeline['actual_idx'], df_timeline['predicted_idx'], target_names=['Away', 'Draw', 'Home'], zero_division=0))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Test 7: Event Responsiveness (Goal Impact)\n",
    "Look at the average probability swing immediately following a goal."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "swings = []\n",
    "for g in finished_games:\n",
    "    match_id = str(g[\"id\"])\n",
    "    t = timelines[match_id]\n",
    "    for i in range(1, len(t)):\n",
    "        prev, curr = t[i-1], t[i]\n",
    "        goal_scored = (curr['home_score'] > prev['home_score']) or (curr['away_score'] > prev['away_score'])\n",
    "        if goal_scored:\n",
    "            if curr['home_score'] > prev['home_score']:\n",
    "                swing = curr['p_home'] - prev['p_home']\n",
    "                team = \"Home\"\n",
    "            else:\n",
    "                swing = curr['p_away'] - prev['p_away']\n",
    "                team = \"Away\"\n",
    "            swings.append({'minute': curr['minute'], 'team_scored': team, 'prob_increase': swing})\n",
    "\n",
    "df_swings = pd.DataFrame(swings)\n",
    "if len(df_swings) > 0:\n",
    "    print(f\"Average Win Probability increase after scoring: {df_swings['prob_increase'].mean():.3f}\")\n",
    "    df_swings['prob_increase'].plot(kind='hist', bins=20, title='Distribution of Probability Swings upon Goal', xlabel='Probability Increase')\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"No goals recorded to calculate swings.\")\n"
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
with open('notebooks/worldcup2026_post_mortem.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Notebook generated at notebooks/worldcup2026_post_mortem.ipynb")