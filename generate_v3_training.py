import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# V3 Universal Football Model — Training & Calibration Loop\n",
    "\n",
    "This notebook covers the final steps of the Machine Learning pipeline:\n",
    "1. **The PyTorch Training Loop:** Feeding the continuous features and league embeddings into the network using `FocalLoss`.\n",
    "2. **Multi-Class Isotonic Calibration:** Neural networks are notoriously uncalibrated (often overconfident). We fit three separate Isotonic Regressors (Away, Draw, Home) on a validation set to map raw softmax outputs to true historical probabilities, ensuring that when the model says \"70%\", it exactly means 70%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.isotonic import IsotonicRegression\n",
    "from sklearn.metrics import brier_score_loss\n",
    "\n",
    "# Set seed for reproducibility\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Re-define the Model & Loss (From Phase 2)\n",
    "We redefine them here so this notebook is fully self-contained and runnable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class UniversalFootballNet(nn.Module):\n",
    "    def __init__(self, n_continuous=10, num_leagues=100, embed_dim=4, h1=64, h2=32, dropout=0.3):\n",
    "        super().__init__()\n",
    "        self.league_embed = nn.Embedding(num_embeddings=num_leagues, embedding_dim=embed_dim)\n",
    "        input_dim = n_continuous + embed_dim\n",
    "        \n",
    "        self.fc1 = nn.Linear(input_dim, h1)\n",
    "        self.bn1 = nn.BatchNorm1d(h1)\n",
    "        self.fc2 = nn.Linear(h1, h2)\n",
    "        self.bn2 = nn.BatchNorm1d(h2)\n",
    "        self.head = nn.Linear(h2, 3)\n",
    "        self.drop = nn.Dropout(dropout)\n",
    "        self.act = nn.ReLU()\n",
    "        \n",
    "    def forward(self, x_cont, x_league):\n",
    "        league_vec = self.league_embed(x_league)\n",
    "        x = torch.cat([x_cont, league_vec], dim=1)\n",
    "        x = self.drop(self.act(self.bn1(self.fc1(x))))\n",
    "        x = self.drop(self.act(self.bn2(self.fc2(x))))\n",
    "        return self.head(x)\n",
    "\n",
    "class FocalLoss(nn.Module):\n",
    "    def __init__(self, alpha=None, gamma=2.0):\n",
    "        super().__init__()\n",
    "        self.gamma = gamma\n",
    "        self.alpha = alpha\n",
    "        \n",
    "    def forward(self, inputs, targets):\n",
    "        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)\n",
    "        pt = torch.exp(-ce_loss)\n",
    "        focal_loss = ((1 - pt) ** self.gamma) * ce_loss\n",
    "        return focal_loss.mean()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Mock Dataset Setup\n",
    "We generate 5,000 synthetic rows of data to act as our training and validation sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class MockFootballDataset(Dataset):\n",
    "    def __init__(self, num_samples=5000):\n",
    "        # Continuous: min, home_xi, away_xi, days_h, days_a, gd, score_state, h_red, a_red, is_ko\n",
    "        self.cont = torch.rand(num_samples, 10) * 10 \n",
    "        self.league = torch.randint(0, 5, (num_samples,)) # 5 dummy leagues\n",
    "        self.targets = torch.randint(0, 3, (num_samples,)) # 0: Away, 1: Draw, 2: Home\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.targets)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        return self.cont[idx], self.league[idx], self.targets[idx]\n",
    "\n",
    "# Create Train and Validation DataLoaders\n",
    "train_data = MockFootballDataset(num_samples=4000)\n",
    "val_data = MockFootballDataset(num_samples=1000)\n",
    "\n",
    "train_loader = DataLoader(train_data, batch_size=128, shuffle=True)\n",
    "val_loader = DataLoader(val_data, batch_size=128, shuffle=False)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. The Training Loop\n",
    "A standard PyTorch training loop using the AdamW optimizer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = UniversalFootballNet(n_continuous=10, num_leagues=5)\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)\n",
    "loss_fn = FocalLoss(alpha=torch.tensor([1.0, 1.5, 1.0]), gamma=2.0) # Bonus weight to draws\n",
    "\n",
    "epochs = 3\n",
    "print(\"Starting Training Loop...\")\n",
    "for epoch in range(epochs):\n",
    "    model.train()\n",
    "    total_loss = 0\n",
    "    for batch_cont, batch_league, batch_targets in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "        logits = model(batch_cont, batch_league)\n",
    "        loss = loss_fn(logits, batch_targets)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item()\n",
    "    \n",
    "    print(f\"Epoch {epoch+1}/{epochs} | Training Focal Loss: {total_loss/len(train_loader):.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Multi-Class Isotonic Calibration\n",
    "Once the model is trained, we pass the Validation Set through it. \n",
    "We extract the raw softmax probabilities, and fit three separate `IsotonicRegression` models (one for Away, one for Draw, one for Home) against the true outcomes. \n",
    "Finally, we normalize them so they always sum perfectly to 1.0."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.eval()\n",
    "val_logits = []\n",
    "val_targets = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for batch_cont, batch_league, batch_targets in val_loader:\n",
    "        logits = model(batch_cont, batch_league)\n",
    "        val_logits.append(logits)\n",
    "        val_targets.append(batch_targets)\n",
    "        \n",
    "val_logits = torch.cat(val_logits, dim=0)\n",
    "val_targets = torch.cat(val_targets, dim=0)\n",
    "\n",
    "# Get raw softmax probabilities\n",
    "raw_probs = F.softmax(val_logits, dim=1).numpy()\n",
    "y_true = val_targets.numpy()\n",
    "\n",
    "# Fit 3 independent Isotonic Regressors\n",
    "iso_regressors = {}\n",
    "classes = [0, 1, 2] # Away, Draw, Home\n",
    "calibrated_probs = np.zeros_like(raw_probs)\n",
    "\n",
    "for c in classes:\n",
    "    iso = IsotonicRegression(out_of_bounds='clip')\n",
    "    # Binary target: 1 if this class was the true outcome, 0 otherwise\n",
    "    y_binary = (y_true == c).astype(int)\n",
    "    \n",
    "    # Fit the regressor on the raw probabilities for this class\n",
    "    calibrated_probs[:, c] = iso.fit_transform(raw_probs[:, c], y_binary)\n",
    "    iso_regressors[c] = iso\n",
    "\n",
    "# Normalize so the 3 calibrated probabilities sum to 1.0 for every row\n",
    "row_sums = calibrated_probs.sum(axis=1)[:, np.newaxis]\n",
    "# Avoid divide-by-zero just in case\n",
    "row_sums[row_sums == 0] = 1.0 \n",
    "calibrated_probs_normalized = calibrated_probs / row_sums\n",
    "\n",
    "# Show comparison on the first 5 samples\n",
    "df_compare = pd.DataFrame({\n",
    "    \"True Outcome\": y_true[:5],\n",
    "    \"Raw_Away\": raw_probs[:5, 0], \"Cal_Away\": calibrated_probs_normalized[:5, 0],\n",
    "    \"Raw_Draw\": raw_probs[:5, 1], \"Cal_Draw\": calibrated_probs_normalized[:5, 1],\n",
    "    \"Raw_Home\": raw_probs[:5, 2], \"Cal_Home\": calibrated_probs_normalized[:5, 2],\n",
    "})\n",
    "\n",
    "print(\"Calibration Complete! Sample comparison:\")\n",
    "display(df_compare.round(3))\n",
    "\n",
    "print(\"\\n✅ The Training and Calibration loop is ready for real data.\")\n"
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

with open('notebooks/phase3_v3_training_calibration.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
