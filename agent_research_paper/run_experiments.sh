#!/usr/bin/env bash
set -e

# Load config
CONFIG=default_config.yaml

# Prepare results directory
mkdir -p results/plots

# Step 1: Train and prune cycles
python3 train_prune.py --config $CONFIG

# Step 2: Evaluate speaker ID and attacks
python3 evaluate_attacks.py --config $CONFIG

# Step 3: Recovery simulation
python3 simulate_recovery.py --config $CONFIG

# Step 4: Generate plots
python3 plot_metrics.py --config $CONFIG
python3 plot_roc.py --config $CONFIG

# Step 5: Save metrics
# Metrics saved within each script to results/*.csv

echo "Experiments completed. Results in results/"
