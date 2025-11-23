Advanced DL Hyperparameter Optimization Project
==================================================

Objective
---------
Implement and compare baseline search (Grid/Random) vs Bayesian optimization for tuning a deep learning model (TensorFlow/Keras) on a tabular dataset (California Housing). Provide reproducible scripts, walk-forward/backtesting skeleton, evaluation, and a report with results and analysis.

Contents
--------
- data/: dataset CSV (produced by data_prep.py)
- src/: scripts (data_prep.py, model_builder.py, tune_kt.py, baseline_search.py, train_final.py, evaluate.py, backtest.py)
- notebooks/: starter notebook to run experiments interactively
- reports/: final report and screenshots
- artifacts/: where models/checkpoints and tuner results will be saved

How to run (example):
1. Create environment and install dependencies:
   pip install -r requirements.txt
2. Prepare data:
   python src/data_prep.py
3. Run baseline search (random/grid):
   python src/baseline_search.py --n_iter 20
4. Run Bayesian optimization (Keras Tuner):
   python src/tune_kt.py --max_trials 20 --executions_per_trial 1
5. Train final model using best hyperparameters:
   python src/train_final.py --config artifacts/best_config.json
6. Evaluate / Backtest:
   python src/evaluate.py --model artifacts/final_model.h5

Notes:
- TensorFlow and Keras-Tuner are NOT preinstalled in this environment; scripts are ready-to-run locally.
- The report contains methodology, parameter search spaces, and evaluation guidance.
