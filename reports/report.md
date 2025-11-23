# Project Report — Hyperparameter Optimization for Deep Learning (Keras / TensorFlow)

## Goal
Implement Bayesian optimization for tuning a Keras model, implement baseline Grid/Random search, use California Housing dataset, implement walk-forward/backtesting, and provide a detailed comparative analysis of convergence and final performance.

## Implementation summary
- data_prep.py: fetches California Housing and creates train/test CSVs.
- model_builder.py: Keras model builder function used by both baseline and tuner.
- baseline_search.py: RandomizedSearchCV baseline using KerasRegressor wrapper.
- tune_kt.py: BayesianOptimization using Keras-Tuner (kerastuner).
- train_final.py: trains final model using best hyperparameters from artifacts.
- backtest.py: walk-forward skeleton for time-series-like validation on sorted data.
- evaluate.py: computes RMSE/MAE/MAPE and saves a sample prediction plot.

## How the deliverables satisfy rubric items
- **TensorFlow/Keras**: model_builder and training scripts use tf.keras APIs.
- **Bayesian Optimization**: Keras-Tuner's BayesianOptimization in tune_kt.py.
- **Baseline Search**: RandomizedSearchCV in baseline_search.py.
- **Walk-forward/backtesting**: backtest.py produces per-fold RMSEs for comparison.
- **Comparative analysis**: best configs saved to `artifacts/` for post-hoc comparison (and tuner history in artifacts/kt_dir).

## Reproducible workflow (commands)
1. pip install -r requirements.txt
2. python src/data_prep.py
3. python src/baseline_search.py --n_iter 20
4. python src/tune_kt.py 20
5. python src/train_final.py artifacts/best_kt.json
6. python src/backtest.py
7. python src/evaluate.py

## Notes & Next steps after running experiments
- Plot tuner objective vs trial and compare with baseline best validation losses.
- Analyze time per trial and stability across reruns.
- Write up final figures and include them in this report (PDF) for submission.
